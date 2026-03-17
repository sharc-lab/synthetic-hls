######################################################################################
### generates the graph to be processed with HARP                                  ###
### requires: ProGraML [ICML'21]                                                   ###
######################################################################################

import os
import networkx as nx
import json
import shutil
from os.path import join, abspath, basename, exists, dirname, isfile
from subprocess import Popen, PIPE
from collections import OrderedDict
from copy import deepcopy
import ast
from pprint import pprint
from glob import glob
import csv
import re
# import programl

from utils import create_dir_if_not_exists, get_root_path, natural_keys

from collections import defaultdict
import shlex

PRAGMA_POSITION = {'PIPELINE': 0, 'TILE': 2, 'PARALLEL': 1}

type_graph = 'harp'


# ======================================================================================
# Graph IR node/edge wrappers (kept consistent with original schema)
# ======================================================================================

class Node():
    def __init__(self, block, function, text, type_n, features=None):
        self.block: int = block
        self.function: int = function
        self.text: str = text
        self.type_n: int = type_n  # 0: instr, 1: var, 2: imm, 3: pragma, 4: pseudo block
        self.features: str = features

    def get_attr(self, after_process=True):
        n_dict = {}
        n_dict['block'] = self.block
        n_dict['function'] = self.function
        n_dict['text'] = self.text
        n_dict['type'] = self.type_n
        if after_process:
            n_dict['full_text'] = self.features
        else:
            n_dict['features'] = {'full_text': [self.features]}
        return n_dict


class Edge():
    def __init__(self, src, dst, flow, position):
        self.src: int = src
        self.dst: int = dst
        self.flow: int = flow  # 0 control, 1 data, 2 call, 3 pragma, 4 block, 5 block-chain, 6 loop-hier
        self.position: int = position

    def get_attr(self):
        e_dict = {}
        e_dict['flow'] = self.flow
        e_dict['position'] = self.position
        return e_dict


def create_pseudo_node_block(block, function):
    return Node(block, function, text='pseudo_block', type_n=4, features='auxiliary node for each block')


def add_to_graph(g_nx, nodes, edges):
    if len(nodes) > 0:
        g_nx.add_nodes_from(nodes)
    if len(edges) > 0:
        g_nx.add_edges_from(edges)


# ======================================================================================
# ProGraML conversion
# ======================================================================================
def read_json_graph(name, readable=False):
    '''
        reads a graph in json format as a netwrokx graph
        args:
            name: name of the json file/ kernel's name
            readable: whether to store a readable format of the json file
        returns:
            g_nx: graph in networkx format
    '''
    filename = name + '.json'
    with open(filename) as f:
        js_graph = json.load(f)

    if "links" in js_graph:
        edge_key = "links"
    elif "edges" in js_graph:
        edge_key = "edges"
    else:
        raise KeyError(
            f"Invalid node-link JSON: missing 'links'/'edges' in {filename}. "
            f"Keys found: {list(js_graph.keys())}"
        )

    try:
        g_nx = nx.readwrite.json_graph.node_link_graph(js_graph, link=edge_key)
    except TypeError:
        g_nx = nx.readwrite.json_graph.node_link_graph(js_graph, edges=edge_key)

    return g_nx


def llvm_to_nx(name):
    """
    We keep the same call site, but use JSON graph as the source.
    """
    return read_json_graph(name, readable=False)


# ======================================================================================
# Source parsing helpers
# ======================================================================================

def extract_function_names(c_code):
    pattern = r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
    function_matches = re.finditer(pattern, c_code)
    function_names = []
    for match in function_matches:
        function_name = match.group().split()[1]
        line_number = c_code.count('\n', 0, match.start()) + 1
        function_names.append((function_name.split('(')[0], line_number))
    return function_names


def get_tc_for_loop(for_loop_text):
    comp = for_loop_text.split(';')[1].strip()
    delims = ['<=', '>=', '<', '>', '--']  # FIXME: support for other condition types
    delim = None
    for d in delims:
        if d in comp:
            delim = d
            break
    if not delim:
        raise RuntimeError(f'no comparison sign found in {for_loop_text}')

    if delim == '--':
        return 0

    rhs = comp.replace(" ", "").split(delim)[-1].strip()

    if not re.fullmatch(r"[0-9]+", rhs):
        return None

    return int(rhs)


def get_icmp(path, name, log=False):
    for_dict_llvm = OrderedDict()
    with open(join(path, f'{name}.ll'), 'r') as f_llvm:
        lines_llvm = f_llvm.readlines()

    for_count_llvm, local_for_count_llvm = 0, 0
    func_inst = None
    for idx, line in enumerate(lines_llvm):
        if line.strip().startswith('define'):
            for_dict_llvm[line.strip()] = OrderedDict()
            func_inst = line.strip()
            local_for_count_llvm = 0
        # elif line.strip().startswith('for.cond'):
        elif re.match(r'^for\.cond(\d+)?\s*:', line.strip()):
            for_count_llvm += 1
            local_for_count_llvm += 1
            for idx2, line2 in enumerate(lines_llvm[idx+1:]):
                if line2.strip().startswith('for.body'):
                    raise RuntimeError(f'No icmp found after for.cond at line {idx}')
                elif 'icmp' in line2.strip():
                    assert func_inst is not None
                    for_dict_llvm[func_inst][local_for_count_llvm] = [line2.strip(), idx, idx2 + idx + 1]
                    break

    if log:
        print(json.dumps(for_dict_llvm, indent=4))
    return for_dict_llvm, for_count_llvm


# ======================================================================================
# OptDSLv2: parse opt.tcl and map to (function,label)
# ======================================================================================


def parse_optdsl_tcl(tcl_path, log=False):
    """
    Parse OptDSLv2-produced Tcl (opt.tcl).

    Returns:
        loop_dirs: dict[(func, label)] -> list[dict] for loop directives
        array_dirs: dict[str func] -> list[dict] for array_partition directives
    Conventions assumed:
        - pipeline/unroll target: <func>/<label>
        - array_partition target: ... <func> <var>
    """
    loop_dirs = defaultdict(list)
    array_dirs = defaultdict(list)

    if not isfile(tcl_path):
        raise FileNotFoundError(f"OptDSL tcl file not found: {tcl_path}")

    with open(tcl_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            toks = shlex.split(s)
            if not toks:
                continue

            cmd = toks[0]

            if cmd == "set_directive_pipeline":
                target = toks[-1]
                if "/" not in target:
                    raise RuntimeError(f"Unexpected pipeline target format: {target} in line: {s}")
                func, label = target.split("/", 1)
                loop_dirs[(func, label)].append({"kind": "PIPELINE", "factor": None, "raw": s})

            elif cmd == "set_directive_unroll":
                factor = None
                if "-factor" in toks:
                    factor = int(toks[toks.index("-factor") + 1])
                target = toks[-1]
                if "/" not in target:
                    raise RuntimeError(f"Unexpected unroll target format: {target} in line: {s}")
                func, label = target.split("/", 1)
                loop_dirs[(func, label)].append({"kind": "UNROLL", "factor": factor, "raw": s})

            elif cmd == "set_directive_array_partition":
                # OptDSLv2 convention: ... <func> <var>
                if len(toks) < 3:
                    raise RuntimeError(f"Malformed array_partition directive line: {s}")

                ap_type = None
                ap_factor = None
                ap_dim = None
                if "-type" in toks:
                    ap_type = toks[toks.index("-type") + 1]
                if "-factor" in toks:
                    ap_factor = int(toks[toks.index("-factor") + 1])
                if "-dim" in toks:
                    ap_dim = int(toks[toks.index("-dim") + 1])

                func = toks[-2]
                var = toks[-1]
                array_dirs[func].append({
                    "kind": "ARRAY_PARTITION",
                    "ptype": ap_type,
                    "factor": ap_factor,
                    "dim": ap_dim,
                    "var": var,
                    "raw": s,
                })

            else:
                continue

    if log:
        pprint({"loop_dirs": dict(loop_dirs), "array_dirs": dict(array_dirs)})
    return loop_dirs, array_dirs

def get_single_cpp_under_src(design_dir):
    src_dir = join(design_dir, "src")
    if not exists(src_dir):
        raise FileNotFoundError(f"Expected src/ under design dir: {design_dir}")

    cpps = sorted(glob(join(src_dir, "*.cpp")))
    if len(cpps) == 0:
        raise FileNotFoundError(f"No .cpp found under: {src_dir}")
    if len(cpps) > 1:
        raise RuntimeError(f"Multiple .cpp under {src_dir}, cannot choose uniquely: {cpps}")
    return cpps[0]


def get_labeled_loops_from_cpp(cpp_path, log=False):
    """
    Enumerate for-loops in each function in encounter order, capturing optional C labels.
    Same logic as before, but reads from an explicit cpp file path.
    """
    loops_by_func = OrderedDict()

    with open(cpp_path, "r") as f:
        lines = f.readlines()

    with open(cpp_path, "r") as f:
        function_names_list = extract_function_names(f.read())

    for_count_source = 0

    for f_id, (f_name, idx_start) in enumerate(function_names_list):
        loops_by_func[f_name] = []
        last_line = function_names_list[f_id+1][1] if f_id + 1 < len(function_names_list) else len(lines)

        pending_label = None
        for idx in range(idx_start, last_line):
            s = lines[idx].strip()
            if not s or 'scop' in s:
                continue

            m = re.match(r"^([A-Za-z_]\w*)\s*:\s*(.*)$", s)
            if m:
                pending_label = m.group(1)
                rest = m.group(2).strip()
                if rest.startswith("for"):
                    loops_by_func[f_name].append({"label": pending_label, "for_text": rest.strip("{")})
                    pending_label = None
                    for_count_source += 1
                continue

            if s.startswith("for(") or s.startswith("for "):
                loops_by_func[f_name].append({"label": pending_label, "for_text": s.strip("{")})
                pending_label = None
                for_count_source += 1

    if log:
        pprint(loops_by_func)

    return loops_by_func, for_count_source


# ======================================================================================
# Pragma overlay (same node/edge schema as original)
# ======================================================================================

### [MOD] helper: only accept *pure integer* literals; reject %regs / symbols.
def _safe_parse_int_literal(token: str):
    token = token.strip()
    # strip common trailing tokens (rare, but safe)
    token = token.strip(")")

    # accept only pure decimal integer (optionally negative)
    if re.fullmatch(r"-?\d+", token):
        return int(token)

    return None



def create_pragma_nodes(g_nx, g_nx_nodes, for_dict_source, for_dict_llvm, array_dirs=None, log=False):
    """
    Add pragma nodes (type=3) and pragma edges (flow=3) to the ProGraML graph.

    - Loop directives (PIPELINE/UNROLL) are attached to loop anchor (icmp) nodes.
    - ARRAY_PARTITION directives are array-scoped:
        *If pseudo_block nodes exist* (auxiliary graphs), attach ARRAY_PARTITION only to
        pseudo_block nodes (type=4, text='pseudo_block') corresponding to blocks that use
        the array (best-effort via name match in full_text / var nodes).
        Otherwise (initial graphs), attach to variable nodes if found, else near-use nodes
        (avoid node 0 if possible), else a stable function-local fallback.

    Schema compatibility:
      - UNROLL -> PARALLEL (original HARP vocabulary).
      - ARRAY_PARTITION -> TILE slot (position=PRAGMA_POSITION['TILE']),
        while full_text/ap_* preserve semantics.
    """
    if array_dirs is None:
        array_dirs = {}

    new_nodes, new_edges = [], []
    new_node_id = g_nx_nodes

    # -------------------------
    # helpers (local)
    # -------------------------
    def _node_int(nid):
        try:
            return int(nid)
        except Exception:
            return int(str(nid))

    def _safe_get_node(nid):
        # networkx may store node ids as str in gexf/json; handle both
        if nid in g_nx.nodes:
            return g_nx.nodes[nid]
        s = str(nid)
        if s in g_nx.nodes:
            return g_nx.nodes[s]
        i = _node_int(nid)
        if i in g_nx.nodes:
            return g_nx.nodes[i]
        raise KeyError(f"node {nid} not found")

    def _stable_fallback_node_id(prefer_function_id=None):
        """Pick a stable anchor node id, preferably within a function scope, and avoid 0 if possible."""
        if len(g_nx.nodes()) == 0:
            return None
        scoped = []
        if prefer_function_id is not None:
            for nid, nd in g_nx.nodes(data=True):
                try:
                    if int(nd.get("function", -1)) == int(prefer_function_id):
                        scoped.append(_node_int(nid))
                except Exception:
                    continue
        if scoped:
            scoped = sorted(set(scoped))
            if len(scoped) > 1 and scoped[0] == 0:
                return scoped[1]
            return scoped[0]
        all_ids = sorted(_node_int(n) for n in g_nx.nodes())
        if len(all_ids) > 1 and all_ids[0] == 0:
            return all_ids[1]
        return all_ids[0]

    # build pseudo_block mapping if present: (function, block) -> pseudo_id
    pseudo_by_fb = {}
    pseudo_ids = []
    for nid, nd in g_nx.nodes(data=True):
        if str(nd.get("text", "")) == "pseudo_block" and int(nd.get("type", -1)) == 4:
            fb = (int(nd.get("function", -1)), int(nd.get("block", -1)))
            pid = _node_int(nid)
            pseudo_by_fb[fb] = pid
            pseudo_ids.append(pid)
    has_pseudo = len(pseudo_ids) > 0

    def _near_use_node_ids(var_name, prefer_function_id=None, max_hits=16):
        """Find node ids whose full_text contains var_name; prefer type==1 if available."""
        hits = []
        for nid, nd in g_nx.nodes(data=True):
            if prefer_function_id is not None and int(nd.get("function", -1)) != int(prefer_function_id):
                continue
            ft = str(nd.get("full_text", ""))
            if var_name and (var_name in ft):
                ntype = int(nd.get("type", -1))
                hits.append((0 if ntype == 1 else 1, _node_int(nid)))
        if not hits:
            return []
        ids = [_id for _prio, _id in sorted(hits)]
        # avoid 0 if possible
        ids_no0 = [i for i in ids if i != 0]
        ids = ids_no0 if ids_no0 else ids
        return ids[:max_hits]

    # -------------------------
    # 1) Loop pragmas (PIPELINE / UNROLL->PARALLEL)
    # -------------------------
    for f_name, f_content in for_dict_source.items():
        has_loop_pragmas = any(len(pragmas) > 0 for (_loop_text, pragmas) in f_content.values())
        if not has_loop_pragmas:
            continue

        # match function name to llvm define key
        mangled = [k for k in for_dict_llvm.keys() if f"{len(f_name)}{f_name}" in k]
        cand = [k for k in for_dict_llvm.keys() if f_name in k]
        llvm_key = None
        if len(mangled) == 1:
            llvm_key = mangled[0]
        elif len(cand) == 1:
            llvm_key = cand[0]
        else:
            print(f"[WARN] Function '{f_name}' has loop pragmas but cannot be uniquely matched to LLVM define; skipping loop pragmas.")
            continue

        llvm_content = for_dict_llvm.get(llvm_key, {})

        # Build loop anchors: loop_id -> (node_id, block_id, function_id)
        loop_anchors = {}
        for loop_id, (icmp_inst, _l1, _l2) in llvm_content.items():
            found = False
            for node, ndata in g_nx.nodes(data=True):
                if 'features' not in ndata:
                    continue
                feat = ast.literal_eval(str(ndata['features']))
                if icmp_inst == feat['full_text'][0]:
                    loop_anchors[loop_id] = (_node_int(node), int(ndata.get('block', 0)), int(ndata.get('function', 0)))
                    found = True
                    break
            if not found:
                continue

        for for_loop_id, (_for_loop_text, pragmas) in f_content.items():
            if len(pragmas) == 0:
                continue
            if for_loop_id not in loop_anchors:
                print(f"[WARN] Loop id {for_loop_id} not found in graph anchors for function {f_name}; skipping.")
                continue

            anchor_id, block_id, function_id = loop_anchors[for_loop_id]
            for pragma in pragmas:
                kind = pragma.get("kind")
                factor = pragma.get("factor", None)

                if kind == "UNROLL":
                    kind = "PARALLEL"

                if kind is None or kind.upper() not in PRAGMA_POSITION:
                    raise RuntimeError(f"Unsupported pragma kind '{kind}' from: {pragma}")

                full_text = pragma.get("raw", kind) if factor is None else f"{kind} factor={factor}"

                p_dict = {
                    'type': 3,
                    'block': block_id,
                    'function': function_id,
                    'features': {'full_text': [full_text]},
                    'text': kind,
                }
                if factor is not None:
                    p_dict['factor'] = int(factor)

                new_nodes.append((new_node_id, p_dict))
                e_dict = {'flow': 3, 'position': PRAGMA_POSITION[kind.upper()]}
                new_edges.append((anchor_id, new_node_id, e_dict))
                new_edges.append((new_node_id, anchor_id, e_dict))
                new_node_id += 1

    # -------------------------
    # 2) ARRAY_PARTITION pragmas (mapped to TILE slot)
    # -------------------------
    for func_name, dirs in array_dirs.items():
        if not dirs:
            continue

        # try to pick a function_id preference from any existing node full_text that contains the function name (rare)
        prefer_fid = None
        # if we have pseudo blocks, we can just use those
        for ap in dirs:
            ap_var = ap.get("var")
            ap_type = ap.get("ptype")
            ap_dim = ap.get("dim")
            ap_factor = ap.get("factor")

            full_text = f"ARRAY_PARTITION var={ap_var} type={ap_type} dim={ap_dim} factor={ap_factor}"

            # Find potential use nodes (var node exact name or near-use by full_text)
            use_ids = []
            # exact var nodes
            for nid, nd in g_nx.nodes(data=True):
                if int(nd.get('type', -1)) == 1 and str(nd.get('text', '')) == str(ap_var):
                    if prefer_fid is None or int(nd.get("function", -1)) == int(prefer_fid):
                        use_ids.append(_node_int(nid))
            if not use_ids:
                use_ids = _near_use_node_ids(str(ap_var), prefer_function_id=prefer_fid, max_hits=16)

            anchor_ids = []

            if has_pseudo:
                # ### [MOD] PSEUDO-BLOCK-ONLY ATTACHMENT:
                # Convert use nodes -> pseudo_block anchors via (function, block).
                anchors = set()
                for uid in use_ids:
                    nd = _safe_get_node(uid)
                    fb = (int(nd.get("function", -1)), int(nd.get("block", -1)))
                    pid = pseudo_by_fb.get(fb)
                    if pid is not None:
                        anchors.add(pid)

                if anchors:
                    anchor_ids = sorted(anchors)
                else:
                    # no local uses found; attach to a stable pseudo_block (avoid any direct edge to node 0)
                    anchor_ids = [min(pseudo_ids)]
            else:
                # initial graphs: attach to use nodes if any, else stable fallback (avoid 0 if possible)
                if use_ids:
                    anchor_ids = use_ids
                else:
                    fb = _stable_fallback_node_id(prefer_function_id=prefer_fid)
                    if fb is None:
                        print(f"[WARN] ARRAY_PARTITION var='{ap_var}' in function '{func_name}' but graph empty; skipping.")
                        continue
                    anchor_ids = [fb]

            # Create pragma nodes and connect ONLY to anchor_ids.
            for anchor_id in anchor_ids:
                nd = _safe_get_node(anchor_id)
                block_id = int(nd.get('block', 0))
                function_id = int(nd.get('function', 0))

                p_dict = {
                    'type': 3,
                    'block': block_id,
                    'function': function_id,
                    'features': {'full_text': [full_text]},
                    'text': 'TILE',  # reuse TILE slot for ARRAY_PARTITION
                    'ap_kind': 'ARRAY_PARTITION',
                    'ap_var': ap_var,
                    'ap_func': func_name,
                    'ap_type': ap_type,
                }
                if ap_dim is not None:
                    p_dict['ap_dim'] = int(ap_dim)
                if ap_factor is not None:
                    p_dict['ap_factor'] = int(ap_factor)

                new_nodes.append((new_node_id, p_dict))
                e_dict = {'flow': 3, 'position': PRAGMA_POSITION['TILE']}
                new_edges.append((_node_int(anchor_id), new_node_id, e_dict))
                new_edges.append((new_node_id, _node_int(anchor_id), e_dict))
                new_node_id += 1

    if log:
        pprint(new_nodes)
        pprint(new_edges)

    return new_nodes, new_edges
def prune_redundant_nodes(g_new):
    while True:
        remove_nodes = set()
        for node in list(g_new.nodes()):
            if node is None or len(list(g_new.neighbors(node))) == 0:
                remove_nodes.add(node)
        for node in remove_nodes:
            if node in g_new:
                g_new.remove_node(node)
        if not remove_nodes:
            break


def process_graph(name, g, processed_gexf_folder, csv_dict=None, design_dir=None, optdsl_tcl_file='opt.tcl'):
    g_new = nx.MultiDiGraph()
    for node, ndata in g.nodes(data=True):
        attrs = deepcopy(ndata)
        if 'features' in ndata:
            feat = ndata['features']
            attrs['full_text'] = feat['full_text'][0]
            del attrs['features']
        g_new.add_node(node)
        nx.set_node_attributes(g_new, {node: attrs})

    edge_list = []
    eid = 0
    for nid1, nid2, edata in g.edges(data=True):
        edata = dict(edata)
        edata['id'] = eid
        edge_list.append((nid1, nid2, edata))
        eid += 1
    g_new.add_edges_from(edge_list)

    prune_redundant_nodes(g_new)

    new_gexf_file = join(processed_gexf_folder, f'{name}_processed_result.gexf')

    # -------------------------
    # [FIX] Add ARRAY_PARTITION (TILE slot) pragmas ONLY at auxiliary stage
    #       so they can attach to pseudo_block anchors instead of falling back to node 0/1.
    # -------------------------
    if design_dir is not None:
        tcl_path = join(design_dir, optdsl_tcl_file)
        try:
            _loop_dirs, _array_dirs = parse_optdsl_tcl(tcl_path, log=False)
        except FileNotFoundError:
            _array_dirs = {}
        except Exception as e:
            raise

        if _array_dirs:
            # next available numeric node id
            try:
                _max_nid = max(int(str(n)) for n in g_new.nodes())
                _next_nid = _max_nid + 1
            except Exception:
                _next_nid = g_new.number_of_nodes()

            ap_nodes, ap_edges = create_pragma_nodes(
                g_new,
                _next_nid,
                for_dict_source={},
                for_dict_llvm={},
                array_dirs=_array_dirs,
                log=False,
            )

            # convert node schema to "processed" form: full_text instead of features
            ap_nodes_proc = []
            for nid, nd in ap_nodes:
                nd = dict(nd)
                if "features" in nd and isinstance(nd["features"], dict) and "full_text" in nd["features"]:
                    nd["full_text"] = nd["features"]["full_text"][0]
                    del nd["features"]
                ap_nodes_proc.append((nid, nd))

            # assign unique edge ids continuing from eid
            ap_edges_proc = []
            for u, v, ed in ap_edges:
                ed = dict(ed)
                ed["id"] = eid
                eid += 1
                ap_edges_proc.append((u, v, ed))

            add_to_graph(g_new, nodes=ap_nodes_proc, edges=ap_edges_proc)

    nx.write_gexf(g_new, new_gexf_file)

    if csv_dict is not None:
        csv_dict[name] = {
            'name': name,
            'num_node': len(g_new.nodes),
            'num_edge': len(g_new.edges),
        }


# ======================================================================================
# Graph generator (OptDSLv2-only)
# ======================================================================================

def graph_generator(
    name,
    path,
    generate_programl=False,
    csv_dict=None,
    optdsl_tcl_file="opt.tcl",
    processed_gexf_folder=None,
):
    assert processed_gexf_folder is not None

    if generate_programl:
        p = Popen(f"{get_root_path()}/clang_script.sh {name} {path} {type_graph}",
                  shell=True, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(
                f"clang_script.sh failed for {name}\n"
                f"STDOUT:\n{out.decode(errors='ignore')}\n"
                f"STDERR:\n{err.decode(errors='ignore')}\n"
            )

    g_nx = llvm_to_nx(join(path, name))
    g_nx_nodes = g_nx.number_of_nodes()

    for_dict_llvm, for_count_llvm = get_icmp(path, name)

    cpp_path = get_single_cpp_under_src(path)
    loops_by_func, for_count_source = get_labeled_loops_from_cpp(cpp_path)
    if for_count_llvm != for_count_source:
        print(
            f"[WARN] Loop count mismatch for {name}: llvm={for_count_llvm} vs source={for_count_source}. "
            f"Proceeding without strict check."
        )
    # assert for_count_llvm == for_count_source, (
    #     f'Loop count mismatch for {name}: llvm={for_count_llvm} vs source={for_count_source}'
    # )

    tcl_path = join(path, optdsl_tcl_file)
    if not isfile(tcl_path):
        raise FileNotFoundError(f"Missing opt.tcl: {tcl_path}")
    loop_dirs, array_dirs = parse_optdsl_tcl(tcl_path)

    for_dict_source = OrderedDict()
    for f_name, loops in loops_by_func.items():
        for_dict_source[f_name] = OrderedDict()
        local_loop_id = 0
        for loop in loops:
            local_loop_id += 1
            label = loop.get("label", None)
            pragma_list = []
            if label is not None:
                pragma_list.extend(loop_dirs.get((f_name, label), []))
            for_dict_source[f_name][local_loop_id] = [loop["for_text"], pragma_list]

    new_nodes, new_edges = create_pragma_nodes(
        g_nx, g_nx_nodes, for_dict_source, for_dict_llvm,
        array_dirs={},
        log=False
    )
    add_to_graph(g_nx, new_nodes, new_edges)

    process_graph(name, g_nx, processed_gexf_folder, csv_dict=csv_dict)

    global_gexf = join(processed_gexf_folder, f"{name}_processed_result.gexf")
    local_gexf = join(path, f"{name}_processed_result.gexf")

    if isfile(global_gexf):
        shutil.copy2(global_gexf, local_gexf)
    else:
        raise FileNotFoundError(f"Expected processed gexf not found: {global_gexf}")


# ======================================================================================
# Auxiliary + hierarchy stages (same semantics as original)
# ======================================================================================

def add_auxiliary_nodes(name, path, processed_path, csv_dict, design_dir=None, optdsl_tcl_file="opt.tcl", node_type='block', connected=False):
    if node_type != 'block':
        raise NotImplementedError()

    gexf_file = join(path, f'{name}_processed_result.gexf')
    new_gexf_file = join(processed_path, f'{name}_processed_result.gexf')
    if not isfile(gexf_file):
        return None

    g = nx.readwrite.gexf.read_gexf(gexf_file)
    g_nx_nodes, g_nx_edges = g.number_of_nodes(), len(g.edges)
    current_g_value = {'name': name, 'prev_node': g_nx_nodes, 'prev_edge': g_nx_edges}
    orig_nodes = g_nx_nodes

    block_nodes = {}
    new_edges = [(nid1, nid2, edata) for nid1, nid2, edata in g.edges(data=True)]
    new_nodes = [(node, ndata) for node, ndata in g.nodes(data=True)]
    block_func = {}
    max_block = 0
    g_new = nx.MultiDiGraph()
    eid = g_nx_edges

    for node, ndata in g.nodes(data=True):
        key = f"function-{ndata['function']}-block-{ndata['block']}"
        if key not in block_nodes:
            new_node = create_pseudo_node_block(ndata['block'], ndata['function'])
            block_nodes[key] = {'id': g_nx_nodes, 'node': new_node, 'last_position': 0}
            new_nodes.append((g_nx_nodes, new_node.get_attr(after_process=True)))
            g_nx_nodes += 1

        if ndata['function'] not in block_func:
            block_func[ndata['function']] = {'count': 1, 'blocks': [ndata['block']]}
        else:
            if ndata['block'] not in block_func[ndata['function']]['blocks']:
                block_func[ndata['function']]['count'] += 1
                block_func[ndata['function']]['blocks'].append(ndata['block'])

        pseudo_id = block_nodes[key]['id']
        pseudo_position = block_nodes[key]['last_position']

        e_dict = {'id': eid, 'flow': 4, 'position': pseudo_position}
        new_edges.append((node, pseudo_id, e_dict))
        eid += 1
        e_dict = {'id': eid, 'flow': 4, 'position': pseudo_position}
        new_edges.append((pseudo_id, node, e_dict))
        eid += 1

        block_nodes[key]['last_position'] = pseudo_position + 1

    if connected:
        sorted_nodes = sorted(block_nodes.keys(), key=natural_keys)
        for idx, key in enumerate(sorted_nodes[:-1]):
            id1 = block_nodes[key]['id']
            id2 = block_nodes[sorted_nodes[idx+1]]['id']
            e_dict = {'id': eid, 'flow': 5, 'position': 0}
            new_edges.append((id1, id2, e_dict))
            eid += 1
            e_dict = {'id': eid, 'flow': 5, 'position': 0}
            new_edges.append((id2, id1, e_dict))
            eid += 1

    add_to_graph(g_new, nodes=new_nodes, edges=new_edges)
    prune_redundant_nodes(g_new)

    g_nx_nodes2, g_nx_edges2 = g_new.number_of_nodes(), len(g_new.edges)
    for f, b in block_func.items():
        max_block += b['count']
    assert g_nx_nodes2 == orig_nodes + max_block

    current_g_value['new_node'] = g_nx_nodes2
    current_g_value['new_edge'] = g_nx_edges2
    current_g_value['block'] = max_block
    if csv_dict is not None:
        csv_dict[name] = current_g_value


    # -------------------------
    # [FIX] Add ARRAY_PARTITION (TILE slot) pragmas ONLY at auxiliary stage
    #       so they can attach to pseudo_block anchors instead of falling back to node 0/1.
    # -------------------------
    if design_dir is not None:
        tcl_path = join(design_dir, optdsl_tcl_file)
        try:
            _loop_dirs, _array_dirs = parse_optdsl_tcl(tcl_path, log=False)
        except FileNotFoundError:
            _array_dirs = {}
        except Exception as e:
            raise

        if _array_dirs:
            # next available numeric node id
            try:
                _max_nid = max(int(str(n)) for n in g_new.nodes())
                _next_nid = _max_nid + 1
            except Exception:
                _next_nid = g_new.number_of_nodes()

            ap_nodes, ap_edges = create_pragma_nodes(
                g_new,
                _next_nid,
                for_dict_source={},
                for_dict_llvm={},
                array_dirs=_array_dirs,
                log=False,
            )

            # convert node schema to "processed" form: full_text instead of features
            ap_nodes_proc = []
            for nid, nd in ap_nodes:
                nd = dict(nd)
                if "features" in nd and isinstance(nd["features"], dict) and "full_text" in nd["features"]:
                    nd["full_text"] = nd["features"]["full_text"][0]
                    del nd["features"]
                ap_nodes_proc.append((nid, nd))

            # assign unique edge ids continuing from eid
            ap_edges_proc = []
            for u, v, ed in ap_edges:
                ed = dict(ed)
                ed["id"] = eid
                eid += 1
                ap_edges_proc.append((u, v, ed))

            add_to_graph(g_new, nodes=ap_nodes_proc, edges=ap_edges_proc)

    nx.write_gexf(g_new, new_gexf_file)
    if design_dir is not None:
        local_gexf = join(design_dir, f"{name}_processed_result.gexf")
        if isfile(new_gexf_file):
            shutil.copy2(new_gexf_file, local_gexf)

def augment_graph_hierarchy_from_ll(name, ll_path, src_path, dst_path, csv_dict=None):
    gexf_file = join(src_path, f'{name}_processed_result.gexf')
    new_gexf_file = join(dst_path, f'{name}_processed_result.gexf')
    if not isfile(gexf_file):
        return None
    if not isfile(ll_path):
        raise FileNotFoundError(f"Missing LLVM IR: {ll_path}")

    g = nx.readwrite.gexf.read_gexf(gexf_file)
    g_nx_nodes, g_nx_edges = g.number_of_nodes(), len(g.edges)

    with open(ll_path, "r") as f_llvm:
        lines_llvm = f_llvm.readlines()

    for_blocks_info = OrderedDict()
    for_stack = []
    i = 0
    for idx, line in enumerate(lines_llvm):
        if line.startswith('for.'):
            content = line.strip().split(';')
            line0 = content[0].strip()
            if 'for.cond' in line0:
                key = f'{line0}{idx}'
                for_blocks_info[key] = {
                    'ind': i,
                    'preds': content[1] if len(content) > 1 else "",
                    'next_instr': [lines_llvm[idx+1].strip(), lines_llvm[idx+2].strip(), lines_llvm[idx+3].strip()],
                    'line_num': idx
                }
                for_stack.append(key)
                i += 1
            elif 'for.end' in line0:
                res_cond = for_stack.pop()
                for_blocks_info[res_cond]['end'] = (idx, line0)

    for_start, for_end, for_label = [], [], []
    for for_l, v in for_blocks_info.items():
        if 'cond' in for_l:
            for_start.append(v['line_num'])
            for_end.append(v['end'][0])
            for_label.append(for_l)

    for idx, start_num in enumerate(for_start):
        child_idx = idx + 1
        possible_children = []
        for s, e in zip(for_start[idx+1:], for_end[idx+1:]):
            if s > start_num and e < for_end[idx]:
                possible_children.append(for_label[child_idx])
                child_idx += 1
            else:
                break
        for_blocks_info[for_label[idx]]['possible_children'] = possible_children

    for for_l, v in for_blocks_info.items():
        possible_children = v.get('possible_children', [])
        children = []
        j = 0
        while j < len(possible_children):
            children.append(possible_children[j])
            j += len(for_blocks_info[possible_children[j]].get('possible_children', [])) + 1
        v['children'] = children

    new_edges = [(nid1, nid2, edata) for nid1, nid2, edata in g.edges(data=True)]
    new_nodes = [(node, ndata) for node, ndata in g.nodes(data=True)]
    g_new = nx.MultiDiGraph()
    eid = g_nx_edges
    current_g_value = {'name': name, 'prev_node': g_nx_nodes, 'prev_edge': g_nx_edges}

    block_ids = {}
    for for_l, v in for_blocks_info.items():
        found = False
        for node, ndata in g.nodes(data=True):
            if 'full_text' not in ndata:
                continue
            if v['next_instr'][0] in ndata['full_text']:
                block, function = (ndata['block'], ndata['function'])
                correct_node = 1
                for neighbor in g.neighbors(node):
                    if v['next_instr'][1] in g.nodes[neighbor].get('full_text', ''):
                        correct_node += 1
                    if correct_node == 2:
                        for nn in g.neighbors(neighbor):
                            if v['next_instr'][2] in g.nodes[nn].get('full_text', ''):
                                correct_node += 1
                                break
                    if correct_node == 3:
                        break
                if correct_node == 3:
                    found = True
                    block_ids[for_l] = (block, function)
                    break
        if not found:
            raise RuntimeError(f'could not find block for loop label {for_l}')

    node_ids_block = {}
    for for_l in for_blocks_info:
        for node, ndata in g.nodes(data=True):
            if 'pseudo_block' not in ndata.get('text', ''):
                continue
            if ndata['block'] == block_ids[for_l][0] and ndata['function'] == block_ids[for_l][1]:
                node_ids_block[for_l] = node
                break

    for for_l, v in for_blocks_info.items():
        if len(v['children']) == 0:
            continue
        id1 = node_ids_block[for_l]
        position = 0
        for child in v['children']:
            id2 = node_ids_block[child]
            e_dict = {'id': eid, 'flow': 6, 'position': position}
            new_edges.append((id1, id2, e_dict))
            eid += 1
            e_dict = {'id': eid, 'flow': 6, 'position': position}
            new_edges.append((id2, id1, e_dict))
            eid += 1
            position += 1

    add_to_graph(g_new, nodes=new_nodes, edges=new_edges)
    prune_redundant_nodes(g_new)

    current_g_value['new_node'] = g_new.number_of_nodes()
    current_g_value['new_edge'] = len(g_new.edges)
    if csv_dict is not None:
        csv_dict[name] = current_g_value


    # -------------------------
    # [FIX] Add ARRAY_PARTITION (TILE slot) pragmas ONLY at auxiliary stage
    #       so they can attach to pseudo_block anchors instead of falling back to node 0/1.
    # -------------------------
    if design_dir is not None:
        tcl_path = join(design_dir, optdsl_tcl_file)
        try:
            _loop_dirs, _array_dirs = parse_optdsl_tcl(tcl_path, log=False)
        except FileNotFoundError:
            _array_dirs = {}
        except Exception as e:
            raise

        if _array_dirs:
            # next available numeric node id
            try:
                _max_nid = max(int(str(n)) for n in g_new.nodes())
                _next_nid = _max_nid + 1
            except Exception:
                _next_nid = g_new.number_of_nodes()

            ap_nodes, ap_edges = create_pragma_nodes(
                g_new,
                _next_nid,
                for_dict_source={},
                for_dict_llvm={},
                array_dirs=_array_dirs,
                log=False,
            )

            # convert node schema to "processed" form: full_text instead of features
            ap_nodes_proc = []
            for nid, nd in ap_nodes:
                nd = dict(nd)
                if "features" in nd and isinstance(nd["features"], dict) and "full_text" in nd["features"]:
                    nd["full_text"] = nd["features"]["full_text"][0]
                    del nd["features"]
                ap_nodes_proc.append((nid, nd))

            # assign unique edge ids continuing from eid
            ap_edges_proc = []
            for u, v, ed in ap_edges:
                ed = dict(ed)
                ed["id"] = eid
                eid += 1
                ap_edges_proc.append((u, v, ed))

            add_to_graph(g_new, nodes=ap_nodes_proc, edges=ap_edges_proc)

    nx.write_gexf(g_new, new_gexf_file)


def write_csv_file(csv_dict, csv_header, file_path):
    with open(file_path, mode='w') as f:
        f_writer = csv.DictWriter(f, fieldnames=csv_header)
        f_writer.writeheader()
        for d, value in csv_dict.items():
            if d == 'header':
                continue
            f_writer.writerow(value)


# ======================================================================================
# OptDSLv2-only: design discovery + driver
# ======================================================================================

def discover_optdsl_design_dirs(designs_root, opt_tcl_name="opt.tcl"):
    design_dirs = []
    for dirpath, _, filenames in os.walk(designs_root):
        if opt_tcl_name in filenames:
            design_dirs.append(abspath(dirpath))
    return sorted(design_dirs)


def run_graph_gen(
    mode='initial',
    connected=True,
    designs_root=None,
    out_root=None,
    opt_tcl_name="opt.tcl",
):
    if designs_root is None or out_root is None:
        raise ValueError("Must provide designs_root and out_root")

    processed_gexf_folder = join(out_root, "processed")
    auxiliary_base_folder = join(out_root, "processed", "extended-pseudo-block-base")
    auxiliary_conn_folder = join(out_root, "processed", "extended-pseudo-block-connected")
    hierarchy_folder = join(out_root, "processed", "extended-pseudo-block-connected-hierarchy")

    create_dir_if_not_exists(processed_gexf_folder)
    create_dir_if_not_exists(auxiliary_base_folder)
    create_dir_if_not_exists(auxiliary_conn_folder)
    create_dir_if_not_exists(hierarchy_folder)

    design_dirs = discover_optdsl_design_dirs(designs_root, opt_tcl_name=opt_tcl_name)

    if mode == 'initial':
        csv_header = ['name', 'num_node', 'num_edge']
        csv_dict = {'header': csv_header}

        for ddir in design_dirs:
            name = basename(ddir)
            graph_generator(
                name=name,
                path=ddir,
                generate_programl=True,
                csv_dict=csv_dict,
                optdsl_tcl_file=opt_tcl_name,
                processed_gexf_folder=processed_gexf_folder,
            )

        write_csv_file(csv_dict, csv_header, join(out_root, 'initial.csv'))

    elif mode == 'auxiliary':
        csv_header = ['name', 'prev_node', 'prev_edge', 'new_node', 'new_edge', 'block']
        csv_dict = {'header': csv_header}

        dst_folder = auxiliary_conn_folder if connected else auxiliary_base_folder
        create_dir_if_not_exists(dst_folder)

        design_dir_by_name = {basename(d): d for d in design_dirs}

        processed_files = sorted(glob(join(processed_gexf_folder, "*_processed_result.gexf")))
        for pf in processed_files:
            name = basename(pf).replace("_processed_result.gexf", "")
            add_auxiliary_nodes(
                name,
                processed_gexf_folder,
                dst_folder,
                csv_dict=csv_dict,
                design_dir=design_dir_by_name.get(name),
                optdsl_tcl_file=opt_tcl_name,
                node_type='block',
                connected=connected,
            )

        write_csv_file(csv_dict, csv_header, join(out_root, f'auxiliary_{connected}.csv'))

    elif mode == 'hierarchy':
        csv_header = ['name', 'prev_node', 'prev_edge', 'new_node', 'new_edge']
        csv_dict = {'header': csv_header}

        src_folder = auxiliary_conn_folder
        dst_folder = hierarchy_folder
        create_dir_if_not_exists(dst_folder)

        for ddir in design_dirs:
            name = basename(ddir)
            ll_path = join(ddir, f"{name}.ll")
            augment_graph_hierarchy_from_ll(
                name=name,
                ll_path=ll_path,
                src_path=src_folder,
                dst_path=dst_folder,
                csv_dict=csv_dict,
            )

        write_csv_file(csv_dict, csv_header, join(out_root, 'hierarchy.csv'))

    else:
        raise NotImplementedError()