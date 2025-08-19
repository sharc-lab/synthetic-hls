from string import Template

from synthetic_hls.design import Design


def build_prompt_fix(current_design: Design, design_error: str) -> Template:
    raise NotImplementedError


def build_prompt_mutate_target(current_design: Design, target: str) -> Template:
    raise NotImplementedError
