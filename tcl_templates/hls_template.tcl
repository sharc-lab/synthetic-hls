open_project -reset hls_prj
set_top [top_function_name]
add_files src/[kernel_name].cpp
add_files src/[kernel_name].h
open_solution -reset solution
set_part xczu9eg-ffvb1156-2-i
create_clock -period 10 -name default
source opt.tcl
csynth_design
close_project
