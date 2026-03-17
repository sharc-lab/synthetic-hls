PROGRAML_BIN="/usr/scratch/mzhou346/miniforge/envs/synthetic_hls_test/lib/python3.12/site-packages/programl/bin"
XILINX_INCLUDE_DIR=/tools/software/xilinx/ARCHIVE/Vitis_HLS/2023.1/include
export PATH="$PROGRAML_BIN:$PATH"

cd $2
SRC_DIR=src
if [ $3 == 'multi_modality' ]
then
    clang-10 -emit-llvm -fno-discard-value-names -g -S -c -I "$SRC_DIR" -I "$XILINX_INCLUDE_DIR" $(ls "$SRC_DIR"/*.cpp) -o $1.ll
else
    clang-10 -emit-llvm -fno-discard-value-names -S -c -I "$SRC_DIR" -I "$XILINX_INCLUDE_DIR" $(ls "$SRC_DIR"/*.cpp) -o $1.ll
fi
llvm2graph-10 < $1.ll > $1.pbtxt
graph2json < $1.pbtxt > $1.json