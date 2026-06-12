PROGRAML_BIN="[env_path]/lib/python3.12/site-packages/programl/bin"
XILINX_INCLUDE_DIR=/tools/software/xilinx/ARCHIVE/Vitis_HLS/2023.1/include
export PATH="$PROGRAML_BIN:$PATH"

cd $2
SRC_DIR=src

if ls "$SRC_DIR"/*.cpp >/dev/null 2>&1
then
    SRC_FILE=$(ls "$SRC_DIR"/*.cpp)
elif ls "$SRC_DIR"/*.c >/dev/null 2>&1
then
    SRC_FILE=$(ls "$SRC_DIR"/*.c)
else
    echo "No .cpp or .c found under: $SRC_DIR"
    exit 1
fi

if [ $3 == 'multi_modality' ]
then
    clang-10 -emit-llvm -fno-discard-value-names -g -S -c -I "$SRC_DIR" -I "." -I "$XILINX_INCLUDE_DIR" "$SRC_FILE" -o $1.ll
else
    clang-10 -emit-llvm -fno-discard-value-names -S -c -I "$SRC_DIR" -I "." -I "$XILINX_INCLUDE_DIR" "$SRC_FILE" -o $1.ll
fi
llvm2graph-10 < $1.ll > $1.pbtxt
graph2json < $1.pbtxt > $1.json