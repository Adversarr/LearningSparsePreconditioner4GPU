# current file path
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "Working in directory: $CURRENT_DIR"


export AMGX_DIR=$CURRENT_DIR/AMGX
export AMGX_BUILD_DIR=$CURRENT_DIR/build
cmake -S $AMGX_DIR -B $AMGX_BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DCMAKE_NO_MPI=TRUE
cmake --build $AMGX_BUILD_DIR --target all -j$(nproc)

# Install pyAMGX
uv pip install --no-build-isolation $CURRENT_DIR/pyamgx