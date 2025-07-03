# cwd=third_party
third_party_dir=$(pwd)
cd AutoDMP
mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=${third_party_dir}/../AutoDMP/ -DPYTHON_EXECUTABLE=/home/zhaoxueyan/anaconda3/envs/iEDA-DSE/bin/python -DCMAKE_BUILD_TYPE=Release
make -j32 install