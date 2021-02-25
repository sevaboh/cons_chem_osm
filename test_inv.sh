#!/bin/bash
ulimit -c unlimited
function test {
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.75 10 0.5 0.5 0.5 0.1 30 $fl
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.75 20 0.5 0.5 0.5 0.1 30 $fl
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.75 30 0.5 0.5 0.5 0.1 30 $fl
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.75 40 0.5 0.5 0.5 0.1 30 $fl
}
function test2 {
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.75 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 2.0 0.75 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 3.0 0.75 10 0.5 0.5 0.5 0.1 30 4 

./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 3.0 0.5 10 0.5 0.5 0.5 0.1 30 4 

./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.25 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 2.0 0.25 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 3.0 0.25 10 0.5 0.5 0.5 0.1 30 4 

./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 1.0 0.1 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 2.0 0.1 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.8 0.01 1.0 3.0 0.1 10 0.5 0.5 0.5 0.1 30 4 

./fr_cons_chem_osm_inv_gpu 0.55 0.75 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.6 0.75 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.7 0.75 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.8 0.75 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.95 0.75 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 

./fr_cons_chem_osm_inv_gpu 0.75 0.55 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.75 0.6 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.75 0.7 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.75 0.8 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
./fr_cons_chem_osm_inv_gpu 0.75 0.95 0.01 1.0 2.0 0.5 10 0.5 0.5 0.5 0.1 30 4 
}

mpic++ -O3 -fopenmp -msse3 -DM_=50 --param max-inline-insns-auto=4000 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -DUSE_OPENCL -DOCL ../src/render/high_level/opencl_base.cpp ../src/g_malloc/g_malloc.cpp ../src/sys_funcs/*.cpp fr_cons_chem_osm_inv_gpu.cpp -ldl -lOpenCL -o fr_cons_chem_osm_inv_gpu
fl=5
test
fl=4
test

mpic++ -O3 -fopenmp -msse3 -DM_=100 --param max-inline-insns-auto=4000 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -DUSE_OPENCL -DOCL ../src/render/high_level/opencl_base.cpp ../src/g_malloc/g_malloc.cpp ../src/sys_funcs/*.cpp fr_cons_chem_osm_inv_gpu.cpp -ldl -lOpenCL -o fr_cons_chem_osm_inv_gpu
fl=5
test
fl=4
test

mpic++ -O3 -fopenmp -msse3 -DM_=200 --param max-inline-insns-auto=4000 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -DUSE_OPENCL -DOCL ../src/render/high_level/opencl_base.cpp ../src/g_malloc/g_malloc.cpp ../src/sys_funcs/*.cpp fr_cons_chem_osm_inv_gpu.cpp -ldl -lOpenCL -o fr_cons_chem_osm_inv_gpu
fl=5
test
fl=4
test
