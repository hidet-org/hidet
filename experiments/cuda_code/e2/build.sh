mkdir -p outs
cd outs || exit;
nvcc -lineinfo ../main.cu -gencode arch=compute_86,code=sm_86 --keep || exit;
mv main.ptx main.o a.out ..
cd ..
cuobjdump -sass main.o > sass.txt

