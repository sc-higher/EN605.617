echo Compiling...
nvcc final.cu -o final
nvcc final_rk2.cu -o final_rk2
nvcc final_cpu.cu -o final_cpu

echo Running...
echo FTCS...
./final
echo RK2...
./final_rk2
echo FTCS CPU...
./final_cpu