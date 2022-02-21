#!/bin/bash
echo "Average Paged Time (ms), Average Pinned Time (ms)" > summarized_results.txt
./assignment.exe 128 32 1024 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 128 32 10240 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 128 32 102400 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 128 32 1024000 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 1024 256 1024 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 1024 256 10240 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 1024 256 102400 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 1024 256 1024000 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 4096 1024 1024 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 4096 1024 10240 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt 
./assignment.exe 4096 1024 102400| awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 4096 1024 1024000 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 16384 2048 1024 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 16384 2048 10240 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 16384 2048 102400 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
./assignment.exe 16384 2048 1024000 | awk '{printf "%s ",$0;next;}' | awk -F ' ' '{print $15,$19}' | tee -a summarized_results.txt
