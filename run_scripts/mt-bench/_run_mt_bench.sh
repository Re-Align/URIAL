# try to reproduce the previous results
version=inst_1k_v4.help
bash run_scripts/mt-bench/phi-2-urial.sh $version 0 1.15
bash run_scripts/mt-bench/olmo-urial.sh $version 0 1.15 
bash run_scripts/mt-bench/llama-70b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/mixtral-urial.sh $version 0.5 1
bash run_scripts/mt-bench/llama-13b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/mistral-7b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/llama-7b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/yi-34b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/yi-6b-urial.sh $version 0 1.15 
wait 

 