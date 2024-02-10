# try to reproduce the previous results
version=inst_1k_v4.help
bash run_scritps/mt-bench/mixtral-urial.sh $version 0.7 1
bash run_scritps/mt-bench/llama-70b-urial.sh $version 0 1.15
bash run_scritps/mt-bench/phi-2-urial.sh $version 0 1.15
bash run_scritps/mt-bench/olmo-urial.sh $version 0 1.15 
bash run_scritps/mt-bench/mistral-7b-urial.sh $version 0 1.15
bash run_scritps/mt-bench/llama-13b-urial.sh $version 0.3 1.15
bash run_scritps/mt-bench/yi-34b-urial.sh $version 0 1.15
bash run_scritps/mt-bench/yi-6b-urial.sh $version 0 1.15 
bash run_scritps/mt-bench/llama-7b-urial.sh $version 0 1.15
wait  

version=inst_1k_v4.help

 