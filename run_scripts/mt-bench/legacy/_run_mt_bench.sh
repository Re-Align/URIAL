# bash run_scripts/mt-bench/llama-70b-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/llama-13b-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/llama-7b-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/mistral-7b-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/mixtral-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/yi-34b-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/yi-6b-urial.sh inst_help_v5-1k
# bash run_scripts/mt-bench/phi-2-urial.sh inst_help_v5-1k

# bash run_scripts/mt-bench/olmo-urial.sh inst_help_v5-1k

# bash run_scripts/mt-bench/phi-2-urial.sh inst_1k_v3.help 0.5 1 "0" 1 &
# bash run_scripts/mt-bench/olmo-urial.sh inst_1k_v3.help 0.5 1 "1" 1 &
# bash run_scripts/mt-bench/yi-6b-urial.sh inst_1k_v3.help 0.5 1 "2" 1 &
# bash run_scripts/mt-bench/llama-7b-urial.sh inst_1k_v3.help 0.5 1 "3" 1 &
# # wait
# bash run_scripts/mt-bench/mistral-7b-urial.sh inst_1k_v3.help 0.5 1
# bash run_scripts/mt-bench/llama-13b-urial.sh inst_1k_v3.help 0.5 1
# bash run_scripts/mt-bench/mixtral-urial.sh inst_1k_v3.help 0.5 1
# bash run_scripts/mt-bench/yi-34b-urial.sh inst_1k_v3.help 0.5 1
# bash run_scripts/mt-bench/llama-70b-urial.sh inst_1k_v3.help 0.5 1


# try to reproduce the previous results
version=inst_1k_v4.help
bash run_scripts/mt-bench/mixtral-urial.sh $version 0.7 1
bash run_scripts/mt-bench/llama-70b-urial.sh $version 0.7 1
bash run_scripts/mt-bench/mistral-7b-urial.sh $version 0.7 1
bash run_scripts/mt-bench/llama-13b-urial.sh $version 0.7 1
bash run_scripts/mt-bench/yi-34b-urial.sh $version 0.7 1
bash run_scripts/mt-bench/phi-2-urial.sh $version 0.7 1
bash run_scripts/mt-bench/olmo-urial.sh $version 0.7 1 
bash run_scripts/mt-bench/yi-6b-urial.sh $version 0.7 1 
bash run_scripts/mt-bench/llama-7b-urial.sh $version 0.7 1
wait  

version=inst_1k_v4.help


# bash run_scripts/mt-bench/llama-70b-urial.sh $version 0.7 1




# bash run_scripts/mt-bench/tulu-2-70b-dpo.sh
# bash run_scripts/mt-bench/tulu-2-70b-sft.sh

# bash run_scripts/mt-bench/llama-70b-urial.sh inst_help_v5-2k
# bash run_scripts/mt-bench/llama-13b-urial.sh inst_help_v5-2k
# bash run_scripts/mt-bench/llama-7b-urial.sh inst_help_v5-2k
# bash run_scripts/mt-bench/mistral-7b-urial.sh inst_help_v5-2k
# bash run_scripts/mt-bench/mixtral-urial.sh inst_help_v5-2k
# bash run_scripts/mt-bench/yi-34b-urial.sh inst_help_v5-2k
# bash run_scripts/mt-bench/yi-6b-urial.sh inst_help_v5-2k