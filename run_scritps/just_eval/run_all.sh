# bash run_scritps/alpaca_eval/llama-7b-urial.sh inst_help_v3-1k 1.2
# bash run_scritps/alpaca_eval/llama-13b-urial.sh inst_help_v3-1k 1.2
# bash run_scritps/alpaca_eval/llama-70b-urial.sh inst_help_v3-1k 1.2
# bash run_scritps/alpaca_eval/mistral-urial.sh inst_help_v3-1k 1.2
# bash run_scritps/alpaca_eval/mixtral-urial.sh inst_help_v3-1k 1.2

bash run_scritps/alpaca_eval/llama-7b-urial.sh inst_help_v4 1 3
bash run_scritps/alpaca_eval/llama-70b-urial.sh inst_help_v4 1 3
bash run_scritps/alpaca_eval/mistral-urial.sh inst_help_v4 1 3
 


# bash run_scritps/alpaca_eval/llama-7b-urial.sh inst_help_v2 1 3
# bash run_scritps/alpaca_eval/llama-70b-urial.sh inst_help_v2 1 3
# bash run_scritps/alpaca_eval/mistral-urial.sh inst_help_v2 1 3


bash run_scritps/alpaca_eval/llama-13b-urial.sh inst_help_v4 1
bash run_scritps/alpaca_eval/mixtral-urial.sh inst_help_v4 1



# new 

bash run_scritps/alpaca_eval/llama-7b-urial.sh inst_help_v5-1k 1 3
bash run_scritps/alpaca_eval/llama-70b-urial.sh inst_help_v5-1k 1 3
bash run_scritps/alpaca_eval/mistral-urial.sh inst_help_v5-1k 1 3

bash run_scritps/alpaca_eval/llama-7b-urial.sh inst_help_v5 1 3
bash run_scritps/alpaca_eval/llama-70b-urial.sh inst_help_v5 1 3
bash run_scritps/alpaca_eval/mistral-urial.sh inst_help_v5 1 3

python run_scritps/alpaca_eval/urial_filter.py \
        result_dirs/alpaca_eval/vllm_urial-inst_help_v5-1k/rp=1_N=3/Llama-2-70b-hf.json \
        result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json


# aligned 
bash run_scritps/alpaca_eval/llama-7b-chat.sh
bash run_scritps/alpaca_eval/llama-70b-chat.sh
bash run_scritps/alpaca_eval/mistral-instruct.sh



