
# new 

bash run_scripts/alpaca_eval/llama-7b-urial.sh inst_help_v5-1k 1 3 0.1
bash run_scripts/alpaca_eval/llama-70b-urial.sh inst_help_v5-1k 1 3 0.1
bash run_scripts/alpaca_eval/mistral-urial.sh inst_help_v5-1k 1 3 0.1

bash run_scripts/alpaca_eval/llama-7b-urial.sh inst_help_v5-2k 1 3 0.1
bash run_scripts/alpaca_eval/llama-70b-urial.sh inst_help_v5-2k 1 3 0.1
bash run_scripts/alpaca_eval/mistral-urial.sh inst_help_v5-2k 1 3 0.1


bash run_scripts/alpaca_eval/llama-7b-urial.sh inst_help_v5-1k 1 1 0.3
bash run_scripts/alpaca_eval/llama-70b-urial.sh inst_help_v5-1k 1 1 0.3
bash run_scripts/alpaca_eval/mistral-urial.sh inst_help_v5-1k 1 1 0.3

bash run_scripts/alpaca_eval/olmo-7b-urial.sh inst_help_v5-1k 1.2 1 0.7

bash run_scripts/alpaca_eval/olmo-7b-urial.sh inst_1k_v3 1.2 1 0.7

python run_scripts/alpaca_eval/urial_filter.py \
        result_dirs/alpaca_eval/vllm_urial-inst_help_v5-1k/rp=1_N=3/Llama-2-7b-hf.json \
        result_dirs/alpaca_eval/urial/llama-7b-urial.inst_help_v5-1k.json

python run_scripts/alpaca_eval/urial_filter.py \
        result_dirs/alpaca_eval/vllm_urial-inst_help_v5-1k/rp=1_N=3/Llama-2-70b-hf.json \
        result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json

python run_scripts/alpaca_eval/urial_filter.py \
        result_dirs/alpaca_eval/vllm_urial-inst_help_v5-2k/rp=1_N=3/Llama-2-7b-hf.json \
        result_dirs/alpaca_eval/urial/llama-7b-urial.inst_help_v5-2k.json

python run_scripts/alpaca_eval/urial_filter.py \
        result_dirs/alpaca_eval/vllm_urial-inst_help_v5-2k/rp=1_N=3/Llama-2-70b-hf.json \
        result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-2k.json

python run_scripts/alpaca_eval/urial_filter.py \
        result_dirs/alpaca_eval/vllm_urial-inst_help_v5-1k/rp=1_N=3/Mistral-7b-v0.1.json \
        result_dirs/alpaca_eval/urial/mistral-urial.inst_help_v5-1k.json


# python run_scripts/alpaca_eval/urial_filter.py \
#         result_dirs/alpaca_eval/vllm_urial-inst_help_v5-2k/rp=1_N=3/Llama-2-70b-hf.json \
#         result_dirs/alpaca_eval/urial/mistral.inst_help_v5-2k.json

# aligned 
bash run_scripts/alpaca_eval/llama-7b-chat.sh
bash run_scripts/alpaca_eval/llama-70b-chat.sh
bash run_scripts/alpaca_eval/mistral-instruct.sh



