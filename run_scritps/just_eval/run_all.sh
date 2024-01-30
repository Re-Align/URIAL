# new 

bash run_scritps/just_eval/llama-70b-urial.sh inst_help_v5-1k 1 1 0.3
bash run_scritps/just_eval/llama-7b-urial.sh inst_help_v5-1k 1 3 0.3
bash run_scritps/just_eval/mistral-urial.sh inst_help_v5-1k 1 3 0.3

python run_scritps/just_eval/urial_filter.py \
        result_dirs/just_eval/vllm_urial-inst_help_v5-1k/rp=1_N=3/Llama-2-70b-hf.json \
        result_dirs/just_eval/urial/llama-70b-urial.inst_help_v5-1k.json


# aligned 
bash run_scritps/just_eval/llama-7b-chat.sh
bash run_scritps/just_eval/llama-70b-chat.sh
bash run_scritps/just_eval/mistral-instruct.sh
bash run_scritps/just_eval/yi-34b-chat.sh
bash run_scritps/just_eval/tulu-2-70b-dpo.sh


