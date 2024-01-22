


```bash
wget -O result_dirs/alpaca_eval/aligned/text_davinci_003.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/text_davinci_003/model_outputs.json
wget -O result_dirs/alpaca_eval/aligned/gpt-3.5-turbo-0301.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt-3.5-turbo-0301/model_outputs.json
wget -O result_dirs/alpaca_eval/aligned/gpt4_turbo.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt4_turbo/model_outputs.json
wget -O result_dirs/alpaca_eval/aligned/gpt4.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt4/model_outputs.json
# wget -O result_dirs/alpaca_eval/aligned/llama-2-70b-chat-hf.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/llama-2-70b-chat-hf/model_outputs.json

```

```bash

target_file="result_dirs/alpaca_eval/vllm_urial-inst_help_v2/rp=1.15/Llama-2-70b-hf-URIAL.json"
# ref_file="result_dirs/alpaca_eval/aligned/gpt-3.5-turbo-0301.json"
target_file="result_dirs/alpaca_eval/aligned/gpt4.json"
ref_file="result_dirs/alpaca_eval/aligned/gpt4_turbo.json"
eval_file="evaluate/results/gpt4.ref=gpt4_turbo.multi.json"
python evaluate/eval.py \
    --action eval \
    --mode pariwise \
    --eval_template evaluate/eval_template_multi.md \
    --model_output_file $target_file \
    --ref_output_file $ref_file \
    --eval_output_file $eval_file \
    --start_idx 0 \
    --end_idx 10

```