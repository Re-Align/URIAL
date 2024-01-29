



```bash
wget -O result_dirs/alpaca_eval/aligned/text_davinci_003.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/text_davinci_003/model_outputs.json
wget -O result_dirs/alpaca_eval/aligned/gpt-3.5-turbo-0301.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt-3.5-turbo-0301/model_outputs.json
wget -O result_dirs/alpaca_eval/aligned/gpt4_turbo.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt4_turbo/model_outputs.json
wget -O result_dirs/alpaca_eval/aligned/gpt4.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/gpt4/model_outputs.json
# wget -O result_dirs/alpaca_eval/aligned/llama-2-70b-chat-hf.json https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/llama-2-70b-chat-hf/model_outputs.json

```

## Pairwise
```bash 


# model_name="Llama-2-70b-chat-hf"
# target_file="result_dirs/alpaca_eval/aligned/${model_name}.json"
# ref_name="text_davinci_003"
# ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"

model_name="Llama-2-70b-urial"
target_file="result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json"
ref_name="Llama-2-70b-chat-hf"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"

model_name="Llama-2-7b-urial"
target_file="result_dirs/alpaca_eval/urial/llama-7b-urial.inst_help_v5-1k.json"
ref_name="Llama-2-7b-chat-hf"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"

model_name="Llama-2-70b-urial-2k"
target_file="result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-2k.json"
ref_name="Llama-2-70b-chat-hf"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"

model_name="Llama-2-7b-urial-2k"
target_file="result_dirs/alpaca_eval/urial/llama-7b-urial.inst_help_v5-2k.json"
ref_name="Llama-2-7b-chat-hf"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"



model_name="Llama-2-70b-urial"
target_file="result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json"
ref_name="tulu-2-70b"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"

model_name="Llama-2-70b-urial"
target_file="result_dirs/alpaca_eval/urial/llama-70b-urial.inst_help_v5-1k.json"
ref_name="tulu-2-70b-dpo"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"


eval_folder="evaluate/results/ref=${ref_name}/"
mkdir -p $eval_folder

n_shards=8
shard_size=101
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${model_name}.$start-$end.json"
    python evaluate/eval.py \
        --action eval \
        --mode pairwise \
        --eval_template evaluate/eval_template_pairwise.md \
        --model_output_file $target_file \
        --ref_output_file $ref_file \
        --eval_output_file $eval_file \
        --start_idx $start --end_idx $end  &
done

python evaluate/merge_results.py $eval_folder $model_name
```


## (Ref+)Score  

```bash 
ref_name="gpt4_turbo"
ref_file="result_dirs/alpaca_eval/aligned/${ref_name}.json"
eval_folder="evaluate/results/ref=${ref_name}_score/"

model_name="Llama-2-7b-chat-hf"
target_file="result_dirs/alpaca_eval/aligned/${model_name}.json"

# model_name="Llama-2-7b-urial"
# target_file="result_dirs/alpaca_eval/urial/llama-7b-urial.inst_help_v5-1k.json"

mkdir -p $eval_folder

n_shards=1
shard_size=101
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    eval_file="${eval_folder}/${model_name}.$start-$end.json"
    python evaluate/eval.py \
        --action eval \
        --mode score \
        --eval_template evaluate/eval_template_score.md \
        --model_output_file $target_file \
        --ref_output_file $ref_file \
        --eval_output_file $eval_file \
        --start_idx $start --end_idx $end
        #  &
done

wait 

python evaluate/merge_results.py $eval_folder $model_name
```