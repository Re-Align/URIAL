# Run URIAL for AlpacaEval 


```bash 
version="inst_help"
output_dir="result_dirs/alpaca_eval/urial-${version}/"
mkdir -p $output_dir

n_shards=8
shard_size=101
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu python src/infer.py \
        --start_index $start --end_index $end \
        --interval 1 \
        --model_path "mistralai/Mistral-7B-v0.1" \
        --bf16 \
        --max_output_tokens 2048 \
        --data_name alpaca_eval \
        --adapt_mode "urial" \
        --urial_prefix_path "urial_prompts/${version}.txt" \
        --repetition_penalty 1.2 \
        --output_folder $output_dir/rp=1.2/ &
done


python src/scripts/merge_results.py result_dirs/alpaca_eval/urial-inst_help/ Mistral-7B-v0.1
python src/scripts/reformat.py result_dirs/alpaca_eval/urial-inst_help/Mistral-7B-v0.1.json
```