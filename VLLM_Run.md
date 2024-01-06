# Run URIAL for AlpacaEval 


```bash 
version="inst_help_v2"
output_dir="result_dirs/alpaca_eval/vllm_urial-${version}/"
mkdir -p $output_dir

n_shards=1
shard_size=101
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu python src/vllm_infer.py \
        --start_index $start --end_index $end \
        --urial $version \
        --download_dir /net/nfs/s2-research/llama2/ \
        --model_name meta-llama/Llama-2-7b-hf \
        --tensor_parallel_size 1 \
        --dtype bfloat16 \
        --max_tokens 2048 \
        --data_name alpaca_eval \
        --top_p 1 --temperature 0 --repetition_penalty 1.2 --batch_size 4 --max_tokens 4096 \
        --output_folder $output_dir/rp=1.2/ 
        # &
done


python src/scripts/merge_results.py result_dirs/alpaca_eval/urial-inst_help/ Mistral-7B-v0.1
python src/scripts/reformat.py result_dirs/alpaca_eval/urial-inst_help/Mistral-7B-v0.1.json
```