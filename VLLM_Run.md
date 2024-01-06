# Run URIAL for AlpacaEval 


```bash 
version="inst_help_v2"
# version="inst_1k_v3"
output_dir="result_dirs/alpaca_eval/vllm_urial-${version}/"
mkdir -p $output_dir

n_shards=8
shard_size=101
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    # gpu=0
    CUDA_VISIBLE_DEVICES=$gpu python src/vllm_infer.py \
        --start_index $start --end_index $end \
        --urial $version \
        --download_dir /net/nfs/s2-research/llama2/ \
        --model_name meta-llama/Llama-2-7b-hf \
        --tensor_parallel_size 1 \
        --dtype bfloat16 \
        --data_name alpaca_eval \
        --top_p 1 --temperature 0 --repetition_penalty 1.0 --batch_size 1 --max_tokens 2048 \
        --output_folder $output_dir/rp=1.0/ \
        --overwrite
        &
done




# HF
version="inst_help_v2"
output_dir="result_dirs/alpaca_eval/hf_urial-${version}/"
mkdir -p $output_dir
start=0
end=101
python src/vllm_infer.py \
        --engine hf \
        --start_index $start --end_index $end \
        --urial $version \
        --download_dir /net/nfs/s2-research/llama2/ \
        --model_name meta-llama/Llama-2-7b-hf --hf_bf16 \
        --tensor_parallel_size 1 \
        --dtype bfloat16 \
        --tokenizer_mode slow \
        --max_tokens 2048 \
        --data_name alpaca_eval \
        --batch_size 1 --max_tokens 1024 \
        --output_folder $output_dir/rp=1.0/ \
        --overwrite

        # --top_p 1 --temperature 0 --repetition_penalty 1.0 
        --dtype bfloat16 \


python src/scripts/merge_results.py result_dirs/alpaca_eval/urial-inst_help/ Mistral-7B-v0.1
python src/scripts/reformat.py result_dirs/alpaca_eval/urial-inst_help/Mistral-7B-v0.1.json
```