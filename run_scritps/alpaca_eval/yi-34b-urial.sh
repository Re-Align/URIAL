version=$1
rp=$2
N=$3
T=$4
output_dir="result_dirs/alpaca_eval/vllm_urial-${version}/rp=${rp}_N=${N}_T=${T}/"
mkdir -p $output_dir
gpu=0,1,2,3
tps=4
model_name="01-ai/Yi-34B"
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name $model_name \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature 0.5  --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --output_folder $output_dir/ \
    --overwrite 

# n_shards=4
# shard_size=202
# start_gpu=0
# for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
#     # gpu=0
#     CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
#         --start_index $start --end_index $end \
#         --urial $version \
#         --download_dir /net/nfs/s2-research/llama2/ \
#         --model_name meta-llama/Llama-2-7b-hf \
#         --tensor_parallel_size 1 \
#         --dtype bfloat16 \
#         --data_name alpaca_eval --num_outputs $N \
#         --top_p 1.0 --temperature 0.5  --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
#         --output_folder $output_dir/ \
#         --overwrite &
# done
 