version=$1
rp=$2
N=$3
T=$4
output_dir="result_dirs/alpaca_eval/vllm_urial-${version}/rp=${rp}_N=${N}_T=${T}/"
mkdir -p $output_dir 
tps=1
# gpu=0
# tps=1

n_shards=4
shard_size=202
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do

    CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
        --urial $version \
        --engine hf \
        --download_dir /net/nfs/s2-research/llama2/ \
        --model_name allenai/OLMo-7B \
        --tensor_parallel_size $tps \
        --start_index $start --end_index $end \
        --dtype bfloat16 \
        --data_name alpaca_eval --num_outputs $N \
        --top_p 1.0 --temperature $T --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
        --output_folder $output_dir/ \
        --overwrite &
done 