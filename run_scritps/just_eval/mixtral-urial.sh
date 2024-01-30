version=$1
rp=$2
N=$3
output_dir="result_dirs/just_eval/vllm_urial-${version}/rp=${rp}_N=${N}_T=${T}/"
mkdir -p $output_dir
gpu=0,1
tps=2
T=$4
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name mistralai/Mixtral-8x7B-v0.1 \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name just_eval --num_outputs $N \
    --top_p 0.9 --temperature $T  --repetition_penalty $rp --batch_size 16 --max_tokens 2048 \
    --output_folder $output_dir/ 
    # \
    # --overwrite  
 