pretty_name="falcon-7b"
model_name="tiiuae/falcon-7b"
version=$1
temp=${2:-0}
rp=${3:-1}
CACHE_DIR="/net/nfs/climate/tmp_cache/"
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
gpu=${4:-"0,1,2,3"}
tsp=${5:-4}
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir $CACHE_DIR \
    --model_name ${model_name} \
    --tensor_parallel_size ${tsp} \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 4 --max_tokens 2048 \
    --filepath $output_dir/${pretty_name}.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir $CACHE_DIR \
    --model_name ${model_name} \
    --tensor_parallel_size ${tsp} \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/${pretty_name}.turn1.json \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/${pretty_name}.turn2.json \
    --overwrite 