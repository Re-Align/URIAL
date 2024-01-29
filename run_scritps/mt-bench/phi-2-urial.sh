version=$1
output_dir="result_dirs/mt-bench/vllm_urial-${version}/"
mkdir -p $output_dir
gpu=0
tsp=1
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --engine hf \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name microsoft/phi-2 \
    --tensor_parallel_size $tsp \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature 0 --repetition_penalty 1.15 --batch_size 4 --max_tokens 2048 \
    --filepath $output_dir/rp=1.15/phi-2.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --engine hf \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name microsoft/phi-2 \
    --tensor_parallel_size $tsp \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/rp=1.15/phi-2.turn1.json \
    --top_p 1 --temperature 0 --repetition_penalty 1.15 --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/rp=1.15/phi-2.turn2.json \
    --overwrite 