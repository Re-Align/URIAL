version="inst_help_v2" 
output_dir="result_dirs/mt-bench/vllm_urial-${version}/"
mkdir -p $output_dir
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name meta-llama/Llama-2-7b-hf \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature 0 --repetition_penalty 1.15 --batch_size 4 --max_tokens 2048 \
    --filepath $output_dir/rp=1.15/Llama-2-7b-hf.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name meta-llama/Llama-2-7b-hf \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/rp=1.15/Llama-2-7b-hf.turn1.json \
    --top_p 1 --temperature 0 --repetition_penalty 1.15 --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/rp=1.15/Llama-2-7b-hf.turn2.json \
    --overwrite 