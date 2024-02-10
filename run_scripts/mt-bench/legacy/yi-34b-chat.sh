output_dir="result_dirs/mt-bench/aligned/"
mkdir -p $output_dir
gpu=0,1
n=2
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name 01-ai/Yi-34B-chat \
    --tensor_parallel_size $n \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/yi-34B-chat.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name 01-ai/Yi-34B-chat \
    --tensor_parallel_size $n \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/yi-34B-chat.turn1.json \
    --top_p 1 --temperature $temp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/yi-34B-chat.turn2.json \
    --overwrite 