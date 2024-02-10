output_dir="result_dirs/mt-bench/aligned/"
mkdir -p $output_dir
gpu=0,1,2,3
n=4
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name allenai/tulu-2-70b \
    --tensor_parallel_size $n \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/tulu-2-70b.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name allenai/tulu-2-70b \
    --tensor_parallel_size $n \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/tulu-2-70b.turn1.json \
    --top_p 1 --temperature $temp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/tulu-2-70b.turn2.json \
    --overwrite 