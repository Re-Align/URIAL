filepath="result_dirs/just_eval/aligned/Llama-2-7b-chat-hf.json"
gpu=0,1
tps=2
rp=1
N=1
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name just_eval --num_outputs $N \
    --top_p 1.0 --temperature 0 --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $filepath \
    --overwrite