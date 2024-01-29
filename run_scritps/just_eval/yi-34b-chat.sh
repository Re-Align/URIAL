filepath="result_dirs/alpaca_eval/aligned/Yi-34B-chat.json"
gpu=0,1,2,3
tps=4
rp=1
N=1
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name 01-ai/Yi-34B-chat \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature 0.1 --repetition_penalty $rp --batch_size 16 --max_tokens 2048 \
    --filepath $filepath \
    --overwrite