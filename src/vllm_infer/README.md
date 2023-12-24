## URIAL with VLLM 

```bash
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
pip install datasets

```

<!-- 
### Using vLLM APIs.
```bash 
# https://docs.vllm.ai/en/latest/models/engine_args.html

python -m vllm.entrypoints.api_server \
        --download-dir /net/nfs/s2-research/llama2/ \
        --host 127.0.0.1 --port 2333 \
        --model meta-llama/Llama-2-70b-hf --tensor-parallel-size 4 --dtype bfloat16 

python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --vllm_url http://localhost:2333/generate \
    --model_name Llama-2-70b-hf \
    --urial_name inst_help \
    --top_p 1 --temperature 0 --repetition_penalty 1.05


python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --vllm_url http://localhost:2333/generate \
    --model_name Llama-2-70b-hf \
    --urial_name inst_help \
    --top_p 0.9 --temperature 0.7 --repetition_penalty 1.1
```
 -->

### Using vLLM local inference 
```bash
python src/vllm_infer/vllm_urial.py \
    --data_name just_eval \
    --vllm_mode local \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name meta-llama/Llama-2-70b-hf --tensor_parallel_size 4  --dtype bfloat16 \
    --urial_name inst_1k \
    --output_folder "vllm_outputs/just_eval/" \
    --top_p 1 --temperature 0 --repetition_penalty 1.1 --batch_size 32 --max_tokens 4096

# python src/vllm_infer/vllm_urial.py \
#     --data_name alpaca_eval \
#     --vllm_mode local \
#     --download_dir /net/nfs/s2-research/llama2/ \
#     --model_name mistralai/Mixtral-8x7B-v0.1 --tensor_parallel_size 4   --dtype bfloat16 \
#     --urial_name inst_1k \
#     --output_folder "vllm_outputs/alpaca_eval/" \
#     --top_p 1 --temperature 0 --repetition_penalty 1 --batch_size 32 --max_tokens 4096

    

CUDA_VISIBLE_DEVICES=0,1,2,3
python src/vllm_infer/vllm_urial.py \
    --data_name just_eval \
    --vllm_mode local \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name allenai/tulu-2-70b --tensor_parallel_size 4  --dtype bfloat16 \
    --output_folder "vllm_outputs/just_eval/" \
    --top_p 1 --temperature 0 --batch_size 32 --max_tokens 2048

#  --tokenizer_mode "slow" 

# CUDA_VISIBLE_DEVICES=2,3 python src/vllm_infer/vllm_urial.py \
#     --data_name just_eval \
#     --vllm_mode local \
#     --download_dir /net/nfs/s2-research/llama2/ \
#     --model_name meta-llama/Llama-2-70b-chat-hf --tensor_parallel_size 2  --dtype bfloat16 \
#     --output_folder "vllm_outputs/just_eval/" \
#     --top_p 1 --temperature 0 --batch_size 800 --max_tokens 7500
```


## Evaluation 

```bash
to_eval_file="vllm_outputs/just_eval/tulu-2-dpo-70b.URIAL=none.p=1.0.t=0.0.r=1.json"
run_name="tulu-2-dpo-70b"


to_eval_file="vllm_outputs/just_eval/tulu-2-dpo-7b.p=1.0.t=0.0.r=1.json"
run_name="tulu-2-dpo-7b"

to_eval_file="vllm_outputs/just_eval/Llama-2-70b-chat-hf.URIAL=none.p=1.0.t=0.0.r=1.json"
run_name="Llama-2-70b-chat-hf"

to_eval_file="vllm_outputs/just_eval/Llama-2-70b-hf.URIAL=inst_1k.p=1.0.t=0.0.r=1.1.json"
run_name="Llama-2-70b-hf.URIAL=inst_1k"


to_eval_file="vllm_outputs/just_eval/Yi-34B-Chat.p=1.0.t=0.0.r=1.json"
run_name="Yi-34B-Chat"

to_eval_file="vllm_outputs/just_eval/Yi-6B-Chat.p=1.0.t=0.0.r=1.json"
run_name="Yi-6B-Chat"


result_dir="result_dirs/just-eval_results/"
mkdir -p "$result_dir"
# GPT-4 for first five aspects on 0-800 examples 

n_shards=8
shard_size=100
for ((i=0; i<n_shards; i++)); do
    start=$((i * shard_size))
    end=$((start + shard_size))
    result_file="$result_dir/${run_name}.score_multi.gpt-4.${start}-${end}.json"
    just_eval \
        --mode "score_multi" \
        --model "gpt-4-0314" \
        --start_idx $start --end_idx $end \
        --first_file $to_eval_file \
        --output_file $result_file &
done
python src/scripts/merge_results.py $result_dir "${run_name}.score_multi.gpt-4"

just_eval --report_only --mode "score_multi" --output_file "$result_dir/${run_name}.score_multi.gpt-4.json"


safety_result_file="$result_dir/${run_name}.score_safety.chatgpt.json"
just_eval \
    --mode "score_safety" \
    --model "gpt-3.5-turbo-0613" \
    --first_file $to_eval_file \
    --start_idx 800 --end_idx 1000 \
    --output_file $safety_result_file

just_eval --report_only --mode "score_safety" --output_file $safety_result_file

```
 