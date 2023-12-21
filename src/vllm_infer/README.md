## URIAL with VLLM 

```bash
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
pip install datasets

```


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


### Using vLLM local inference 
```bash
python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --vllm_mode local \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name meta-llama/Llama-2-13b-hf --tensor_parallel_size 4  --dtype bfloat16 \
    --urial_name inst_help \
    --output_folder "vllm_outputs/alpaca_eval/" \
    --top_p 0.9 --temperature 0.5 --repetition_penalty 1.0 --batch_size 32 --max_tokens 4096

python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --vllm_mode local \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name mistralai/Mixtral-8x7B-v0.1 --tensor_parallel_size 4   --dtype bfloat16 \
    --urial_name inst_help \
    --output_folder "vllm_outputs/alpaca_eval/" \
    --top_p 1 --temperature 0 --repetition_penalty 1.2 --batch_size 32 --max_tokens 4096

    

CUDA_VISIBLE_DEVICES=2,3 python src/vllm_infer/vllm_urial.py \
    --data_name just_eval \
    --vllm_mode local \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name allenai/tulu-2-dpo-70b --tensor_parallel_size 2  --dtype bfloat16 \
    --output_folder "vllm_outputs/just_eval/" \
    --top_p 1 --temperature 0 --batch_size 800 --max_tokens 7500

CUDA_VISIBLE_DEVICES=2,3 python src/vllm_infer/vllm_urial.py \
    --data_name just_eval \
    --vllm_mode local \
    --download_dir /net/nfs/s2-research/llama2/ \
    --model_name meta-llama/Llama-2-70b-chat-hf --tensor_parallel_size 2  --dtype bfloat16 \
    --output_folder "vllm_outputs/just_eval/" \
    --top_p 1 --temperature 0 --batch_size 800 --max_tokens 7500
```


## Using Mixtral 8x7b for AlpacaEval 


```bash 
python -m vllm.entrypoints.api_server \
        --download-dir /net/nfs/s2-research/llama2/ \
        --host 127.0.0.1 --port 2333 \
        --model mistralai/Mixtral-8x7B-v0.1 --tensor-parallel-size 8

CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.api_server \
        --download-dir /net/nfs/s2-research/llama2/ \
        --host 127.0.0.1 --port 2335 \
        --model mistralai/Mistral-7B-v0.1 --tensor-parallel-size 2 --dtype bfloat16

python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --vllm_url http://localhost:2335/generate \
    --model_name Mistral-7B-v0.1 \
    --urial_name inst_1k \
    --top_p 0.9 --temperature 0.7 --repetition_penalty 1.1
```