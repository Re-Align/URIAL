## URIAL with VLLM 

```bash
conda create -n vllm python=3.10
conda activate vllm
pip install vllm


python -m vllm.entrypoints.api_server \
        --download-dir /net/nfs/s2-research/llama2/ \
        --host 127.0.0.1 --port 2333 \
        --model meta-llama/Llama-2-70b-hf --tensor-parallel-size 4


python src/vllm_infer/vllm_urial.py \
    --data_name alpaca_eval \
    --model_name Llama-2-70b-hf \
    --urial_name inst_help \
    --top_p 1 --temperature 0
```