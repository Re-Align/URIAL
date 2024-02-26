## Run URIAL Inference 

```bash
version=inst_1k_v4.help
bash run_scripts/mt-bench/phi-2-urial.sh $version 0 1.15
bash run_scripts/mt-bench/olmo-urial.sh $version 0 1.15 
bash run_scripts/mt-bench/llama-70b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/mixtral-urial.sh $version 0.5 1
bash run_scripts/mt-bench/llama-13b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/mistral-7b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/llama-7b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/yi-34b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/yi-6b-urial.sh $version 0 1.15 
bash run_scripts/mt-bench/gemma-7b-urial.sh $version 0.5 1 
bash run_scripts/mt-bench/gemma-2b-urial.sh $version 0.5 1 
```

## Formatting Results

```bash
suffix="0210v1"
python run_scripts/mt-bench/formatting_results.py Llama-2-70b-hf ${suffix} 
python run_scripts/mt-bench/formatting_results.py Mixtral-8x7B-v0.1 ${suffix} 
python run_scripts/mt-bench/formatting_results.py Mistral-7b-v0.1 ${suffix} 
python run_scripts/mt-bench/formatting_results.py Llama-2-7b-hf ${suffix} 
python run_scripts/mt-bench/formatting_results.py Llama-2-13b-hf ${suffix} 
python run_scripts/mt-bench/formatting_results.py Yi-6B ${suffix} 
python run_scripts/mt-bench/formatting_results.py Yi-34B ${suffix} 
python run_scripts/mt-bench/formatting_results.py olmo ${suffix} 
python run_scripts/mt-bench/formatting_results.py phi-2 ${suffix} 
python run_scripts/mt-bench/formatting_results.py gemma-7b ${suffix}
python run_scripts/mt-bench/formatting_results.py gemma-2b ${suffix}
```


## Generate MT-Bench Scores

```bash 
suffix="0210v1"
# git clone our modified version of FastChat from [url]
cd /net/nfs/mosaic/yuchenl/FastChat/fastchat/llm_judge/
ls -lht /net/nfs/mosaic/yuchenl/FastChat/fastchat/llm_judge//data/mt_bench/model_answer/ # make sure the model answer is there
# conda activate mb
# python --> /home/yuchenl/.conda/envs/mb/bin/python
python gen_judgment.py  --parallel 8 --model-list Llama-2-70b-hf-URIAL-${suffix} 
python gen_judgment.py  --parallel 8 --model-list Mixtral-8x7B-v0.1-URIAL-${suffix} 
python gen_judgment.py  --parallel 8 --model-list olmo-URIAL-${suffix}  
python gen_judgment.py  --parallel 8 --model-list phi-2-URIAL-${suffix}  
python gen_judgment.py  --parallel 8 --model-list Llama-2-13b-hf-URIAL-${suffix}
python gen_judgment.py  --parallel 8 --model-list Llama-2-7b-hf-URIAL-${suffix} 
python gen_judgment.py  --parallel 8 --model-list Yi-6B-URIAL-${suffix} 
python gen_judgment.py  --parallel 8 --model-list Yi-34B-URIAL-${suffix}
python gen_judgment.py  --parallel 8 --model-list Mistral-7b-v0.1-URIAL-${suffix}

python gen_judgment.py  --parallel 8 --model-list gemma-7b-URIAL-${suffix}  
python gen_judgment.py  --parallel 8 --model-list gemma-2b-URIAL-${suffix}  
```