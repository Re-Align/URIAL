## Run URIAL Inference 


<!-- 
bash run_scripts/mt-bench/olmo-7b-urial.sh $version 0 1.15  1 1 
bash run_scripts/mt-bench/falcon-7b-urial.sh $version 0 1.15  0 1
bash run_scripts/mt-bench/mpt-7b-urial.sh $version 0 1.15  2 1 
bash run_scripts/mt-bench/phi-2-urial.sh $version 0 1.15  1 1 
bash run_scripts/mt-bench/amber-urial.sh $version 0 1.15  0 1 

 -->
```bash
version=inst_1k_v4.help
# bash run_scripts/mt-bench/phi-2-hf-urial.sh $version 0 1.15 # hf version (deprecated)
# bash run_scripts/mt-bench/olmo-hf-urial.sh $version 0 1.15 # hf version (deprecated)
bash run_scripts/mt-bench/falcon-7b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/mpt-7b-urial.sh $version 0 1.15
bash run_scripts/mt-bench/amber-urial.sh $version 0 1.15
bash run_scripts/mt-bench/olmo-7b-urial.sh $version 0 1.15 # vllm version 
bash run_scripts/mt-bench/phi-2-urial.sh $version 0 1.15 # vllm version 
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

<details>
<summary>More details</summary>

- The first argument is the `version` of the urial prompt. Here we use the same version for all models.
- The second argument is the `temperature` for the sampling. We use 0.5 for gemma and mixtral, and 0 for the rest. 
- The third argument is the `repetition_penalty` for the sampling. We use 1.15 for most models to avoid repetition, and 1 for gemma and mixtral.
- The `top_p` is set to 1 for all models. 
- All models are run with vLLM. Previously, phi-2 and olmo were run with HF, but we changed to vLLM for consistency after they're supported by vLLM.
</details>

<details>
<summary>FAQ</summary>

> Q: How do you decide the hyperparameters (temperature `t` and repetition_penalty `rp`)?
>> A: We tried three different settings for the temperature and repetition_penalty, i.e., (`t=0+rp=1.15`; `t=0.5+rp=1`; `t=1+rp=1`), and found that the above settings give the best results for each model. If the performance is similar for different settings, we choose the one with `t=0+rp=1.15` for best **reproducibility**.  

--- 
> Q: Why do you use vLLM for all models?
>> A: We use vLLM for all models to ensure consistency. Previously, phi-2 and olmo were run with HF, but we changed to vLLM for consistency after they're supported by vLLM.
</details>


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
python run_scripts/mt-bench/formatting_results.py gemma-7b ${suffix}
python run_scripts/mt-bench/formatting_results.py gemma-2b ${suffix}

python run_scripts/mt-bench/formatting_results.py olmo-7b-vllm ${suffix} 
python run_scripts/mt-bench/formatting_results.py phi-2-vllm ${suffix} 
python run_scripts/mt-bench/formatting_results.py mpt-7b ${suffix} 
python run_scripts/mt-bench/formatting_results.py falcon-7b ${suffix} 
python run_scripts/mt-bench/formatting_results.py amber ${suffix} 


# python run_scripts/mt-bench/formatting_results.py olmo ${suffix} 
# python run_scripts/mt-bench/formatting_results.py phi-2 ${suffix} 

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


python gen_judgment.py  --parallel 8 --model-list olmo-7b-vllm-URIAL-${suffix}
python gen_judgment.py  --parallel 8 --model-list phi-2-vllm-URIAL-${suffix}
python gen_judgment.py  --parallel 8 --model-list mpt-7b-URIAL-${suffix}
python gen_judgment.py  --parallel 8 --model-list falcon-7b-URIAL-${suffix}
python gen_judgment.py  --parallel 8 --model-list amber-URIAL-${suffix}
```