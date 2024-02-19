import json 
import sys 
import random, string
import os 

answer_dir = "/home/yuchenl/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/" 

model_name = sys.argv[1]
suffix = sys.argv[2]
output_folder = f"result_dirs/mt-bench/urial_bench/"
model_id = f"{model_name}-URIAL-{suffix}"
turn1_path = f"result_dirs/mt-bench/urial_bench/{model_name}.turn1.json"
turn2_path = f"result_dirs/mt-bench/urial_bench/{model_name}.turn2.json"

turn1_results = json.load(open(turn1_path))
turn2_results = json.load(open(turn2_path))

results = []
for item1, item2 in zip(turn1_results, turn2_results):
    assert item1["question_id"] == item2["question_id"]
    res_item = {}
    res_item["question_id"] = item1["question_id"]
    # generate a random string
    res_item["answer_id"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=22))
    res_item["model_id"] = model_id
    res_item["choices"] = [
        {
            "index": 0,
            "turns": [
                item1["turn1_output"].replace("<|endoftext|>", "").strip(),
                item2["turn2_output"].replace("<|endoftext|>", "").strip()
            ]
        }
    ]
    results.append(res_item)
    
with open(f"{output_folder}/{model_id}.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

# copy the file to `/home/yuchenl/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/`

os.system(f"cp {output_folder}/{model_id}.jsonl {answer_dir}")

print(f"/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 4 --model-list {model_id}")


"""
python run_scripts/mt-bench/formatting_results.py Llama-2-70b-hf
python run_scripts/mt-bench/formatting_results.py Mixtral-8x7B-v0.1
python run_scripts/mt-bench/formatting_results.py Mistral-7b-v0.1
python run_scripts/mt-bench/formatting_results.py Llama-2-7b-hf
python run_scripts/mt-bench/formatting_results.py Llama-2-13b-hf
python run_scripts/mt-bench/formatting_results.py Yi-6B
python run_scripts/mt-bench/formatting_results.py Yi-34B
python run_scripts/mt-bench/formatting_results.py olmo
python run_scripts/mt-bench/formatting_results.py phi-2


version="0210v1"
python gen_judgment.py  --parallel 8 --model-list Llama-2-70b-hf-URIAL-${version} 
python gen_judgment.py  --parallel 8 --model-list Mixtral-8x7B-v0.1-URIAL-${version} 
python gen_judgment.py  --parallel 8 --model-list olmo-URIAL-${version}  
python gen_judgment.py  --parallel 8 --model-list phi-2-URIAL-${version}  
python gen_judgment.py  --parallel 8 --model-list Llama-2-13b-hf-URIAL-${version}
python gen_judgment.py  --parallel 8 --model-list Llama-2-7b-hf-URIAL-${version} 
python gen_judgment.py  --parallel 8 --model-list Yi-6B-URIAL-${version} 
python gen_judgment.py  --parallel 8 --model-list Yi-34B-URIAL-${version}
python gen_judgment.py  --parallel 8 --model-list Mistral-7b-v0.1-URIAL-${version}
"""
    

