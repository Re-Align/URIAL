import json 
import sys 
import random, string
import os 

turn1_path = "result_dirs/mt-bench/vllm_urial-inst_help_v2/rp=1.15/Mixtral-8x7B-v0.1.turn1.json"
turn2_path = "result_dirs/mt-bench/vllm_urial-inst_help_v2/rp=1.15/Mixtral-8x7B-v0.1.turn2.json"
output_folder = "result_dirs/mt-bench/vllm_urial-inst_help_v2/rp=1.15/"
model_id = "Mixtral-8x7B-v0.1-URIAL"
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
                item1["turn1_output"],
                item2["turn2_output"]
            ]
        }
    ]
    results.append(res_item)
    
with open(f"{output_folder}/{model_id}.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

# copy the file to `/home/yuchenl/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/`
os.system(f"cp {output_folder}/{model_id}.jsonl /home/yuchenl/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/")

print(f"python gen_judgment.py --model-list {model_id} --parallel 4")
    

