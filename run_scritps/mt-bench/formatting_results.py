import json 
import sys 
import random, string
import os 

# model_name = "Mixtral-8x7B-v0.1"
model_name = sys.argv[1]


if model_name in ["tulu-2-70b", "tulu-2-dpo-70b"]:
    turn1_path = f"result_dirs/mt-bench/aligned/{model_name}.turn1.json"
    turn2_path = f"result_dirs/mt-bench/aligned/{model_name}.turn2.json"
    output_folder = f"result_dirs/mt-bench/aligned/"
    model_id = model_name
else:
    version="inst_help_v5-2k"
    turn1_path = f"result_dirs/mt-bench/vllm_urial-{version}/rp=1.15/{model_name}.turn1.json"
    turn2_path = f"result_dirs/mt-bench/vllm_urial-{version}/rp=1.15/{model_name}.turn2.json"
    output_folder = f"result_dirs/mt-bench/vllm_urial-{version}/rp=1.15/"
    model_id = f"{model_name}-URIAL-2k"

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

print(f"/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 4 --model-list {model_id}")


"""
python run_scritps/mt-bench/formatting_results.py Llama-2-70b-hf
python run_scritps/mt-bench/formatting_results.py Mixtral-8x7B-v0.1
python run_scritps/mt-bench/formatting_results.py Mistral-7b-v0.1
python run_scritps/mt-bench/formatting_results.py Llama-2-7b-hf
python run_scritps/mt-bench/formatting_results.py Llama-2-13b-hf
python run_scritps/mt-bench/formatting_results.py Yi-6B
python run_scritps/mt-bench/formatting_results.py Yi-34B




cd /home/yuchenl/FastChat/fastchat/llm_judge
ls -lht /home/yuchenl/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Llama-2-70b-hf-URIAL-2k &
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Llama-2-7b-hf-URIAL-2k &
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Llama-2-13b-hf-URIAL-2k &
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Yi-6B-URIAL-2k &
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Yi-34B-URIAL-2k  &
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Mixtral-8x7B-v0.1-URIAL-2k &
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 2 --model-list Mistral-7b-v0.1-URIAL-2k &



python run_scritps/mt-bench/formatting_results.py tulu-2-70b
/home/yuchenl/.conda/envs/fschat/bin/python gen_judgment.py  --parallel 8 --model-list tulu-2-70b
"""
    

