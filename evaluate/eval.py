import argparse
import os
import json
import openai 
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from eval_utils import (
    retry_handler, 
    openai_chat_request, 
)
from datasets import load_dataset
 
def get_args():
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--action", type=str, default="trial", required=True)
    parser.add_argument("--mode", type=str, default="pairwise", required=True)
    parser.add_argument("--eval_template", type=str, default="", required=True)
    parser.add_argument("--model_output_file", type=str, required=True) 
    parser.add_argument("--ref_output_file", type=str, required=True)
    parser.add_argument("--eval_output_file", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)  
    parser.add_argument("--save_interval", type=int, default=3)
    
    # Prompt configs 
    parser.add_argument("--max_words_to_eval", type=int, default=-1)
    
    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    args = parser.parse_args() 
    if args.api_key is not None:
        openai.api_key = args.api_key 
     
    return args
        

# def parse_result(result_str, mode="json"):
#     if "neither" in result_str.lower():
#         return "neither"
#     elif "tie" in result_str:
#         return "tie"
#     elif "A" in result_str:
#         return "A"
#     elif "B" in result_str:
#         return "B"
#     else:
#         return "Not Matched"

def parse_result(result_str, mode="json"): 
    result_str = result_str.strip()
    result_str = result_str.strip("```")
    try:
        result_str = result_str.replace("\\", "\\\\")
        parsed_result = json.loads(result_str)
    except Exception as e:
        print(e)
        raise Exception(f"Failed to parse the result: {result_str}")
        exit()
    return parsed_result
                    
def gpt_eval(results, args):
    # try to load the existing results from args.eval_output_file 
    if os.path.exists(args.eval_output_file):
        cnt = 0 
        with open(args.eval_output_file, "r") as f:
            existing_results = json.load(f) 
        for i in range(len(existing_results)):
            e = existing_results[i]
            t = results[i]
            if e["prompt"] != t["prompt"]:
                continue
            # if e["prompt"] == t["prompt"] and e["result"] != "N/A":
            #     results[i]["result"] = e["result"]
            #     cnt += 1 
            if "result" in e:
                t["result"] = e["result"]
                if "parsed_result" in e: 
                    t["parsed_result"] = e["parsed_result"]
                cnt += 1
        print(f"loading {cnt} results from {args.eval_output_file}")
    openai_args = {
        "prompt": "TODO",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stop": []
    }
    if args.model:
        openai_args['model'] = args.model
    if args.engine:
        openai_args['engine'] = args.engine
        
    @retry_handler(retry_limit=10)
    def api(ind, item, **kwargs):
        result = openai_chat_request(**kwargs)
        result = result[0]  
        return result
    
    # results = results[args.start_idx:args.end_idx] # for debug
    for ind, item in tqdm(enumerate(results), total=len(results), desc=f"Evaluating: {args.eval_output_file} "):
        if item["result"] != "N/A": 
            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            print(f"Skipping {ind} for {args.eval_output_file}")
            # skip the existing results
            continue
            
        openai_args["prompt"] = item["prompt"]
        try:
            result = api(ind, item, **openai_args)
            results[ind]["result"] = result
            results[ind]["parsed_result"] = parse_result(results[ind]["result"])
            r = results[ind]["parsed_result"]
            # if r in ["A", "B"]:
            #     results[ind]["winner"] = item["assignment"][r]
            # else:
            #     results[ind]["winner"] = r 
                
        except Exception as e:
            print(e)
            raise Exception("Failed!")
        
        # print("Done!") 
        if ind % args.save_interval == 0 or ind == len(results)-1:
            with open(args.eval_output_file, "w") as f:
                json.dump(results, f, indent=2) 
    with open(args.eval_output_file, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def shorten(text, K=-1):
    # if K > 0 and len(text.split(" ")) > K:
    #     text = " ".join(text.split(" ")[:K]) + "... (truncated)"
    pass 
    return text
 
    
def placeholder_generation(args): 
    
    with open(args.eval_template) as f:
        eval_template = f.read() 
    results = []

    with open(args.model_output_file, 'r') as f:
        candidates = json.load(f) 

    with open(args.ref_output_file, 'r') as f:
        references = json.load(f)  

    assert len(candidates) == len(references)
            
    L = len(candidates)
    if args.end_idx < 0 or args.end_idx > L:
        args.end_idx = L

    print(f"# examples in candidates: {len(candidates)}; We take {args.end_idx-args.start_idx} for evaluation.")
    candidates = candidates[args.start_idx:args.end_idx]
    references = references[args.start_idx:args.end_idx]
    
    results = []
    for item, ref_item in zip(candidates, references):
        instruction = item["instruction"] 
    
        o = item["output"][0] if type(item["output"]) == list else item["output"]
        r = ref_item["output"][0] if type(ref_item["output"]) == list else ref_item["output"]
        # random decide which is A and which is B 
        d = {}
        # d["id"] = item["id"]
        d["input"] = instruction           
        d["model_output"] = item["output"]
        d["ref_output"] =  r
        d["generator"] = item["generator"] 
        d["ref_generator"] = ref_item["generator"] 
        d["eval_config"] = {"mode": args.mode, "gpt": args.model, "max_words": args.max_words_to_eval}
        
        ## Prompt composition for pairwise evaluation
        if args.mode == "pairwise":
            if random.random() < 0.5:
                A = o
                B = r
                d["assignment"] = {"A": d["generator"], "B": d["ref_generator"]}
            else:
                A = r
                B = o
                d["assignment"] = {"A": d["ref_generator"], "B": d["generator"]} 
            prompt = eval_template
            prompt = prompt.replace("{$instruction}", instruction)
            prompt = prompt.replace("{$candidate_A}", A)
            prompt = prompt.replace("{$candidate_B}", B)
        elif args.mode == "ref_score" or args.mode == "score":
            prompt = eval_template
            prompt = prompt.replace("{$instruction}", instruction)
            prompt = prompt.replace("{$reference}", r)
            prompt = prompt.replace("{$candidate}", o)
        d["prompt"] = prompt
        d["result"] = "N/A" 
        results.append(d)
    return results 


def main():
    random.seed(42)
    args = get_args()
    
    if args.action.startswith("trial"):
        results = placeholder_generation(args)
        print(f"We have {len(results)} examples to evaluate!")
        with open(args.eval_output_file, "w") as f:
            json.dump(results, f, indent=2) 
    elif args.action.startswith("eval"):
        results = placeholder_generation(args)
        results = gpt_eval(results, args) 
    else:
        print("Not implemented yet!")

if __name__ == "__main__": 
    main()
    

"""
https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/text_davinci_003/model_outputs.json
"""