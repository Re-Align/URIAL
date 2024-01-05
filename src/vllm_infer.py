import requests
from typing import List 
import argparse
from datasets import load_dataset
import urllib.request
from tqdm import tqdm
import json
import os  
from vllm import LLM, SamplingParams
from vllm_utils import load_eval_data, save_outputs
from vllm_utils import openai_chat_request, retry_handler

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--engine', default="vllm", type=str)
    parser.add_argument('--output_folder', default="vllm_outputs", type=str)
    parser.add_argument('--download_dir', default=None, type=str)    
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--tokenizer_name', default="auto", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--tokenizer_mode', type=str, default="auto") 
    parser.add_argument('--data_name', default="alpaca_eval", type=str) 
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--temperature',default=0, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=7500, type=int)
    parser.add_argument('--start_index',default=0, type=int) # 0 means from the beginning of the list
    parser.add_argument('--end_index',default=-1, type=int) # -1 means to the end of the list 
    parser.add_argument('--filepath',default="auto", type=str)  
    return parser.parse_args()

 

if __name__ == "__main__":
    args = parse_args()    
    
    
    
    # Load the model
    print("loading model!")
    if args.tokenizer_name == "auto":
        args.tokenizer_name = args.model_name
    if args.engine == "vllm":
        llm = LLM(model=args.model_name, tokenizer=args.tokenizer_name, tensor_parallel_size=args.tensor_parallel_size, download_dir=args.download_dir, dtype=args.dtype, tokenizer_mode=args.tokenizer_mode)        
    
    print("loading dataset!")
    # Data loading 
    id_strs, chat_history, model_inputs, metadata = load_eval_data(args) 
    
    # Decide the output filepath
    if args.filepath == "auto":
        # Decide the output filepath 
        if "/" in args.model_name:
            args.model_name = args.model_name.split("/")[-1]   
        os.system(f"mkdir -p {args.output_folder}")
        if args.end_index == -1 and args.start_index == 0:
            filepath = f"{args.output_folder}/{args.model_name}.json"
        else:
            filepath = f"{args.output_folder}/{args.model_name}.{args.start_index}-{args.end_index}.json"
    else:
        filepath = args.filepath
        
    if args.end_index < 0 or args.end_index > len(model_inputs):
        args.end_index = len(model_inputs)
    model_inputs = model_inputs[args.start_index:args.end_index]
    id_strs = id_strs[args.start_index:args.end_index]
    chat_history = chat_history[args.start_index:args.end_index]
    metadata = {key: metadata[key][args.start_index:args.end_index] for key in metadata}
    
    print("loading dataset ... done!")
    
    
    # speical handling
    stop_words = []
    if args.data_name == "commongen":
        stop_words = ["."]
    stop_token_ids = []
    if "yi-" in args.model_name.lower() and "chat" in args.model_name.lower():
        stop_token_ids = [7]
     
   
    outputs = [] 
    # Load the existing outputs
    if os.path.exists(filepath):
        with open(filepath) as f:
            formatted_outputs = json.load(f)
        for output_item in formatted_outputs:
            outputs.append(output_item["output"])
    num_skipped = len(outputs)
    print(f"We skipped the first {num_skipped} examples")
    
    
    todo_inputs = model_inputs[num_skipped:]
    
    if args.engine == "vllm":
        sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens, stop=stop_words, stop_token_ids=stop_token_ids, include_stop_str_in_output=True)
        for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_params, use_tqdm=False)
            outputs.extend([[x.outputs[0].text] for x in batch_outputs]) # TODO: enbale multiple generation 
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
        save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath)
    elif args.engine == "openai":        
        @retry_handler(retry_limit=10)
        def api(**kwargs):
            result = openai_chat_request(**kwargs) 
            return result
        
         
        for cur_id in tqdm(range(0, len(todo_inputs)), desc=f"Generating {args.model_name} from {args.start_index} to {args.end_index}"):
            input_text = todo_inputs[cur_id] 
            openai_args = {
                "model": args.model_name,
                "prompt": input_text,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "stop": stop_words,
            }  
            result = api(**openai_args)
            if args.data_name == "commongen":
                result = [r+"." for r in result]
            outputs.append(result) 
            save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath) 
    