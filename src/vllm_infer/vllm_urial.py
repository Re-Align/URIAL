import requests
from typing import List 
import argparse
from datasets import load_dataset
import urllib.request
from tqdm import tqdm
import json
import os 



def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--vllm_mode', default="api", type=str) # api or local
    parser.add_argument('--vllm_url', default="http://localhost:2333/generate", type=str)
    parser.add_argument('--output_folder', default="vllm_outputs", type=str)
    parser.add_argument('--download_dir', default=None, type=str)
    
    parser.add_argument('--model_name', default="meta-llama/Llama-2-70b-hf", type=str)
    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--tokenizer_mode', type=str, default="auto") 
    parser.add_argument('--data_name', default="alpaca_eval", type=str) 
    parser.add_argument('--urial_name', default="none", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--temperature',default=0, type=float)
    parser.add_argument('--repetition_penalty',default=1, type=float)
    parser.add_argument('--max_tokens',default=7500, type=int)
    return parser.parse_args()

def load_eval_data(data_name="alpaca_eval"):
    pure_input_texts = []
    id_strs = []
    metadata = {}
    if data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        metadata = {"dataset": []}
    elif data_name == "just_eval":
        dataset = load_dataset("re-align/just-eval-instruct", split="test") 
        metadata = {"dataset": [], "source_id": []}
    else:
        print("ERROR: data_name not supported")
            
    for ind, item in enumerate(dataset):
        in_text = item["instruction"]    
        id_strs.append(item.get("id", str(ind)))
        pure_input_texts.append(in_text)
        for key in metadata:
            metadata[key].append(item[key])
        
    return id_strs, pure_input_texts, metadata

def load_urial_prompt(split_name="inst_help"):
    urial_url = f"https://raw.githubusercontent.com/Re-Align/URIAL/main/urial_prompts/{split_name}.txt"
    urial_prompt = urllib.request.urlopen(urial_url).read().decode('utf-8')
    return urial_prompt

def apply_urial(pure_input_texts, urial_prompt):
    model_inputs = []
    for text_in in pure_input_texts:
        urial_in = f"{urial_prompt}\n# Query:\n```{text_in}```\n\n# Answer:\n```\n"
        model_inputs.append(urial_in)
    return model_inputs


def apply_template(pure_input_texts, model_name):
    model_inputs = []
    for instruction in pure_input_texts:
        if "tulu-2" in model_name.lower():
            prompt_url = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/src/alpaca_eval/models_configs/tulu-2-dpo-70b/prompt.txt"
        elif "llama" in model_name.lower() and "chat" in model_name.lower():
            prompt_url = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/src/alpaca_eval/models_configs/llama-2-7b-chat-hf/prompt.txt"
        elif "yi" in model_name.lower() and "chat" in model_name.lower():
            prompt_url = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/src/alpaca_eval/models_configs/Yi-34B-Chat/prompt.txt"
        prompt = urllib.request.urlopen(prompt_url).read().decode('utf-8')
        model_intput = prompt.replace("{instruction}", instruction)
        model_inputs.append(model_intput)
    return model_inputs

def vllm_request(
    url='http://localhost:2333/generate',
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    repetition_penalty: float=1.0,
    prompt: str=None,
    n: int=1, 
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    
    """
    # Call openai api to generate aspects
    assert prompt is not None  
    

    # Define the request payload as a dictionary
    payload = {
        "prompt": prompt,
        "use_beam_search": False,
        "n": n,
        "best_of": n,
        "temperature": temperature,
        "stop": stop,
        "max_tokens": max_tokens,
        "top_p": top_p,  
        "repetition_penalty": repetition_penalty,
    }

    # Make a POST request to the API
    response = requests.post(url, json=payload)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        response = response.json()
        return [x.replace(prompt, "") for x in response["text"]]
    else:
        # Print an error message if the request failed
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def save_outputs(args, outputs, pure_input_texts, metadata, filepath):
    formatted_outputs = []
    if args.data_name == "alpaca_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["instruction"] = pure_input_texts[ind]
            if args.urial_name != "none":
                output_item["output"] = outputs[ind][0].replace("```", " ").rstrip()
                output_item["generator"] = args.model_name + "+URIAL=" + args.urial_name + f".p={args.top_p}.t={args.temperature}.r={args.repetition_penalty}"
            else:
                output_item["output"] = outputs[ind][0].rstrip()
                output_item["generator"] = args.model_name + f".p={args.top_p}.t={args.temperature}"
            output_item["dataset"] = metadata["dataset"][ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "just_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["id"] = ind
            output_item["instruction"] = pure_input_texts[ind]
            if args.urial_name != "none":
                output_item["output"] = outputs[ind][0].replace("```", " ").rstrip()
                output_item["generator"] = args.model_name + "+URIAL=" + args.urial_name + f".p={args.top_p}.t={args.temperature}.r={args.repetition_penalty}"
            else:
                output_item["output"] = outputs[ind][0].rstrip()
                output_item["generator"] = args.model_name + f".p={args.top_p}.t={args.temperature}"
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["source_id"] = metadata["source_id"][ind]
            output_item["datasplit"] = "just_eval"
            formatted_outputs.append(output_item)
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)


def vllm_local_completions():
    pass         

if __name__ == "__main__":
    args = parse_args()
    
    # Model loading 
    if args.vllm_mode == "local":
        from vllm import LLM, SamplingParams # when running locally 
        llm = LLM(model=args.model_name, tokenizer=args.model_name, tensor_parallel_size=args.tensor_parallel_size, download_dir=args.download_dir, dtype=args.dtype, tokenizer_mode=args.tokenizer_mode)        
        
    # Decide the output filepath 
    if "/" in args.model_name:
        args.model_name = args.model_name.split("/")[1]   
    
    os.system(f"mkdir -p {args.output_folder}")
    if args.urial_name != "none":
        filepath = f"{args.output_folder}/{args.model_name}.URIAL={args.urial_name}.p={args.top_p}.t={args.temperature}.r={args.repetition_penalty}.json" 
    else:
        filepath = f"{args.output_folder}/{args.model_name}.p={args.top_p}.t={args.temperature}.r={args.repetition_penalty}.json"
    
    
    # Data loading 
       
    id_strs, pure_input_texts, metadata = load_eval_data(args.data_name)
    if args.urial_name != "none":
        urial_prompt = load_urial_prompt(args.urial_name)
        model_inputs = apply_urial(pure_input_texts, urial_prompt)
    else:
        model_inputs = apply_template(pure_input_texts, args.model_name)

    # if the file exists, load it and skip the processed examples
    outputs = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            formatted_outputs = json.load(f)
        for output_item in formatted_outputs:
            outputs.append([output_item["output"]])
    start_index = len(outputs)
    print(f"We skipped the first {start_index} examples")
    
    
    stop_words = []
    if args.urial_name != "none":
        stop_words = ["# Query"]  
    stop_token_ids = []
    if "yi-" in args.model_name.lower() and "chat" in args.model_name.lower():
        stop_token_ids = [7]
    
    # Run the inference 
    if args.vllm_mode == "api":
        for ind, prompt in tqdm(enumerate(tqdm(model_inputs[start_index:]))):
            output = vllm_request(url=args.vllm_url, prompt=prompt, n=1, stop=stop_words, max_tokens=args.max_tokens, top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty)
            outputs.append(output)
            # if len(outputs) % 2 == 0:
            save_outputs(args, outputs, pure_input_texts, metadata, filepath)
    elif args.vllm_mode == "local":
        sampling_params = SamplingParams(top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens, stop=stop_words, stop_token_ids=stop_token_ids)
        todo_inputs = model_inputs[start_index:]
        for cur_id in tqdm(range(0, len(todo_inputs), args.batch_size), desc="Batching"):
            batch_inputs = todo_inputs[cur_id:cur_id+args.batch_size]
            batch_outputs = llm.generate(batch_inputs, sampling_params)
            outputs.extend([[x.outputs[0].text] for x in batch_outputs])
            # print(outputs[-1])
            save_outputs(args, outputs, pure_input_texts, metadata, filepath)
    save_outputs(args, outputs, pure_input_texts, metadata, filepath)
    # args.model_name = args.model_name.split("/")[1]
    
    