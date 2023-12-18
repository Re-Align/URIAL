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
    parser.add_argument('--vllm_url', default="http://localhost:2333/generate", type=str)
    parser.add_argument('--model_name', default="Llama-2-70b-hf", type=str)
    parser.add_argument('--data_name', default="alpaca_eval", type=str) 
    parser.add_argument('--urial_name', default="inst_help", type=str)
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--temperature',default=0, type=float)
    
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
    urial_inputs = []
    for text_in in pure_input_texts:
        urial_in = f"{urial_prompt}\n# Query:\n```{text_in}```\n\n# Answer:\n```\n"
        urial_inputs.append(urial_in)
    return urial_inputs


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
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
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
            output_item["output"] = outputs[ind][0].replace("```", " ").rstrip()
            output_item["generator"] = args.model_name + "+URIAL=" + args.urial_name + f".p={args.top_p}.t={args.temperature}"
            output_item["dataset"] = metadata["dataset"][ind]
            formatted_outputs.append(output_item)
    
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)
        

if __name__ == "__main__":
    args = parse_args()
    filepath = f"vllm_outputs/{args.model_name}.URIAL={args.urial_name}.p={args.top_p}.t={args.temperature}.json" 
    
    id_strs, pure_input_texts, metadata = load_eval_data(args.data_name)
    urial_prompt = load_urial_prompt(args.urial_name)
    urial_inputs = apply_urial(pure_input_texts, urial_prompt)

    # if the file exists, load it
    
    outputs = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            formatted_outputs = json.load(f)
        for output_item in formatted_outputs:
            outputs.append([output_item["output"]])
    
    start_index = len(outputs)
    print(f"We skipped the first {start_index} examples")
    for ind, prompt in tqdm(enumerate(tqdm(urial_inputs[start_index:]))):
        output = vllm_request(prompt=prompt, n=1, stop=["# Query"], max_tokens=2048)
        outputs.append(output)
        if len(outputs) % 5 == 0:
            save_outputs(args, outputs, pure_input_texts, metadata, filepath)
    save_outputs(args, outputs, pure_input_texts, metadata, filepath)
    # args.model_name = args.model_name.split("/")[1]
    
    