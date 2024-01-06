import sys  
import time 
from functools import wraps
from typing import List 
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
from datasets import load_dataset
from tqdm import tqdm
from fastchat_conversation import get_conv_template
import json   

def apply_template(chat_history, model_name, urial=None):
    model_inputs = [] 
    if urial:
        # url = f"https://raw.githubusercontent.com/Re-Align/URIAL/main/urial_prompts/{urial}.txt"
        url = f"urial_prompts/{urial}.txt"
        print(f"Loading URIAL prompt from {url}")
        dataset = load_dataset("text", data_files=url, split="train", sample_by="document", download_mode="force_redownload")
        urial_prompt = dataset["text"][0]
    for chats in tqdm(chat_history, desc="Applying template", disable=True):
        if urial:
            conv = get_conv_template("urial")
            conv.set_system_message(urial_prompt)
        elif "tulu" in model_name.lower():
            conv = get_conv_template("tulu")
        elif "zephyr" in model_name.lower():
            conv = get_conv_template("zephyr")
        elif "llama-2" in model_name.lower():
            conv = get_conv_template("llama-2")
        elif "mixtral" in model_name.lower() or "mistral" in model_name.lower():
            conv = get_conv_template("mistral")
        elif "yi" in model_name.lower() and "chat" in model_name.lower():
            conv = get_conv_template("Yi-34b-chat")
        elif "vicuna" in model_name.lower():
            conv = get_conv_template("vicuna_v1.1")
        elif "gpt-" in model_name.lower():
            model_inputs.append(chats[0])
            continue
        else:
            print("ERROR: model_name not supported")
        for chat_id, chat in enumerate(chats):
            conv.append_message(conv.roles[chat_id%2], chat)
        conv.append_message(conv.roles[1], None)
        model_inputs.append(conv.get_prompt())
    return model_inputs

 
def load_eval_data(args, data_name=None, model_name=None):
    if data_name is None:
        data_name = args.data_name
    if model_name is None:
        model_name = args.model_name    
    chat_history = []
    id_strs = []
    metadata = {}
    if data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
        metadata = {"dataset": []}
    elif data_name == "just_eval":
        dataset = load_dataset("re-align/just-eval-instruct", split="test") 
        metadata = {"dataset": [], "source_id": []}
    elif data_name == "mt-bench":
        dataset = load_dataset("json", data_files="https://huggingface.co/spaces/lmsys/mt-bench/raw/main/data/mt_bench/question.jsonl", split="train")
        metadata = {"question_id": [], "category": []}        
        if args.mt_turn == 2:
            with open(args.mt_turn1_result, "r") as f:
                mt_turn1_result = json.load(f)
            id_to_turn1_result = {}
            for item in mt_turn1_result:
                id_to_turn1_result[item["question_id"]] = item["turn1_output"]
    elif data_name  == "commongen":
        dataset = load_dataset("allenai/commongen_lite", split="train") 
        metadata = {"id": [], "concept_set": []}
    else:
        print("ERROR: data_name not supported")
     
    for ind, item in enumerate(dataset):
        if data_name in ["alpaca_eval", "just_eval", "commongen"]:
            in_text = item["instruction"]    
            id_strs.append(item.get("id", str(ind)))
            chat_history.append([in_text])
        elif data_name == "mt-bench":
            if args.mt_turn == 1:
                chat_history.append([item["turns"][0]])
            elif args.mt_turn == 2:
                chat_history.append([item["turns"][0], 
                                     id_to_turn1_result[item["question_id"]], 
                                     item["turns"][1]]) 
            else:
                raise ValueError("mt_turn should be 1 or 2")
        for key in metadata: 
            metadata[key].append(item[key])
    print("start applying template")
    model_inputs = apply_template(chat_history, model_name, urial=args.urial)
    return id_strs, chat_history, model_inputs, metadata



def clear_output(output, model_name): 
    pass 
    return output


def save_outputs(args, id_strs, outputs, chat_history, metadata, model_inputs, filepath):
    formatted_outputs = []
    if args.data_name == "alpaca_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["instruction"] = chat_history[ind][0]
            output_item["output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["generator"] = args.model_name
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "just_eval":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["id"] = ind
            output_item["instruction"] = chat_history[ind][0]
            output_item["output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["generator"] = args.model_name
            output_item["dataset"] = metadata["dataset"][ind]
            output_item["source_id"] = metadata["source_id"][ind]
            output_item["datasplit"] = "just_eval"
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    elif args.data_name == "mt-bench":
        for ind in range(len(outputs)):
            output_item = {}
            output_item["question_id"] = metadata["question_id"][ind]
            output_item["category"] = metadata["category"][ind]
            output_item[f"turn{args.mt_turn}_output"] = clear_output(outputs[ind][0].rstrip(), args.model_name)
            output_item["model_id"] = args.model_name
            output_item["turn_id"] = args.mt_turn
            output_item["model_input"] = model_inputs[ind]
            formatted_outputs.append(output_item)
    with open(filepath, "w") as f:
        json.dump(formatted_outputs, f, indent=2)
        
 
def retry_handler(retry_limit=10):
    """
        This is an error handler for requests to OpenAI API.
        If will retry for the request for `retry_limit` times if the error is not a rate limit error.
        Otherwise, it will wait for the time specified in the error message and constantly retry.
        You can add specific processing logic for different types of errors here.

        Args:
            retry_limit (int, optional): The number of times to retry. Defaults to 3.
        
        Usage:
            @retry_handler(retry_limit=3)
            def call_openai_api():
                pass
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retried = 0
            while True:
                try:
                    sys.stdout.flush()
                    return func(*args, **kwargs)
                except Exception as e:
                    # if rate limit error, wait 2 seconds and retry
                    if isinstance(e, openai.error.RateLimitError):
                        words = str(e).split(' ')
                        try:
                            time_to_wait = int(words[words.index('after') + 1])
                        except ValueError:
                            time_to_wait = 5
                        # print("Rate limit error, waiting for {} seconds for another try..".format(time_to_wait))
                        time.sleep(time_to_wait) # wait 30 seconds
                        # print("Finished waiting for {} seconds. Start another try".format(time_to_wait))
                    elif isinstance(e, openai.error.APIError):
                        # this is because the prompt contains content that is filtered by OpenAI API
                        print("API error:", str(e))
                        if "Invalid" in str(e):
                            print("Invalid request, returning.")
                            raise e
                    else:
                        print(e.__class__.__name__+":", str(e))
                        if retried < retry_limit:
                            print(f"Retrying for the {retried + 1} time..")
                        else:
                            # finally failed
                            print("Retry limit reached. Saving the error message and returning.")
                            print(kwargs["prompt"])
                            raise e
                        retried += 1
        return wrapper
    return decorate

def openai_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
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
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
    
    response = openai.ChatCompletion.create(
        model=model,
        engine=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        **kwargs,
    )
    contents = []
    for choice in response['choices']:
        # Check if the response is valid
        if choice['finish_reason'] not in ['stop', 'length']:
            raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
        contents.append(choice['message']['content'])

    return contents
     
 