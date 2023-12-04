import json
from datasets import load_dataset
import sys 
import os 
from sentencepiece import SentencePieceProcessor


import re

def remove_duplicate_blocks(s, delimiter="\n"):
    """Remove duplicate blocks of text from the string s.
    
    Args:
    - s (str): The input string containing repetitive blocks.
    - delimiter (str): The delimiter between blocks. Default is two newlines.
    
    Returns:
    - A string with duplicate blocks removed.
    """
    blocks = s.split(delimiter)
    cleaned_blocks = [re.sub(r'\d+\.', '', block).strip() for block in blocks]
    unique_blocks = []
    seen = set()
    
    for i, block in enumerate(cleaned_blocks):
        if block not in seen:
            seen.add(block)
            unique_blocks.append(blocks[i])
    if len(unique_blocks) < len(cleaned_blocks):
      unique_blocks = unique_blocks[:-1]
    return delimiter.join(unique_blocks)


def llama_token_counter(input_text):
    sp = SentencePieceProcessor(model_file="src/scripts/llama2-tokenizer.model")
    tokens = sp.EncodeAsIds(input_text)
    return len(tokens)

def has_continuous_words(sentence, k):
    words = sentence.split()
    consecutive_count = 0

    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            consecutive_count += 1
            if consecutive_count >= k - 1:
                return True
        else:
            consecutive_count = 0

    return False

def has_repeated_subsequence(sentence, m):
    words = sentence.split()
    word_sequence = " ".join(words[:m])  # Initial sequence of M words
    word_sequence_count = 1

    for i in range(m, len(words)):
        current_sequence = " ".join(words[i - m + 1 : i + 1])
        if current_sequence == word_sequence:
            word_sequence_count += 1
            if word_sequence_count >= 2:  # At least 2 repetitions for failure
                return True
        else:
            word_sequence = current_sequence

    return False


result_file = sys.argv[1]


run_name = result_file.split("/")[-1].replace(".json", "")


if os.path.exists(result_file.replace(".json", ".to_eval.json")):
    print("Exists:", result_file.replace(".json", ".to_eval.json"))
    # exit()

with open(result_file, "r") as f:
    load_results = json.load(f)

# results = [None for _ in range(len(load_results))]
# for item in load_results:
#     results[item["id"]] = item 
results = sorted(load_results, key=lambda x: int(x["id"]))

eval_set = load_dataset("re-align/just-eval-instruct", split="test")


outputs = []
eid = 0 
num_token_list = []
num_truncated = 0
num_long = 0
for example in eval_set:
    if eid >= len(results):
        break 
    example["generator"] = result_file.replace("result_dirs/", "")
    example["datasplit"] = "just_eval"
    
    if example["instruction"] not in results[eid]["pure_input"]:
        print(example["instruction"], results[eid]["pure_input"])
        exit()
    
    model_outputs = results[eid]["output"]
    if "falcon" in example["generator"]:
        model_outputs = [o.replace("User", "").replace("<|endoftext|>", "").strip() for o in results[eid]['output']]
    
        
    # First, let's keep the finished outputs first 
    good_model_outputs = []
    for mo in model_outputs:
        mo = mo.strip()
        if has_continuous_words(mo, k=2):
            continue
        if has_repeated_subsequence(mo, m=2):
            continue
        # if "</s>" in mo:
        #     good_model_outputs.append(mo)
        # if mo.replace("\n", "").endswith("```"):
        #     good_model_outputs.append(mo)
        good_model_outputs.append(mo)
    
    if len(good_model_outputs) > 0:
        model_outputs = good_model_outputs        
    
    
    # Some in-context based model_outputs can be too long
    o = model_outputs[0]
    
    # Select the longest one from the multiple results for in-context sampling 
    if len(model_outputs) > 1:
        for p in model_outputs:
            if len(p.split()) > len(o.split()):
                if '```' in o and '```' not in p:
                    pass
                else:
                    o = p
                    
    if "```\n#" in o:
        o = o[:o.rindex("```\n#")]
    if "```" in o[-15:]:
        o = o[:o.rindex("```")]
    if "#" in o[-5:]:
        o = o[:o.rindex("#")]
    if "gpt" not in example["generator"]:
    #     if '```' not in o and '</s>' not in o:
    #         # some are too long 
    #         o += " ... (truncated)"
        if llama_token_counter(o) > 1000:
            o = remove_duplicate_blocks(o, delimiter="\n")
            if llama_token_counter(o) > 1000:
                o += " ... (truncated)"
                num_truncated += 1
    
    
    
    example["output"] = o.replace("```", "").replace("</s>", "").strip()
    example["num_tokens"] = llama_token_counter(o)
    num_token_list.append(example["num_tokens"])
    outputs.append(example)
    eid += 1
    

with open(result_file.replace(".json", ".to_eval.json"), "w") as f:
    json.dump(outputs, f, indent=2)
    print(f.name)
    print("Average of num_tokens:", sum(num_token_list) / len(num_token_list))
    print("num_truncated:", num_truncated)