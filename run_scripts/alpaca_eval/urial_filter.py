import json 
import sys 
import random
from tqdm import tqdm 
random.seed(42)

def has_duplicates(s, K, N=2):
    # Split the string into words
    s = s.replace("\n", " [newline] ")
    words = s.split()

    # # Function to generate substrings longer than K words
    # def generate_substrings(words, K):
    #     for i in range(len(words) - K):
    #         for j in range(i + K + 1, len(words) + 1):
    #             yield ' '.join(words[i:j])

    # Count occurrences of each substring
    substring_counts = {}
    # for substring in generate_substrings(words, K):
    for i in range(len(words) - K):
        # Construct the substring
        substring = ' '.join(words[i:i + K + 1])
        if substring in substring_counts:
            substring_counts[substring] += 1
        else:
            substring_counts[substring] = 1
        if substring_counts[substring] > N:
            return True
 
    return False


def is_end_letter(s):
    return s.strip()[-1].isalpha()

def is_too_long(s, M=800, T=3000):
    return len(s.split()) >= M or len(s) >= T


filepath = sys.argv[1]
with open(filepath) as f:
    data = json.load(f)

for item in tqdm(data):
    filtered_outputs =[]
    back_ups = []
    for o in item["output"]:
        if is_too_long(o) and is_end_letter(o):
            continue
        back_ups.append(o)
        if has_duplicates(o, 5, 2):
            continue 
        filtered_outputs.append(o)
    # if "I like to host guests at my home from time to time, and I am gathering  recipes of different dishes and drinks to keep things interesting. I am interested in trying some Tunisian dishes. Can you give me a recipe Tunisian Kammounia?" in item["instruction"]:
    #     print(item["output"])
    #     print(json.dumps(filtered_outputs, indent=4))

    if len(filtered_outputs) == 0:
        if len(back_ups) == 0:
            item["output"] = random.choice(item["output"])
        else:
            item["output"] = random.choice(back_ups)
    else:
        item["output"] = random.choice(filtered_outputs)
    del item["model_input"]

with open(sys.argv[2], 'w') as f:
    json.dump(data, f, indent=4)
    print(f.name)