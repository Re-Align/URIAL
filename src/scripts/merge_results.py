import json
import os 
import sys 

import os 

# Path to the directory containing the json files
# directory_path = 'result_dirs/retrieve+prefix_ref/K=5_N=2_ae/'
directory_path = sys.argv[1]
prefix = sys.argv[2]
mode = None
if len(sys.argv) > 3:
    mode = sys.argv[3]

target_filepath = os.path.join(directory_path, prefix+".json")


if os.path.exists(target_filepath):
    print("Exists:", target_filepath)
    exit()
# Get a list of all json files in the directory
json_files = []

for file in os.listdir(directory_path):
    if file == target_filepath:
        continue # skip the target file
    if file.startswith(prefix+".") and file.endswith(".json"):
        if mode == "infer":
            if "_multi" in file:
                continue 
        if len(file.split(".")) >= 2:
            ind = file.split(".")[-2]
            if len(ind.split("-")) == 2:
                try:
                    start = int(ind.split("-")[0])
                    end = int(ind.split("-")[1])
                    json_files.append([(start, end), file,])
                except Exception as e:
                    continue

# Sort the json files based on their names
json_files.sort(key=lambda x:x[0])

print(json_files)

# Initialize an empty list to store the merged data
merged_data = []

# Loop through the sorted json files and merge their lists
for (start_ind, end_ind), file in json_files:
    file_path = os.path.join(directory_path, file)
    with open(file_path, 'rb') as file:
        data_list = json.load(file)
        merged_data.extend(data_list)

# Now, the merged_data list contains the merged data from all json files
print("Merged data length:", len(merged_data))

with open(target_filepath, "w") as f:
    json.dump(merged_data, f, indent=2)