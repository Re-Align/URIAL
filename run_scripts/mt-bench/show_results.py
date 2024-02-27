"""
Usage:
python3 show_result.py --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import pandas as pd
import json
from tabulate import tabulate
from datasets import load_dataset

question_id_to_category = {}

dataset = load_dataset("philschmid/mt-bench")
for item in dataset["train"]:
    question_id_to_category[item["question_id"]] = item["category"]

PREFIX_DIR = "/net/nfs/mosaic/yuchenl/FastChat/fastchat/llm_judge/"

def print_df_with_tabulate(df, sort_by='score', ascending=False):
    sorted_df = df.sort_values(by=sort_by, ascending=ascending)
    print(tabulate(sorted_df, headers='keys', tablefmt='grid', showindex=True))


def display_result_single(args):
    global question_id_to_category
    if args.input_file is None:
        input_file = (
            f"{PREFIX_DIR}/data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    with open(input_file, "r") as f:
        lines = f.readlines()

    covered = set()
    deduped_items = []
    for line in lines[::-1]:
        item = json.loads(line)
        uqid = f"{item['model']}_{item['turn']}_{item['question_id']}"
        if uqid not in covered:
            item["category"] = question_id_to_category[item["question_id"]]
            deduped_items.append(item)
            covered.add(uqid)
    df_all = pd.DataFrame(deduped_items)
    print(df_all)
    # df_all = pd.read_json(input_file, lines=True)
    df = df_all[["question_id", "model", "score", "turn"]]
    df = df[df["score"] != -1]
    
    model_list = (
        df_all["model"].unique().tolist()
    )
    model_list = list(set(model_list))
    # only show URIAL-1k-help 
    model_list = [model for model in model_list if ("URIAL-1k-help" in model or "URIAL=" in model or "URIAL-" in model) and args.keyword in model]
    model_list += ["gpt-3.5-turbo", "gpt-4"]
    args.model_list = model_list
    if args.model_list is not None:
        df = df[df["model"].isin(args.model_list)]
 

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print_df_with_tabulate(df_1)

    if args.bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print_df_with_tabulate(df_2)

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print_df_with_tabulate(df_3)

    print(df)
    # Draw the full table in the order: Model, Turn 1, Turn 2, Average
    
    # Calculate turn-1-score
    turn_1_score = df[df['turn'] == 1].groupby('model')['score'].mean().reset_index(name='Turn 1')

    # Calculate turn-2-score
    turn_2_score = df[df['turn'] == 2].groupby('model')['score'].mean().reset_index(name='Turn 2')

    # Merge turn-1 and turn-2 scores
    merged_scores = pd.merge(turn_1_score, turn_2_score, on='model', how='outer')

    # Calculate overall average score
    overall_score = df.groupby('model')['score'].mean().reset_index(name='Overall')
    
    # Merge with overall average score
    final_table = pd.merge(merged_scores, overall_score, on='model')


    # final_table['model'] = final_table['model'].str.replace("-URIAL-0210v1", "")

    # Sort the table by overall score
    final_table_sorted = final_table.sort_values(by='Overall', ascending=False)
 

    print("\n########## Full table ##########")
 

    # Convert the DataFrame to a table in Markdown format without the index
    markdown_table = tabulate(final_table_sorted, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f")
    print(markdown_table) 



    # Reintegrate category information into `df`
    # This assumes there is a way to map each row in `df` back to its category, possibly using a separate mapping if `question_id` is available in `df` 

    # Calculate two-turn averages for each category
    df['category'] = df['question_id'].map(question_id_to_category)

    category_scores = df.groupby(['model', 'category'])['score'].mean().reset_index()

    # Pivot to get category averages for each model
    category_pivot = category_scores.pivot(index='model', columns='category', values='score').reset_index()

    # Rename columns to clarify they are category averages
    category_pivot.columns = [f"{col}" if col != 'model' else col for col in category_pivot.columns]

    # Merge with the final table
    final_table_with_categories = pd.merge(final_table, category_pivot, on='model', how='outer')

    # Sort, print, or export the extended table
    final_table_sorted_with_categories = final_table_with_categories.sort_values(by='Overall', ascending=False)
    final_table_sorted_with_categories['model'] = final_table_sorted_with_categories["model"].str.replace("-URIAL-0210v1", "")

    # Print the extended table
    print("\n########## Extended Full Table with Category Averages ##########")
    markdown_table_with_categories = tabulate(final_table_sorted_with_categories, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".2f")
    print(markdown_table_with_categories)

    # Convert the DataFrame to a list of dictionaries (one dictionary per row)
    records = final_table_sorted_with_categories.to_dict(orient='records')

    # Specify the file path for the JSONL file
    file_path = 'run_scripts/mt-bench/urial_bench.jsonl'

    # Write each dictionary to a separate line in the file
    with open(file_path, 'w') as file:
        for record in records:
            json_str = json.dumps(record)
            file.write(json_str + '\n')

    print(f"Table saved to {file_path}")


def display_result_pairwise(args):
    if args.input_file is None:
        input_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
        )
    else:
        input_file = args.input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df_all = df_all[(df_all["g1_winner"] != "error") & (df_all["g2_winner"] != "error")]

    model_list = (
        df_all["model_1"].unique().tolist() + df_all["model_2"].unique().tolist()
    )
    model_list = list(set(model_list))


    list_res = []
    # traverse df row by row
    for index, row in df_all.iterrows():
        if args.model_list is not None and row["model_1"] not in args.model_list:
            continue
        if args.baseline_model is not None:
            if args.baseline_model not in [row["model_1"], row["model_2"]]:
                continue
        if row["g1_winner"] == "tie" or row["g1_winner"] != row["g2_winner"]:
            list_res.append({"model": row["model_1"], "win": 0, "loss": 0, "tie": 1})
            list_res.append({"model": row["model_2"], "win": 0, "loss": 0, "tie": 1})
        else:
            if row["g1_winner"] == "model_1":
                winner = row["model_1"]
                loser = row["model_2"]
            else:
                winner = row["model_2"]
                loser = row["model_1"]
            list_res.append({"model": winner, "win": 1, "loss": 0, "tie": 0})
            list_res.append({"model": loser, "win": 0, "loss": 1, "tie": 0})

    df = pd.DataFrame(list_res)
    df = df.groupby(["model"]).sum()

    # remove baseline model
    if args.baseline_model is not None:
        df = df[df.index != args.baseline_model]
    # add win rate
    df["win_rate"] = df["win"] / (df["win"] + df["loss"] + df["tie"])
    df["loss_rate"] = df["loss"] / (df["win"] + df["loss"] + df["tie"])
    # each tie counts as 0.5 win + 0.5 loss
    df["win_rate_adjusted"] = (df["win"] + 0.5 * df["tie"]) / (
        df["win"] + df["loss"] + df["tie"]
    )
    # print(df.sort_values(by="win_rate", ascending=False))
    # print(df.sort_values(by="loss_rate", ascending=True))
    print(df.sort_values(by="win_rate_adjusted", ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, default="0210")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        display_result_func = display_result_single
    else:
        if args.mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)