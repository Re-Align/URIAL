import argparse  
import os
from tqdm import tqdm 
from models import DecoderOnlyModelManager, OpenAIModelManager
from datasets import load_dataset
import json 
import adapt_utils

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_type', default="clm", type=str)
    parser.add_argument('--model_path', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--model_name', default="x", type=str)
    parser.add_argument('--data_name', default="alpaca_eval", type=str)
    parser.add_argument('--model_cache_dir', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size',default=1, type=int)
    parser.add_argument('--beam_size',default=1, type=int)
    parser.add_argument('--output_folder',default="results/", type=str)
     
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=-1, type=int)
    parser.add_argument('--num_outputs',default=1, type=int)
    parser.add_argument('--max_output_tokens',default=128, type=int)
    parser.add_argument('--interval',default=1, type=int)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--bnb4', action='store_true')
    parser.add_argument('--gptq', action='store_true')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature',default=1, type=float)
    parser.add_argument('--top_p',default=1, type=float)
    parser.add_argument('--top_k',default=None, type=int)
    parser.add_argument('--repetition_penalty',default=1.0, type=float)
    parser.add_argument('--penalty_alpha',default=1.0, type=float)
    parser.add_argument('--no_repeat_ngram_size',default=0, type=int)
    parser.add_argument('--length_penalty',default=1.0, type=float)

    # test 
    parser.add_argument('--num_ref_prefix',default=0, type=int)
    # parser.add_argument('--num_instruct_prefix',default=0, type=int)
    
    # adapt mode 
    parser.add_argument('--template', default=None, type=str)
    parser.add_argument('--system_msg', default="default", type=str)
    parser.add_argument('--adapt_mode', default=None, type=str)
    parser.add_argument('--adapt_ckpt', type=str)
    parser.add_argument('--urial_prefix_path', type=str)
    parser.add_argument('--instruct_output_path', type=str, default=None)
    parser.add_argument('--retrieve_data_path', type=str, default=None)    
    parser.add_argument('--num_instruct_prefix', type=int, default=3)
    parser.add_argument('--num_demos', type=int, default=1)
    parser.add_argument('--eof_strings', default = "Query|</answer>", type=str)
    
    return parser.parse_args()

def main():
    
    args = parse_args() 
    if args.model_name == "x":
        args.model_name = args.model_path.rsplit("/")[1] 
    
    output_results_file = os.path.join(args.output_folder, f"{args.model_name}.json")
     
    os.system(f"mkdir -p {args.output_folder}")
    

    args_dict = vars(args)
    output_args_file = output_results_file.replace(".json", f".args.json")
    with open(output_args_file, "w") as f:
        json.dump(args_dict, f, indent=2)
     
            
    if args.data_name == "alpaca_eval":
        dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval") # , download_mode="force_redownload"
    elif args.data_name == "lima":
        dataset = load_dataset("GAIR/lima", split="test")
    elif args.data_name == "just_eval":
        dataset = load_dataset("re-align/just-eval-instruct", split="test") 
    else:
        with open(args.data_name) as f:
            dataset = json.load(f)
            
         
    
    pure_input_texts = []
    input_texts = []
    references = []
    id_strs = []
    for ind, item in enumerate(dataset):
        if args.data_name == "alpaca_eval":
            in_text = item["instruction"]
            ref_text = item["output"]
        elif args.data_name == "lima":
            in_text = item["conversations"][0]
            ref_text = "N/A"
        elif args.data_name in ["just_eval"]:
            in_text = item["instruction"]
            ref_text = "N/A"
        else:
            in_text = item["input_text"]
            ref_text = item["output_text"]
        id_strs.append(item.get("id", str(ind)))
        pure_input_texts.append(in_text)
        references.append(ref_text) 
        in_text = adapt_utils.preprocess(in_text, ind, args)
        input_texts.append(in_text)        
    
    if args.end_index < 0:
        end_index = len(input_texts)
    else: 
        if len(input_texts) <= args.end_index:
            end_index = len(input_texts)
        else:
            end_index = args.end_index
            
    if args.start_index != 0 or args.end_index > 0:
        output_results_file = output_results_file.replace(".json", f".{args.start_index}-{end_index}.json")
        
    if args.model_path == "openai":
        mm = OpenAIModelManager(args.model_name)
    else:
        mm = DecoderOnlyModelManager(args.model_path, args.model_name, cache_dir=args.model_cache_dir, 
                                    bf16=args.bf16, int8=args.int8, bnb4=args.bnb4, gptq=args.gptq,
                                    adapt_mode=args.adapt_mode, adapt_ckpt=args.adapt_ckpt)     
    mm.load_model()
 

    output_texts = []
    
    # if os.path.exists(output_results_file):
        # TODO
    
    def save_results(output_texts, remarks=None, start_index=0): 
        results = []
        cnt = 0 
        for p, i, r, o in zip(pure_input_texts[start_index:], input_texts[start_index:], references[start_index:], output_texts):
            ind = start_index+cnt
            # o_processed = adapt_utils.postprocess(p, o[0], ind, args)
            # o = [o_processed]
            o = [adapt_utils.postprocess(p, oo, ind, args) for oo in o]
            item = {"id": id_strs[ind], "pure_input": p, "input": i, "output": o, "reference": r}
            if remarks and len(remarks) > cnt:
                item["remark"] = remarks[cnt]
            results.append(item)
            cnt += 1
        with open(output_results_file, "w") as f:
            json.dump(results, f, indent=2)

    
    remarks = []
    adapt_utils.pure_inputs = pure_input_texts
    
    for i in tqdm(range(args.start_index, end_index, args.batch_size), desc=f'{args.model_name}:[{args.start_index}:{end_index}]', ncols=100):
        batch = input_texts[i:i+args.batch_size] 
        pure_batch = pure_input_texts[i:i+args.batch_size] 
        
        decoded_outputs = mm.infer_generate(batch, args, remarks=remarks, pure_input_data=pure_batch)
        
        if args.num_outputs <= -2: # generate multiple outputs but sample one by one to speed up
            for _ in range(abs(args.num_outputs)-1):
                remarks_now = []
                dos = mm.infer_generate(batch, args, remarks=remarks_now, pure_input_data=pure_batch)                     
                for current, previous in zip(dos, decoded_outputs):
                    previous.append(current[0])
                if remarks_now:
                    remarks[-1].append(remarks_now[0][0])

        output_texts += decoded_outputs
        if args.interval > 0 and len(output_texts) % args.interval == 0:
            save_results(output_texts, remarks, start_index=args.start_index)

    save_results(output_texts, remarks, start_index=args.start_index)
    print("Saved results to", output_results_file)

 
if __name__ == "__main__":
    main()