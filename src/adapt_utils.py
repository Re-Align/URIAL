import json 
import os 
import sys 
from data.long_demos import demo_data, demo_data_short

pure_inputs = None
retrieve_data = None 
chat_output_data = None 
template_prefix = None 
template_apply_prefix = None 
style_prefix = None 
urial_prefix = None 

def shorten(s, n=300):
    ws = s.split(" ")
    if len(ws) <= n:
        return s
    else:
        return " ".join(ws[:n]) + " ... "

def preprocess(in_text, ind, args):
    
    
    global retrieve_data, chat_output_data, template_prefix, template_apply_prefix, style_prefix, urial_prefix
    
    
    
    if args.adapt_mode is None:
        # add special tokens 
        if "chat" in args.model_name and "llama" in args.model_name.lower():
            # system = ""
            if args.system_msg == "default":
                system = """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n"""
            elif args.system_msg == "none":
                system = ""
            else:
                system = f"<<SYS>> {args.system_msg} <</SYS>>"
            # system = """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as briefly as possible, while being safe. Keep the answer short. \n<</SYS>>\n"""
            in_text = f"<s>[INST] {system} \n {in_text} [/INST] "
            # in_text = f"[INST] {in_text} [/INST] "
            return in_text
        elif "vicuna" in args.model_name:
            # system = ""
            if args.system_msg == "default":
                system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            elif args.system_msg == "none":
                system = ""
            else:
                system = args.system_msg
            in_text = f"{system}\n USER: {in_text} \nASSISTANT:"
            # in_text = f"[INST] {in_text} [/INST] "
            args.eof_strings += "|\nUser:"
            return in_text
        elif "falcon" in args.model_name.lower() and "instruct" in args.model_name:
            system = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
            in_text = f"{system}\nUser: {in_text} \nAssistant:"
            return in_text
        elif "mistral" in args.model_name.lower() and "instruct" in args.model_name.lower():
            system = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
            in_text = f"<s>[INST] {system} {in_text} [/INST]"
            return in_text
        elif "zephyr" in args.model_name.lower():
            system = "You are a friendly chatbot who always answer as helpfully as possible, while being safe.</s>"
            in_text = f"<|system|>\n {system} \n <|user|>\n {in_text} </s> \n <|assistant|>\n"
            return in_text
        else:
            return in_text
    
    elif args.adapt_mode == "none":
        return in_text
    
    elif args.adapt_mode == "zero_shot":   
        in_text = f"# Query: \n ```{in_text}``` \n\n \n# Answer: \n ``` "
        return in_text
    
    elif args.adapt_mode == "urial": 
        if urial_prefix is None:
            with open(args.urial_prefix_path) as f:
                urial_prefix = f.read()
        # qid = urial_prefix.count('Query')+1
        # in_text = f"{urial_prefix}\n# Query:\n<query> {in_text} </query>\n\n# Answer:\n<answer>\n"
        in_text = f"{urial_prefix}\n# Query:\n```{in_text}```\n\n# Answer:\n```\n"
        return in_text
    
    
    elif args.adapt_mode == "restyle":
        if retrieve_data is None and os.path.exists(args.retrieve_data_path):
            with open(args.retrieve_data_path) as f:
                retrieve_data = json.load(f)
        assert in_text in retrieve_data[ind]["pure_input"]  
        if style_prefix is None:
            with open(args.adapt_ckpt) as f:
                style_prefix = f.read()
        qid = style_prefix.count('## Query')+1
        init_answer =  retrieve_data[ind]["output"][0]
        in_text = f"{style_prefix}\n## Query {qid}\n\n```\n{in_text}\n```\n\n## Initial Answer {qid}\n\n```\n{init_answer}\n```\n\n## Revised Answer {qid}\n\n```"
        return in_text
     
    if "retrieve" in args.adapt_mode:
        if retrieve_data is None and os.path.exists(args.retrieve_data_path):
            with open(args.retrieve_data_path) as f:
                retrieve_data = json.load(f)    
                         
        if args.retrieve_data_path not in ["none", "random", "long"]: 
            assert in_text in retrieve_data[ind]["instruction"] 
            
        _user = "# Query:"
        _assistant = "\n \n# Answer:"
        in_text = f"{_user} \n ```{in_text} ``` \n"
        in_text += f"{_assistant} \n ``` " # for the model to fill in 
        
        
        N = args.num_demos
        cnt = 0 
        rex = []
        if args.retrieve_data_path not in ["none", "random", "long"]: 
            item = retrieve_data[ind]
            rex = item["rex"]
        elif args.retrieve_data_path == "random": 
            rex = demo_data_short[:N]
        elif args.retrieve_data_path == "long":
            rex = demo_data[:N]
        elif args.retrieve_data_path == "none":
            for _ in range(N):
                rex.append({"instruction":"[This is the placeholder for a user query.]", "input": "", "output": "[This is the placeholder for a helpful response to the above query.]"})
        for r in rex: # the most similar one is the last one
            if 'input' in r and len(r['input']) > 1:
                continue
            demo = f"{_user} \n ``` {r['instruction'] + ' ' + r.get('input', '')} ``` \n"
            demo += f"{_assistant} \n ``` {shorten(r['output'])} ``` " 
            demo += "\n --- \n" 
            # if len(input_text.split()) >= 2500:
            #     break 
            in_text = demo + in_text
            cnt += 1
            if cnt >= N:
                break  
    
    if args.adapt_mode == "retrieve":
        return in_text
    elif args.adapt_mode == "retrieve+prefix":
        return in_text 
    return in_text



def postprocess(in_text, output_text, ind, args):
    global retrieve_data, chat_output_data
    if not args.adapt_mode:
        return output_text
    if "</answer>" in output_text:
        output_text = output_text[:output_text.index("</answer>")]
    if "Query" in output_text:
        output_text = output_text[:output_text.index("Query")]
    if "---" in output_text[-10:]:
        # find the index of the last "---" in output_text
        output_text = output_text[:output_text.rindex("---")]
    if args.adapt_mode == "style":
        if "##" in output_text:
            output_text = output_text[:output_text.index("##")]
    if "prefix_ref" in args.adapt_mode:
        assert in_text in chat_output_data[ind]["input"]
        K = args.num_instruct_prefix
        chat_output = chat_output_data[ind]["output"][0].split(" ")
        prefix = " ".join(chat_output[:K])
        output_text = prefix + " " + output_text
        return output_text
    return output_text
    