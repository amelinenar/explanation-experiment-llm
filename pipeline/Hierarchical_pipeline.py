import requests, json, os, re

from prompt import read_file,call_llm_api,print_response
from utils.constant import HIERARCHICAL_PROMPT
from prompts.prompt_manager import PromptManager


def normalize_triple_quotes(s: str) -> str:
    return re.sub(
        r'"""\s*(.*?)\s*"""',
        lambda m: json.dumps(m.group(1)),
        s,
        flags=re.DOTALL
    )


# def extract_pure_json(llm_output: str) -> dict:
#     """
#     Extract the first valid JSON object from LLM output.
#     """
#     # Remove markdown code fences if present
#     llm_output = re.sub(r"```(?:json)?", "", llm_output, flags=re.IGNORECASE).strip()

#     # Find first JSON object
#     start = llm_output.find("{")
#     end = llm_output.rfind("}")

#     if start == -1 or end == -1 or end <= start:
#         raise ValueError("No valid JSON object found in LLM output")

#     json_str = llm_output[start:end+1]
    
#     candidate = normalize_triple_quotes(json_str)


#     return json.loads(candidate)
 


import ast
import json
import re

def extract_pure_json(llm_output: str) -> dict:
    # strip markdown fences
    llm_output = re.sub(r"```(?:json)?\s*|\s*```", "", llm_output).strip()

    # brace matching
    stack = 0
    start = None

    for i, ch in enumerate(llm_output):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start is not None:
                candidate = llm_output[start:i+1]

                # try strict JSON
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

                #fallback: Python literal (handles """ strings)
                try:
                    return ast.literal_eval(candidate)
                except Exception:
                    pass

    raise ValueError("No valid JSON object found")




def logs_annotation(doc_path, llm, des_path):
    """
    Reads a log document, sends it to the LLM API for segmentation in to
    1. Dataset profiling
    2. Search space / pipeline grammar definition
    3. Pipeline search (e.g., MCTS simulations or rollouts)
    4. Pipeline instantiation and evaluation
    5. Final pipeline selection or termination
    """
    
    # Read input document
    log_file = read_file(doc_path,"r")

    # Generate prompt using PromptManager
    support_prompt = PromptManager.get_prompt("global_prompt", document=log_file)
    
    response = call_llm_api(llm,support_prompt)
    print_response(response, des_path,llm)
    
    
    # text = " "
    # # Parse streamed response lines
    # for line in response.content.splitlines():
    #     if line:
    #         data = json.loads(line.decode("utf-8"))
    #         text += data.get("response", "")
      
    # print(f"raw text { text }")        
    # # # Extracting pure json
    # # parsed_json = extract_pure_json(text)
    # # print(f"parsed_json { parsed_json }")
    # # try:
    # #     with open(des_path, "w") as f:
    # #         json.dump(parsed_json, f, indent=2)
    # # except FileExistsError:
    # #     print("File does not exist")
    
                        

    # # Output response and write to evaluation file
    # print(" ")
    # print("----------------------------------------------")
    # print("Got response")
    # print("----------------------------------------------")
    # print(text, end="")
    # print(" ")
    # print("----------------------------------------------")
    # print("response done")
    # print("----------------------------------------------")
    # print(" ")
    # print("----------------------------------------------")
    # print(f"Writing to the file: {des_path}")
    # print("----------------------------------------------")
 

    
def fact_aggregation(doc_path, llm, des_path):
    """
    Reads a log document, sends it to the LLM API for segmentation in to
    1. Dataset profiling
    2. Search space / pipeline grammar definition
    3. Pipeline search (e.g., MCTS simulations or rollouts)
    4. Pipeline instantiation and evaluation
    5. Final pipeline selection or termination
    """
    
    # Read input document
    log_file = read_file(doc_path,"r")

    # Generate prompt using PromptManager
    support_prompt = PromptManager.get_prompt("fact_aggregation", fact_json=log_file)
    
    response = call_llm_api(llm,support_prompt)
    
    if llm == "gpt-4o-mini":
        text=" "
        with open(des_path, "a", encoding="utf-8") as f:
            for output in response.json()['output']:
                    for content in output['content']:
                        if content:
                            print(content)
                            chunk = content.get("text", "")
                            if chunk:
                                print(chunk, end="", flush=True)
                                text += chunk
                                print(" ")
                                print("----------------------------------------------")
                                print("response done")
                                print("----------------------------------------------")
                                print(" ")
                                print("----------------------------------------------")
                                print(f"Writing to the file: {des_path}")
                                print("----------------------------------------------")
                                
                                f.write(text)
         
    
    else:
        pass
    
        # text = " "
        # # Parse streamed response lines
        # for line in response.content.splitlines():
        #     if line:
        #         data = json.loads(line.decode("utf-8"))
        #         text += data.get("response", "")
        
        # print(f"raw text { text }")        
        # # Extracting pure json
        # parsed_json = extract_pure_json(text)
        # print(f"parsed_json { parsed_json }")
        # try:
        #     with open(des_path, "w") as f:
        #         json.dump(parsed_json, f, indent=2)
        # except FileExistsError:
        #     print("File does not exist")
        
                            

        # # Output response and write to evaluation file
        # print(" ")
        # print("----------------------------------------------")
        # print("Got response")
        # print("----------------------------------------------")
        # print(parsed_json, end="")
        # print(" ")
        # print("----------------------------------------------")
        # print("response done")
        # print("----------------------------------------------")
        # print(" ")
        # print("----------------------------------------------")
        # print(f"Writing to the file: {des_path}")
        # print("----------------------------------------------")
    



def fact_extraction(doc_path, llm, des_path):
    """
    Reads a log document, sends it to the LLM API for segmentation in to
    1. Dataset profiling
    2. Search space / pipeline grammar definition
    3. Pipeline search (e.g., MCTS simulations or rollouts)
    4. Pipeline instantiation and evaluation
    5. Final pipeline selection or termination
    """
    
    # Read input document
    log_file = read_file(doc_path,"r")

    # Generate prompt using PromptManager
    support_prompt = PromptManager.get_prompt("fact_extraction", document=log_file)
    
    response = call_llm_api(llm,support_prompt)
    
    if (llm == "gpt-4o-mini") | (llm == "gpt-4.1-mini"):
        text=" "
        with open(des_path, "a", encoding="utf-8") as f:
            for output in response.json()['output']:
                    for content in output['content']:
                        if content:
                            # print(content)
                            chunk = content.get("text", "")
                            if chunk:
                                # print(chunk, end="", flush=True)
                                text += chunk
                                f.write(text)
                                # parsed_json = extract_pure_json(text)
                                # print(f"parsed_json { parsed_json }")
                                # try:
                                #     with open(des_path, "w") as f:
                                # json.dump(parsed_json, f, indent=2)
                                # except FileExistsError:
                                #     print("File does not exist")
                                
                                
                                print(" ")
                                print("----------------------------------------------")
                                print("response done")
                                print("----------------------------------------------")
                                print(" ")
                                print("----------------------------------------------")
                                print(f"Writing to the file: {des_path}")
                                print("----------------------------------------------")
                                
                                # f.write(text)
         
    
    
    else:
        
        text = " "
        # Parse streamed response lines
        for line in response.content.splitlines():
            if line:
                data = json.loads(line.decode("utf-8"))
                text += data.get("response", "")
        
        # print(f"raw text { text }")        
        # # Extracting pure json
        # parsed_json = extract_pure_json(text)
        # print(f"parsed_json { parsed_json }")
        try:
            with open(des_path, "w") as f:
                f.write(text)
                # json.dump(parsed_json, f, indent=2)
        except FileExistsError:
            print("File does not exist")
        
                            

        # Output response and write to evaluation file
        print(" ")
        print("----------------------------------------------")
        print("Got response")
        print("----------------------------------------------")
        print(text, end="")
        print(" ")
        print("----------------------------------------------")
        print("response done")
        print("----------------------------------------------")
        print(" ")
        print("----------------------------------------------")
        print(f"Writing to the file: {des_path}")
        print("----------------------------------------------")
        
        
 
# def phase_segmentation(doc_path, llm, des_path):
#     """
#     Reads a log document, sends it to the LLM API for segmentation in to
#     1. Dataset profiling
#     2. Search space / pipeline grammar definition
#     3. Pipeline search (e.g., MCTS simulations or rollouts)
#     4. Pipeline instantiation and evaluation
#     5. Final pipeline selection or termination
#     """
    
#     # Read input document
#     log_file = read_file(doc_path,"r")

#     # Generate prompt using PromptManager
#     support_prompt = PromptManager.get_prompt("log_filtering", document=log_file)
    
#     response = call_llm_api(llm,support_prompt)
    
#     if (llm == "gpt-4o-mini") | (llm == "gpt-4.1-mini"):
#         text=" "
#         with open(des_path, "a", encoding="utf-8") as f:
#             for output in response.json()['output']:
#                     for content in output['content']:
#                         if content:
#                             # print(content)
#                             chunk = content.get("text", "")
#                             if chunk:
#                                 # print(chunk, end="", flush=True)
#                                 text += chunk
#                                 f.write(text)
#                                 # parsed_json = extract_pure_json(text)
#                                 # print(f"parsed_json { parsed_json }")
#                                 # try:
#                                 #     with open(des_path, "w") as f:
#                                 # json.dump(parsed_json, f, indent=2)
#                                 # except FileExistsError:
#                                 #     print("File does not exist")
                                
                                
#                                 print(" ")
#                                 print("----------------------------------------------")
#                                 print("response done")
#                                 print("----------------------------------------------")
#                                 print(" ")
#                                 print("----------------------------------------------")
#                                 print(f"Writing to the file: {des_path}")
#                                 print("----------------------------------------------")
                                
#                                 # f.write(text)
         
    
    
#     else:
        
#         text = " "
#         # Parse streamed response lines
#         for line in response.content.splitlines():
#             if line:
#                 data = json.loads(line.decode("utf-8"))
#                 text += data.get("response", "")
        
#         # print(f"raw text { text }")        
#         # # Extracting pure json
#         # parsed_json = extract_pure_json(text)
#         # print(f"parsed_json { parsed_json }")
#         try:
#             with open(des_path, "w") as f:
#                 f.write(text)
#                 # json.dump(parsed_json, f, indent=2)
#         except FileExistsError:
#             print("File does not exist")
        
                            

#         # Output response and write to evaluation file
#         print(" ")
#         print("----------------------------------------------")
#         print("Got response")
#         print("----------------------------------------------")
#         print(text, end="")
#         print(" ")
#         print("----------------------------------------------")
#         print("response done")
#         print("----------------------------------------------")
#         print(" ")
#         print("----------------------------------------------")
#         print(f"Writing to the file: {des_path}")
#         print("----------------------------------------------")
 
 
 
 
 
 
 
 
 
        

    
def phase_segmentation(doc_path, llm, des_path):
    """
    Reads a log document, sends it to the LLM API for segmentation in to generate REGEX
    1. Dataset profiling
    2. Search space / pipeline grammar definition
    3. Pipeline search (e.g., MCTS simulations or rollouts)
    4. Pipeline instantiation and evaluation
    5. Final pipeline selection or termination
    """
    
    # Read input document
    log_file = read_file(doc_path,"r")

    # Generate prompt using PromptManager
    support_prompt = PromptManager.get_prompt("log_filtering", document=log_file)
    
    response = call_llm_api(llm,support_prompt)
    
    
    text = " "
    # Parse streamed response lines
    for line in response.content.splitlines():
        if line:
            data = json.loads(line.decode("utf-8"))
            text += data.get("response", "")
      
    print(f"raw text { text }")        
    # Extracting pure json
    parsed_json = extract_pure_json(text)
    print(f"parsed_json { parsed_json }")
    try:
        with open(des_path, "w") as f:
            json.dump(parsed_json, f, indent=2)
    except FileExistsError:
        print("File does not exist")
    
                        

    # Output response and write to evaluation file
    print(" ")
    print("----------------------------------------------")
    print("Got response")
    print("----------------------------------------------")
    print(parsed_json, end="")
    print(" ")
    print("----------------------------------------------")
    print("response done")
    print("----------------------------------------------")
    print(" ")
    print("----------------------------------------------")
    print(f"Writing to the file: {des_path}")
    print("----------------------------------------------")
 
 
 
 
 
 

def micro_summarization(json_file, llm, des_path,phases):
    """
      Summarizes the different phase of the segmentation phase and aggregate the result
    """
    
    # Read json file
    file = read_file(json_file, "r") 
    support_prompt_profilling = PromptManager.get_prompt("micro_data_summarization", logs=file)
    support_prompt_searchSpace = PromptManager.get_prompt("micro_search_space", logs=file)
    support_prompt_evaluation = PromptManager.get_prompt("micro_evaluation_prompt", logs=file)
    
    

    # micro_outputs = {}

    for phase in phases["phases"]:
        if phase["name"] == "Dataset profiling":
            support_prompt= support_prompt_profilling
        elif phase["name"] == "Search space / pipeline grammar definition":
            support_prompt = support_prompt_searchSpace
        elif phase["name"] == "Pipeline instantiation and evaluation":
            support_prompt= support_prompt_evaluation
        else:
            continue

        response = call_llm_api(llm, support_prompt)
        print_response(response, des_path,llm)
    
    

def meso_summarization():
    pass

def macro_summarization(json_file,llm, des_path,logs):
    
    # #read logs
    # logs = read_file(logs, "r")
    
    # Read json file
    file = read_file(json_file, "r") 
    support_prompt = PromptManager.get_prompt("global_prompt", aggregate_fact=file)
    response = call_llm_api(llm, support_prompt)
    print_response(response, des_path,llm)


def verification(macro_sum,log,llm, des_path):
    
    #Read json file
    file = read_file(macro_sum, "r") 
    
    #Read logs
    logs = read_file(log, "r")
    support_prompt = PromptManager.get_prompt("verification_prompt", macro_summary=file, logs = logs)
    response = call_llm_api(llm, support_prompt)
    
    if (llm == "gpt-4o-mini") | (llm == "gpt-4.1-mini"):
        text=" "
        with open(des_path, "a", encoding="utf-8") as f:
            for output in response.json()['output']:
                    for content in output['content']:
                        if content:
                            # print(content)
                            chunk = content.get("text", "")
                            if chunk:
                                # print(chunk, end="", flush=True)
                                text += chunk
                                
                                parsed_json = extract_pure_json(text)
                                print(f"parsed_json { parsed_json }")
                                # try:
                                #     with open(des_path, "w") as f:
                                json.dump(parsed_json, f, indent=2)
                                # except FileExistsError:
                                #     print("File does not exist")
                                
                                
                                print(" ")
                                print("----------------------------------------------")
                                print("response done")
                                print("----------------------------------------------")
                                print(" ")
                                print("----------------------------------------------")
                                print(f"Writing to the file: {des_path}")
                                print("----------------------------------------------")
    
    else:    
        text = " "
        # Parse streamed response lines
        for line in response.content.splitlines():
            if line:
                data = json.loads(line.decode("utf-8"))
                text += data.get("response", "")
                
        # Extracting pure json
        parsed_json = extract_pure_json(text)

        try:
            with open(des_path, "w") as f:
                json.dump(parsed_json, f, indent=2)
        except FileExistsError:
            print("File does not exist")
        
                            

        # Output response and write to evaluation file
        print(" ")
        print("----------------------------------------------")
        print("Got response")
        print("----------------------------------------------")
        print(parsed_json, end="")
        print(" ")
        print("----------------------------------------------")
        print("response done")
        print("----------------------------------------------")
        print(" ")
        print("----------------------------------------------")
        print(f"Writing to the file: {des_path}")
        print("----------------------------------------------")
    

def revised_summary(macro_sum, log, verification_path, llm, des_path):
    
    #Read json file
    file = read_file(macro_sum, "r") 
    
    #Read logs
    logs = read_file(log, "r")
    
    # #Read verification file
    # verification = read_file(verification, "r")
    
    import json

    with open(verification_path, "r") as f:
        verification = json.load(f)

    overall_assessment = verification["overall_assessment"]

    # verification = verify_global(macro_summary, full_logs)

    if overall_assessment != "PASS":
        support_prompt = PromptManager.get_prompt("revised_summary", macro_summary=file, verification = verification_path  , logs = logs)
        response = call_llm_api(llm, support_prompt)
        des_path = des_path  
        print_response(response, des_path,llm)    
    else:
        with open(macro_sum, "r" , encoding="utf-8") as f:
            content = f.read()
        with open(des_path, "w", encoding="utf-8") as f:
            f.write(content)
       
    
    
def judging_Hierarchal_sum(log_path, summary_path, llm, evaluation_path, prompts):
    """
    Evaluates a summary against logs using an LLM
    and writes the evaluation result to a file.
    """

    # Read log file
    logs = read_file(log_path, "r")

    # Read summary file
    summary = read_file(summary_path, "r")
    
    support_prompt = PromptManager.get_prompt(prompts, logs=logs, summary=summary)
    response = call_llm_api(llm, support_prompt)
    print_response(response, evaluation_path,llm)
    



    
def regex_filtering(log_text_path, phases):
    """
    Filter logs using regex patterns from JSON
    and preserve grouping by phase.
    """

    with open(log_text_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    output_lines = []

    for phase in phases["phases"]:
        phase_name = phase["name"]
        regex_list = phase["REGEX"]   # This is a LIST

        phase_matches = []

        for pattern in regex_list:    # ✅ iterate over list
            matches = re.findall(pattern, log_text)
            phase_matches.extend(matches)

        if phase_matches:
            output_lines.append(f"\n=== {phase_name} ===")
            output_lines.extend(phase_matches)

    return "\n".join(output_lines)

  













# def regex_filtering(log_text_path, phases):
    
#     """
#     filter logs using the differeng REGEX
#     """
#     # read the logs
#     with open(log_text_path, "r", encoding="utf-8") as f:
#         log_text = f.read()
    
    
#     for phase in phases["phases"]:
#         regex= phase["REGEX"]
#         filter_phase_log = reg_filter(regex, log_text)
        
        
#     return  filter_phase_log 
  
  
  
  
  
        
        
        
    #     if phase["name"] == "dataset_facts":
    #         support_prompt= support_prompt_profilling
    #     elif phase["name"] == "search_space_facts":
    #         support_prompt = support_prompt_searchSpace
    #     elif phase["name"] == "search_process_facts":
    #         support_prompt= support_prompt_evaluation
    #     elif phase["name"] == "evaluation_facts":
    #         support_prompt= support_prompt_evaluation
    #     elif phase["name"] == "termination_facts":
    #         support_prompt= support_prompt_evaluation
        
    #     else:
    #         continue

    # for p in j['phases']:
    # print(p['name'])
    # for r in p['REGEX']:
    #     print('  ',r)
    
     
    # for p in j['phases']:
    # print(p['name'])
    # for r in p['REGEX']:
    #     print('  ',r)
     
    
    
    
     
    # # Read json file
    # file = read_file(json_file, "r") 
    
    
    
    
    
    # support_prompt_profilling = PromptManager.get_prompt("micro_data_summarization", logs=file)
    # support_prompt_searchSpace = PromptManager.get_prompt("micro_search_space", logs=file)
    # support_prompt_evaluation = PromptManager.get_prompt("micro_evaluation_prompt", logs=file)
    
    

    # # micro_outputs = {}

    # for phase in phases["phases"]:
    #     if phase["name"] == "Dataset profiling":
    #         support_prompt= support_prompt_profilling
    #     elif phase["name"] == "Search space / pipeline grammar definition":
    #         support_prompt = support_prompt_searchSpace
    #     elif phase["name"] == "Pipeline instantiation and evaluation":
    #         support_prompt= support_prompt_evaluation
    #     else:
    #         continue

    #     response = call_llm_api(llm, support_prompt)
    #     print_response(response, des_path,llm)
     
     
     
     
     
     
     
        
    # # 1. Remove timestamps
    # log_text = re.sub(r'^\d{4}-\d{2}-\d{2} [\d:,]+ - ', '', log_text, flags=re.MULTILINE)

    # # 2. Remove MCTS simulation spam
    # log_text = re.sub(r'.*MCTS SIMULATION \d+.*\n?', '', log_text)

    # # 3. Remove MOVE ACTION lines
    # log_text = re.sub(r'.*MOVE ACTION:.*\n?', '', log_text)

    # # 4. Deduplicate identical lines
    # lines = log_text.splitlines()
    # seen = set()
    # filtered = []
    # for line in lines:
    #     if line not in seen:
    #         seen.add(line)
    #         filtered.append(line)

    # return "\n".join(filtered)


def reg_filter(reg_ex, log_text):
    return re.findall(reg_ex, log_text)  