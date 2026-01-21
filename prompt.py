# import sys
# import requests ,json
# import os
# from utils.constant import SUMMARIZATION_PROMPT, JUDGING_PROMPT
# from prompts.prompt_manager import PromptManager
# from dotenv import load_dotenv, dotenv_values

# sys.path.append("..")

# # support_prompt = PromptManager.get_prompt("first_test", document={"full_log.txt"})
# # helpdesk_prompt = PromptManager.get_prompt("first_test", pipeline="helpdesk", document={""})
 
# load_dotenv()
# url_api = os.getenv("URL_API")
   
# def explain_process(doc_path, llm, des_path,prompts):
    
#     with open(doc_path, "r", encoding="utf-8") as f:
#         documents = f.read()   
#     model = llm
#     url = url_api  
#     support_prompt = PromptManager.get_prompt(prompts, document = documents) 
#     payload = {
#         "model": model, 
#         "prompt": f"{support_prompt}",       
#         #"prompt" : f"{support_prompt} \n\"\"\"\n{documents}\n\"\"\" "
#         }

#     text = " "
#     try:  
#         response = requests.post(url, data=json.dumps(payload), stream=False)
#     except requests.exceptions.RequestException as e:
#         print("Connection error:", e)
 
#     try:
        
#         with open(des_path, "a", encoding="utf-8") as f:
                        
#             for line in response.content.splitlines:
#                 if line:
#                     data = json.loads(line.decode("utf-8"))
#                     text += data.get("response", "")
#             print(text, end="" )
#             print('response done')
#             f.write(text)
    
#     except FileExistsError:
#         print("File does not exist")


# def judging_explanation(log_path,summary_path, llm, evaluation_path,prompts):
        
#     try:   
#         with open(log_path, "r", encoding="utf-8") as f:
#             logs = f.read()
#     except FileNotFoundError:
#         print("The file does not exist")
#     try:   
#         with open(summary_path, "r", encoding="utf-8") as f:
#             summary = f.read()
#     except FileNotFoundError:
#         print("The file does not exist")
 
#     model = llm
#     support_prompt = PromptManager.get_prompt(prompts, logs = logs , summary = summary)

#     url = url_api
#     payload = {
#         "model": model,
#         "prompt": f"{support_prompt}",
#         #"prompt" : f"{support_prompt} \n\"\"\"\n{summary}\n\"\"\" "            
#         #"prompt" : f"{support_prompt} \n\"\"\"\n{logs}\n\"\"\"\n{summary}\n\"\"\" "
    
#     }

#     try:
    
#         response = requests.post(
#             url, 
#             data=json.dumps(payload), 
#             stream=False,
#             timeout=60
#             )


#         text = ""
#         for line in response.content.splitlines():
#             if line:      
#                 data = json.loads(line.decode('utf-8'))
#                 text += data.get("response", "")
#         print(text, end="")
#         print('response done') 
#         print(" ")   
#         print(f'Writing to the file: {evaluation_path}')
#         try:
#             with open(evaluation_path, "a", encoding="utf-8") as f:
#                 f.write(text)
#         except FileExistsError:
#             print("File doesnot exist")
        

    
#     except requests.exceptions.RequestException as e:
#         print("connection error:", e)
#         exit()
#     except:
#         print('regular exception')
    















# imports library
import sys
import os
import json
import requests
from utils.constant import SUMMARIZATION_PROMPT, JUDGING_PROMPT
from prompts.prompt_manager import PromptManager
from dotenv import load_dotenv, dotenv_values


# support_prompt = PromptManager.get_prompt("first_test", document={"full_log.txt"})
# helpdesk_prompt = PromptManager.get_prompt("first_test", pipeline="helpdesk", document={""})

# Load environment variables from .env file


def read_file(file_path,mode):
    """Read and return file content"""
    if mode == "r":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return_file = f.read()
        except FileNotFoundError:
            print("The file does not exist")
    if mode == "w":
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                return_file = f.write()
        except FileNotFoundError:
            print("The file does not exist")
                
    return return_file


def call_llm_api(llm, support_prompt):
    """
    Send prompt to LLM API and return aggregated response text.
    """
    load_dotenv()
    url_api = os.getenv("URL_API")
    model = llm
    url = url_api
    
    # Prepare request payload
    payload = {
        "model": model,
        "prompt": f"{support_prompt}",
    }
    # Send request and handle response
    try:
        response = requests.post(url, data=json.dumps(payload), stream=False, timeout=60,)

    except requests.exceptions.RequestException as e:
        print("connection error:", e) 
        
    return response    
 
 
def print_response(response,path):
            
    text = " "
    # Parse streamed response lines
    for line in response.content.splitlines():
        if line:
            data = json.loads(line.decode("utf-8"))
            text += data.get("response", "")

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
    print(f"Writing to the file: {path}")
    print("----------------------------------------------")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError:
        print("File doesnot exist")

    
def explain_process(doc_path, llm, des_path, prompts):
    """
    Reads a document, sends it to the LLM API for explanation,
    and appends the generated response to a destination file
    """

    # Read input document
    log_file = read_file(doc_path,"r")
    
    # Generate prompt using PromptManager
    support_prompt = PromptManager.get_prompt(prompts, document=log_file)
    response = call_llm_api(llm, support_prompt)
    print_response(response, des_path)
    
 

def judging_explanation(log_path, summary_path, llm, evaluation_path, prompts):
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
    print_response(response, evaluation_path)
    
    