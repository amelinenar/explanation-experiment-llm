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

def get_url(llm, support_prompt):
    load_dotenv()
    key = os.getenv("OPEN_API_KEY")
    if llm == 'gpt-4o-mini':
        url_api = os.getenv("URL_gpt")
        payload = {
        "model": llm,
        "input" : f"{support_prompt}",
    }
        
    else:
        url_api = os.getenv("URL_API")
        # Prepare request payload
        payload = {
            "model": llm,
            "prompt": f"{support_prompt}",
        }

    return url_api,payload,key

def call_llm_api(llm, support_prompt):
    """
    Send prompt to LLM API and return aggregated response text.
    """
    url, payload, key = get_url(llm, support_prompt)
    if llm == 'gpt-4o-mini':
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {key}"}
        try:
            print("REQUEST MADE")
            response = requests.post(url, headers=headers, json=payload, stream=True)
            print(f"RESPONSE GOTTEN {response}")
            print(response.text)

        except requests.exceptions.RequestException as e:
            print("connection error:", e) 
    
    else:
        # Send request and handle response
        try:
            response = requests.post(url, data=json.dumps(payload), stream=False,)

        except requests.exceptions.RequestException as e:
            print("connection error:", e) 
            
    return response    
 
 
def print_response(response,path,llm):
    
    #text = " "
    if llm == 'gpt-4o-mini':
        text=" "
        with open(path, "a", encoding="utf-8") as f:
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
                                print(f"Writing to the file: {path}")
                                print("----------------------------------------------")
                                
                                f.write(text)
                            # print(text, end="", flush=True)
                            # f.write(text)    
    else:         
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
    
    response = call_llm_api(llm,support_prompt)
    print_response(response, des_path, llm)
    
 

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
    print_response(response, evaluation_path,llm)
    
    