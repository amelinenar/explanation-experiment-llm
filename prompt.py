import sys
from utils.constant import SUMMARIZATION_PROMPT, JUDGING_PROMPT


sys.path.append("..")

from prompts.prompt_manager import PromptManager


# support_prompt = PromptManager.get_prompt(
#     "first_test", document={"full_log.txt"}
    
    
# )

# helpdesk_prompt = PromptManager.get_prompt(
#     "first_test", pipeline="helpdesk", document={""}
    
# )

import requests ,json
import os
from dotenv import load_dotenv, dotenv_values


import sys, json

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# with open("output.txt", "w", encoding="utf-8") as f:
#     sys.stdout = Tee(sys.stdout, f)

#     # Now you only need print()
#     data = json.loads(line.decode("utf-8"))
#     text = data.get("response", "")
#     print(text, end="", flush=True)
    
def explain_process(doc_path, llm, des_path,prompts):
    
                
        with open(doc_path, "r", encoding="utf-8") as f:
            documents = f.read()
            
        load_dotenv()
           
        url_api = os.getenv("URL_API")
        model = llm

        support_prompt = PromptManager.get_prompt(
            prompts, document = documents
)

        url = url_api
        payload = {
            "model": model, 
            "prompt": f"{support_prompt}",       
           #"prompt" : f"{support_prompt} \n\"\"\"\n{documents}\n\"\"\" "
     
        }

        try:
        
            response = requests.post(url, data=json.dumps(payload), stream=True)
        except requests.exceptions.RequestException as e:
            print("Connection error:", e)
            exit()
            
        #Check HTTP status 
        if response.status_code != 200:
            print("Server returned error:", response.status_code, response.text)
            exit()
        
        with open(des_path, "a", encoding="utf-8") as f:
            
           # original_stdout = sys.stdout
           # sys.stdout = Tee(sys.stdout, f)
            
            for line in response.iter_lines():
               
                data = json.loads(line.decode("utf-8"))
                text = data.get("response", "")
                print(text, end="", flush=True)
                f.write(text)
                
        
        
            #    sys.stdout = original_stdout
                    
            #         data = json.loads(line.decode('utf-8'))
            #         text = data.get("response", "")
                    
            #         print(text, end="", flush=True)
                    
            #         f.write(text)
           
    
            # print()  

def judging_explanation(log_path,summary_path, llm, des_path,prompts):
    
        with open(log_path, "r", encoding="utf-8") as f:
            logs = f.read()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
        
        load_dotenv()
        
        # print(log_path)
        # print(summary_path)
        
       # print("DOCUMENT READ READ")
           
        url_api = os.getenv("URL_API")
        model = llm
        support_prompt = PromptManager.get_prompt(
          prompts, logs = logs , summary = summary
    
    )

        url = url_api
        payload = {
            "model": model,
            "prompt": f"{support_prompt}",
            #"prompt" : f"{support_prompt} \n\"\"\"\n{summary}\n\"\"\" "            
            #"prompt" : f"{support_prompt} \n\"\"\"\n{logs}\n\"\"\"\n{summary}\n\"\"\" "
     
        }



        print("CONNECTIOM ESTABLISH ESTABLISHED")
        try:
        
            response = requests.post(url, data=json.dumps(payload), stream=True)
        except requests.exceptions.RequestException as e:
            print("connection error:", e)
            exit()
        
        with open(des_path, "a", encoding="utf-8") as f:
        
            
          #  print("RESPOSE GOT GET GOTTEN")

            for line in response.iter_lines():
                if line:      
                    data = json.loads(line.decode('utf-8'))
                    text = data.get("response", "")
                    
                    print(text, end="", flush=True)
                    
                    f.write(text)
            

            print()  