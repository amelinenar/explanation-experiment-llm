import sys


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
def explain_process(doc_path, llm, des_path):
    
                
        with open(doc_path, "r", encoding="utf-8") as f:
            documents = f.read()
            
        load_dotenv()
           
        url_api = os.getenv("URL_API")
        model = llm
        support_prompt = PromptManager.get_prompt(
            "first_test", 
    
    )

        url = url_api
        payload = {
            "model": model,            
            "prompt" : f"{support_prompt} \n\"\"\"\n{documents}\n\"\"\" "
     
        }

        response = requests.post(url, data=json.dumps(payload), stream=True)
        
        with open(des_path, "a", encoding="utf-8") as f:
            

            for line in response.iter_lines():
                if line:      
                    data = json.loads(line.decode('utf-8'))
                    text = data.get("response", "")
                    
                    print(text, end="", flush=True)
                    
                    f.write(text)
           
    
            print()  

