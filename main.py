import sys
import os
import pandas as pd
import numpy as np

from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import read_all_dataset
from utils.utils import generate_results

from utils.constant import TASK
from utils.constant import LLMs
from utils.constant import dataset_names_for_task
from utils.constant import SUMMARIZATION_PROMPT,JUDGING_PROMPT


from prompt import explain_process,judging_explanation
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor

def create_fit_classifier(task_name,X_train,y_train, X_test,y_test,target_column,logs_path):
    if task_name == 'REGRESSION':
          
        from alpha_automl import AutoMLRegressor
        automl = AutoMLRegressor(time_bound=1, txt_file = logs_path)
        # Perform the search
        automl.fit(X_train, y_train)
        
    elif task_name == 'CLASSIFICATION':
        
        from alpha_automl import AutoMLClassifier
        automl = AutoMLClassifier(time_bound=1, verbose=True, txt_file = logs_path)
        automl.fit(X_train, y_train)

    elif task_name.lower() == 'time_series_forecast':
        
        from alpha_automl import AutoMLTimeSeries
        automl = AutoMLTimeSeries(time_bound=1, date_column='Date', target_column=target_column, txt_file = logs_path)
        automl.fit(X_train, y_train)

    elif task_name.lower() == 'semisupervised':
        
        from alpha_automl import AutoMLSemiSupervisedClassifier
        automl = AutoMLSemiSupervisedClassifier(time_bound=1, start_mode='spawn', txt_file = logs_path)
        automl.fit(X_train, y_train)
 
 
 
########################## main ########################

#change this directory for your machine
load_dotenv()
root_dir = os.getenv('ROOT_DIR')


if __name__ == "__main__":

    if sys.argv[1] == 'run_all':
        
        for task_name in TASK:
            print(" ")
            print('task name: ', task_name)
        
            datasets_dict = read_all_dataset(root_dir, task_name)
            
            tmp_output_directory = root_dir + '/results/' + task_name + '/' 
            
          
            for dataset_name in dataset_names_for_task[task_name]:
                print('\tdataset name: ', dataset_name)
                print(" ")
                
                x_train = datasets_dict[dataset_name][0]
                y_train = datasets_dict[dataset_name][1]
                x_test = datasets_dict[dataset_name][2]
                y_test = datasets_dict[dataset_name][3] 
                target_column = datasets_dict[dataset_name][4]


                print('-----------------START FITTING--------------')
                
                create_fit_classifier(task_name,x_train,y_train,x_test,y_test,target_column)
                
                print('--------------------DONE--------------------')
                print(" ")
                
                # for llm in LLMs:
                #     print('\t\tllm: ', llm)
                
                #     output_directory = tmp_output_directory + dataset_name + '/' + llm + '/'
                #     create_directory(output_directory)
                #     file_dir =  os.path.join(output_directory , 'summary_result.txt')
                
                #     dir_test = "/home/nguenang/Master_thesis/experiment_setup/results/regression/196_autoMpg/deepseek-r1:14b/summary_result.txt"
        
                #     explain_process('full_log_MainProcess.txt', llm, file_dir)
                #     print(" ")
                    
                #     print("TESTING ")
                    
                #     print(f'file dir: {file_dir}')
                
                #     for llm_judge in LLMs:
                #         judge_dir = os.path.join(output_directory, f'evaluation_'+llm_judge+'.txt')
                        
                #         if llm_judge is not llm:
                            
                            
                    
                #             judging_explanation('full_log_MainProcess.txt',file_dir,llm,judge_dir)
                
                    
                #     generate_results(root_dir, task_name, dataset_name, llm)
                    
                
                #     print(f"DONE SUMMARIZATION: output stored in {file_dir}")



    elif sys.argv[1] == 'fit':
        for task_name in TASK:
            print(" ")
            print('task name: ', task_name)
            
  
            datasets_dict = read_all_dataset(root_dir, task_name)
            
            tmp_output_directory = root_dir + '/results/' + task_name + '/' 
            
            print(f"DATASET : {dataset_names_for_task[task_name]}")


            for dataset_name in dataset_names_for_task[task_name]:
                print('\tdataset name: ', dataset_name)
                print(" ")
                
                x_train = datasets_dict[dataset_name][0]
                y_train = datasets_dict[dataset_name][1]
                x_test = datasets_dict[dataset_name][2]
                y_test = datasets_dict[dataset_name][3] 
                target_column = datasets_dict[dataset_name][4]
                
                output_directory = tmp_output_directory + dataset_name + '/'

                print('-----------------START FITTING--------------')
                
                create_fit_classifier(task_name,x_train,y_train,x_test,y_test,target_column, output_directory)
                
                print('--------------------DONE--------------------')
                print(" ")
                
    elif sys.argv[1] == 'summarize':
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            for task in TASK:
                
                print(f"\tTASK:   {task}       ")
                
                
                for dataset_name in dataset_names_for_task[task]:
                    
                   # logs_path = root_dir + '/results/' + task + '/' + dataset_name + '/full_log_MainProcess.txt'
                    logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
            
                    for llm in LLMs:
                        print('\t\tllm: ', llm)
                        
                        tmp_output_directory = root_dir + '/results/' + task + '/' + dataset_name + '/' + llm + '/'
                    
                    # output_directory = tmp_output_directory + dataset_name + '/' + llm + '/'

                        
                        for prompt in SUMMARIZATION_PROMPT:
                            
                            output_directory = tmp_output_directory + prompt + '/'
                            create_directory(output_directory)
                            file_dir =  os.path.join(output_directory , 'summary_result.txt')
                            
                            
                        
                            print('\t\t\tprompt: ', prompt)
                            print( " ")
                            
                            print(f"LOGS: {logs_path}")
                            args = (logs_path, llm, file_dir,prompt)
                           # print(logs_path)
                        
                            executor.submit(explain_process, *args)
                        
        print( " " )
                    
        
    elif sys.argv[1] == 'judge':
        
        with ThreadPoolExecutor(max_workers=2) as executor:

        
            for task in TASK:
                # print("\tjudging the TASK:", task)
                
                # print(f"\t\t  logs:  full_log_MainProcess_{task}.txt")
                
                
                # tmp_output_directory = root_dir + '/results/' + task + '/' 
                
                for dataset_name in dataset_names_for_task[task]:
                    
                    for llm in LLMs:
                        # print("\t\t\t using the LLM:", llm)
                        
                        tmp_output_directory = root_dir + '/results/' + task + '/' + dataset_name + '/' + llm + '/'
                        
                        for prompt in SUMMARIZATION_PROMPT:
                    
                            output_directory = tmp_output_directory + prompt + '/'
                            create_directory(output_directory)
                            file_dir =  os.path.join(output_directory , 'summary_result.txt')
                            
                            # print(f"\t\t Judging the summary: {file_dir}")
                            for llm_judge in LLMs:
                                
                                judge_dir = os.path.join(output_directory, f'evaluation_'+llm_judge+'.txt')
                            
                            # print("\tjudging the TASK:", task)
                    
                                print(f"\t\t judging the summary of the logs:  full_log_MainProcess_{task}.txt")
                                
                                print("\t\t\t using the LLM:", llm_judge," to judge")
                                
                                print(f"\t\t Judging the summary located in the directory: {file_dir}")
                                        
                                
                                for judge_prompt in JUDGING_PROMPT:
                                    
                                    print("JUDGING PROMPT:", judge_prompt)
                                    
                                    args = (f'full_log_MainProcess_{task}.txt',file_dir,llm_judge,judge_dir,judge_prompt)
                                    executor.submit(judging_explanation, *args)
                                    
                                    
                        
                                    
                                #    print(f"\t\t Judging the summary:  full_log_MainProcess_{task}.txt")


    else:
        '''this is the code to launch an experiment on a particular task, dataset and llm'''
        
        task_name = sys.argv[1]
        dataset_name = sys.argv[2]
        llm = sys.argv[3]
        
        output_directory  = root_dir + '/results/' + task_name + '/' + dataset_name+ '/' + llm + '/'
    
        test_dir_df_metrics = output_directory + 'df_metrics.csv'
        print(" ")
        print(f"TASK NAME: {task_name} DATASET NAME : {dataset_name} LLM : {llm}")
        print(" ")
        
        if os.path.exists(test_dir_df_metrics):
            print('Already done')
        else:

            create_directory(output_directory)
            datasets_dict = read_dataset(root_dir, task_name, dataset_name)
            
            
            x_train = datasets_dict[dataset_name][0]
            y_train = datasets_dict[dataset_name][1]
            x_test = datasets_dict[dataset_name][2]
            y_test = datasets_dict[dataset_name][3] 
            target_column = datasets_dict[dataset_name][4]
            
        
            print('-----------------START FITTING--------------')
            create_fit_classifier(task_name,x_train,y_train,x_test,y_test,target_column)
            print(" ")
            print('-----------------DONE FITTING---------------')

    
            file_dir =  os.path.join(output_directory , 'summary_result.txt')
    
          #  explain_process('full_log_MainProcess.txt', llm, file_dir)
            
          #  generate_results(root_dir, task_name, dataset_name, llm)
            
            print(" ")
            print('DONE SUMMARIZATION')
