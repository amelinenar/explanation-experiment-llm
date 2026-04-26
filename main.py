import pstats
import sys
import os
import time
import pandas as pd
import numpy as np

from itertools import product
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import read_all_dataset
from utils.utils import generate_results
from utils.utils import filter_automl_logs

from utils.constant import TASK
from utils.constant import LLMs, LLMs_judge
from utils.constant import dataset_names_for_task
from utils.constant import SUMMARIZATION_PROMPT,JUDGING_PROMPT, HIERARCHICAL_PROMPT


from prompt import explain_process,judging_explanation
from dotenv import load_dotenv
from result_script import write_csv, result_script
# from pipeline.stage0_phase_segmentation import explain_process_T
# from pipeline.stage1_dataprofiling import judging_explanation_T
from pipeline.Hierarchical_pipeline import phase_segmentation, micro_summarization,extract_pure_json,macro_summarization,verification,revised_summary,regex_filtering,fact_aggregation
from pipeline.Hierarchical_pipeline import fact_extraction
from filtering_logs import filtering_logs, remove_consecutive_duplicates

from concurrent.futures import ThreadPoolExecutor
import gc




def create_fit_classifier(task_name,X_train,y_train,target_column,logs_path,date_column, original_logs):
    if task_name == 'REGRESSION':
          
        from alpha_automl import AutoMLRegressor
        automl = AutoMLRegressor(time_bound=1, txt_file = logs_path, output_folder=original_logs)
        # Perform the search
        automl.fit(X_train, y_train)
        
    elif task_name == 'CLASSIFICATION':
        
        from alpha_automl import AutoMLClassifier
        automl = AutoMLClassifier(time_bound=1, verbose=True, txt_file = logs_path,  output_folder=original_logs)
        automl.fit(X_train, y_train)

    elif task_name.lower() == 'time_series_forecast':
        
        from alpha_automl import AutoMLTimeSeries
        automl = AutoMLTimeSeries(time_bound=1, date_column=date_column, target_column=target_column, txt_file = logs_path,  output_folder=original_logs)
        automl.fit(X_train, y_train)

    elif task_name.lower() == 'semisupervised':
        
        from alpha_automl import AutoMLSemiSupervisedClassifier
        automl = AutoMLSemiSupervisedClassifier(time_bound=1, start_mode='spawn', txt_file = logs_path,  output_folder=original_logs)
        automl.fit(X_train, y_train)
    

def generate_jobs():
    jobs = []

    for task in TASK:
        
            for dataset_name in dataset_names_for_task[task]:
            
                for llm in LLMs:
                    
                    tmp_output_directory = root_dir + '/RESULT_old/AUTO_SKLN/' + task + '/' + dataset_name + '/' + llm + '/'
                    
                    for sum_prompt in SUMMARIZATION_PROMPT:
                
                        output_directory = tmp_output_directory + sum_prompt + '/'
                        create_directory(output_directory)
                        summary_dir =  os.path.join(output_directory , 'summary_result.txt')
                        logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')

                        jobs.append((logs_path, summary_dir, sum_prompt, output_directory))

    return jobs

def iterate_loop():
    jobs = []
    for task in TASK:
        for dataset_name, sum_llm, sum_prompt in product(dataset_names_for_task[task], LLMs, SUMMARIZATION_PROMPT):
            output_directory = root_dir + '/RESULT_old/results_Hierachical_prompt_AUTOSKLN/' + task + '/' + dataset_name + '/' + sum_llm + '/' 
            # output_directory = root_dir + '/results_Hierarchical_Prompting/' + task + '/' + dataset_name + '/' + sum_llm + '/'
            create_directory(output_directory)
            summary_dir =  os.path.join(output_directory , 'summary_result.txt')
            # if(task == "CLASSIFICATION") | (task == "REGRESSION"):
            #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
            # else:
            #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
                
            # logs_path = "/home/nguenang/Master_thesis/experiment_setup/results/AUTOSKLN_LOGS/CLASSIFICATION/logs_cls.txt"
            # logs_path = "/home/nguenang/Master_thesis/experiment_setup/log_analysis_out.txt"

            logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
            # logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
            jobs.append((logs_path, summary_dir, sum_prompt, output_directory,sum_llm,task,dataset_name))

    return jobs


            
    


########################## main ########################
load_dotenv()
root_dir = os.getenv('ROOT_DIR')


if __name__ == "__main__":

    results = []

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
                # x_test = datasets_dict[dataset_name][2]
                # y_test = datasets_dict[dataset_name][3] 
                target_column = datasets_dict[dataset_name][4]


                print('-----------------START FITTING--------------')
                
                create_fit_classifier(task_name,x_train,y_train,target_column)
                
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
            print(f"DATASET : {dataset_names_for_task[task_name]}")
            datasets_dict = read_all_dataset(root_dir, task_name)
            tmp_output_directory = root_dir + '/results/' + task_name + '/' 
            print(f"DATASET : {dataset_names_for_task[task_name]}")

            for dataset_name in dataset_names_for_task[task_name]:
                print('\tdataset name: ', dataset_name)
                print(" ")
                
                # datasets_dict = read_all_dataset(root_dir, task_name)
                
                x_train = datasets_dict[dataset_name][0]
                y_train = datasets_dict[dataset_name][1]
                # x_test = datasets_dict[dataset_name][2]
                # y_test = datasets_dict[dataset_name][3] 
                target_column = datasets_dict[dataset_name][2]
                date_column = datasets_dict[dataset_name][3]
                
                output_directory = tmp_output_directory + dataset_name + '/'
                
                output_dir = tmp_output_directory + dataset_name + '/'

                print('-----------------START FITTING--------------')
                
                # create_fit_classifier(task_name,x_train,y_train,target_column, output_directory, date_column, output_dir)
                
                print('--------------------DONE--------------------')
                print(" ")
               
               
                
              
              
                logs_path = os.path.join(output_directory, 'full_log_MainProcess.txt')
                
                ### Filter the logs
                
                # if(task_name =="CLASSIFICATION") | (task_name =="REGRESSION"):
                print("------------FILTERING THE LOGS-----------------")               

                fil_tmp = os.path.join(output_directory, 'fil_tmp.txt')
                filter_path = os.path.join(output_directory, 'filter_logs.txt')
                filtering_logs(logs_path, fil_tmp)
                
                remove_consecutive_duplicates(fil_tmp, filter_path)
                    # print("FILTER LOGs:", filter_logs)
                    
                    # filter_path = os.path.join(output_directory, 'filter_logs.txt')
                    
                    # with open(filter_path, mode="w" , encoding="utf-8") as f:
                    #     f.write(filter_logs)
                    
               
               
               
                
    # elif sys.argv[1] == 'summarize':
        
    #     jobs = iterate_loop()
        
    #     with ThreadPoolExecutor(max_workers=0) as executor:
    #         for logs_path, file_dir, sum_prompt, output_directory, sum_llm, _, _ in jobs:
                
    #             print('\t\t\tprompt: ', sum_prompt)
    #             print( " ")
    #             print(f"LOGS: {logs_path}")
                
    #             # filter_logs = filter_automl_logs(logs_path)
    #             # # print("FILTER LOGs:", filter_logs)
                
    #             # filter_path = os.path.join(output_directory, 'filter_logs')
                
    #             # with open(filter_path, mode="w" , encoding="utf-8") as f:
    #             #     f.write(filter_logs)
                    
 
    #             args = (logs_path, sum_llm, file_dir,sum_prompt) 
    #             executor.submit(explain_process, *args)
  
    #     print( " " )
    
    
    elif sys.argv[1] == 'summarize':
        
        jobs = iterate_loop()
        
        # with ThreadPoolExecutor(max_workers=0) as executor:
        for logs_path, file_dir, sum_prompt, output_directory, sum_llm, _, _ in jobs:
            
            print('\t\t\tprompt: ', sum_prompt)
            print( " ")
            print(f"LOGS: {logs_path}")
            
            # filter_logs = filter_automl_logs(logs_path)
            # # print("FILTER LOGs:", filter_logs)
            
            # filter_path = os.path.join(output_directory, 'filter_logs')
            
            # with open(filter_path, mode="w" , encoding="utf-8") as f:
            #     f.write(filter_logs)
            
            print(f"LOG PATH: {logs_path}")

            explain_process(logs_path, sum_llm, file_dir,sum_prompt)


        print( " " )
 
    ### summarization using Hierachical prompt modelling
    
    elif sys.argv[1] == 'summarize_Hierachical_prompt':
        
        #with ThreadPoolExecutor(max_workers=2) as executor:
        for task in TASK:
            for dataset_name, sum_llm in product(dataset_names_for_task[task], LLMs):
                output_directory = root_dir + '/results_Hierarchical_Prompting/' + task + '/' + dataset_name + '/' + sum_llm + '/'
                create_directory(output_directory)
                summary_dir =  os.path.join(output_directory , 'fact.txt')
                # if(task == "CLASSIFICATION") | (task == "REGRESSION"):
                #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
                # else:
                #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
                
                # logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
                logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
                print( " ")
                print(f"LOGS: {logs_path}")
                
                fact_extraction(logs_path, sum_llm, summary_dir)
                
                
                
                
                # ## aggregate fact
                # fact_agg_dir =  os.path.join(output_directory , 'fact_agg_dir.json')
                
                # # fact_aggregation(summary_dir, sum_llm, fact_agg_dir)

                # args = (logs_path, sum_llm, summary_dir) 
                # executor.submit(phase_segmentation, *args)
        
                # # micro summary generation ##
                

                # micro_summary_dir = os.path.join(output_directory, 'micro_summary.json')
                # print( " " )
                
                # # import json

                # # with open(summary_dir, "r") as f:
                # #     phases = json.load(f)
                    
                # # micro_summarization(summary_dir, sum_llm, micro_summary_dir, phases) 
                
                
                
                
                
                
                
                
                
                
                
                # # # macro summary generation
                 
                global_summary_dir = os.path.join(output_directory, 'global_summary_T.txt') 
                macro_summarization(summary_dir, sum_llm, global_summary_dir, logs_path )  
            
                ### VERIFICATION STEP
                verification_dir = os.path.join(output_directory, 'verification.json') 
                verification(global_summary_dir, logs_path, sum_llm, verification_dir)
                
                ###REVISED SUMMARY
                
                revised_summary_dir = os.path.join(output_directory, 'global_summary_revised.txt')
                revised_summary(global_summary_dir, logs_path, verification_dir, sum_llm, revised_summary_dir)
                    
            
    
    elif sys.argv[1] == 'judge_H': 
    
        jobs = iterate_loop()
        
        # with ThreadPoolExecutor(max_workers=4) as executor:
        
        for task in TASK:
            for dataset_name, sum_llm in product(dataset_names_for_task[task], LLMs):
                output_directory = root_dir + '/results_Hierarchical_Prompting/' + task + '/' + dataset_name + '/' + sum_llm + '/'
                create_directory(output_directory)
                summary_dir =  os.path.join(output_directory , 'global_summary_revised.txt')
                # if(task == "CLASSIFICATION") | (task == "REGRESSION"):
                #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
                # else:
                #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
    
                logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
                print( " ")
                print(f"LOGS: {logs_path}")

                for llm_judge, judge_prompt in product(LLMs_judge, JUDGING_PROMPT): 
                    final_dir = output_directory + judge_prompt + '/'     
                    create_directory(final_dir)
                    judge_dir = os.path.join(final_dir, f'evaluation_'+llm_judge+'.txt')
                    # judge_dir = os.path.join(output_directory, f'evaluation_{llm_judge}_mismatch.txt')
                    
                    print(f"\t\t judging the summary of the logs: {logs_path}")
                    print(f"\t\t\t using the LLM:", llm_judge, " to judge")
                    print(f"\t\t Judging the summary located in the directory: {summary_dir}")
                    print("JUDGING PROMPT:", judge_prompt)  
                    print(f"judge file directory: {judge_dir}")   
                    
                    judging_explanation(logs_path,summary_dir,llm_judge,judge_dir,judge_prompt)
                    args = (logs_path,summary_dir,llm_judge,judge_dir,judge_prompt)
                    # executor.submit(judging_explanation, *args)
                    # gc.collect()                    
                
        print( " " )  

                
            
    elif sys.argv[1] == 'judge': 
        
        jobs = iterate_loop()
        
        # with ThreadPoolExecutor(max_workers=4) as executor:
                
        for logs_path, file_dir, sum_prompt, output_directory ,_ ,_, _ in jobs:
            for llm_judge, judge_prompt in product(LLMs_judge, JUDGING_PROMPT):
                dir_temp = output_directory + judge_prompt + '/'
                create_directory(dir_temp)      
                judge_dir = os.path.join(dir_temp, f'evaluation_'+llm_judge+'.txt')
                # judge_dir = os.path.join(output_directory, f'evaluation_{llm_judge}_mismatch.txt')
                
                print(f"\t\t judging the summary of the logs: {logs_path}")
                print(f"\t\t\t using the LLM:", llm_judge, " to judge")
                print(f"\t\t Judging the summary located in the directory: {file_dir}")
                print("JUDGING PROMPT:", judge_prompt)  
                print(f"judge file directory: {judge_dir}")  
                judging_explanation(logs_path,file_dir,llm_judge,judge_dir,judge_prompt) 
                    # args = (logs_path,file_dir,llm_judge,judge_dir,judge_prompt)
                    # executor.submit(judging_explanation, *args)
                    # gc.collect()                    
                        
        print( " " )  
            
            
     
     
     
     ### summarization using Hierachical prompt modelling MODIFIED METHODS
    
    elif sys.argv[1] == 'summarize_H_new':
        
        #with ThreadPoolExecutor(max_workers=2) as executor:
        for task in TASK:
            for dataset_name, sum_llm in product(dataset_names_for_task[task], LLMs):
                output_directory = root_dir + '/results_test_H/' + task + '/' + dataset_name + '/' + sum_llm + '/'
                create_directory(output_directory)
                summary_dir =  os.path.join(output_directory , 'fact.json')
                # if(task == "CLASSIFICATION") | (task == "REGRESSION"):
                #     logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'filter_logs.txt')
                # else:
                logs_path = os.path.join(root_dir, 'results', task, dataset_name, 'full_log_MainProcess.txt')
                
                print( " ")
                print(f"LOGS: {logs_path}")
                
                phase_segmentation(logs_path, sum_llm, summary_dir)
                
                import json
                with open(summary_dir, "r") as f:
                    phases = json.load(f)
                    
                
                phase_filter =  os.path.join(output_directory , 'phase_filter.txt')    
                text = regex_filtering(logs_path, phases)
                
                             
                with open(phase_filter, mode="w" , encoding="utf-8") as f:
                        f.write(text)
                    
                           
            
                
                
                
                
   
                
                
     
     
     
     
     
     
     
     
     
     
     
     
            

    elif sys.argv[1] == 'generate_csv_file':
        csv_df = [ ]
        doc = []
        jobs = iterate_loop()
        for logs_path, summary_dir, sum_prompt, output_directory, sum_llm, task, dataset_name in jobs:
            for  llm_judge, judge_prompt in product(LLMs_judge,JUDGING_PROMPT ):
                # print("\tjudging the TASK:", task)
                # print(f"\t\t judging the summary of the logs:  full_log_MainProcess.txt")
                # print(f"\t\t\t using the LLM:", llm_judge, " to judge")
                # print(f"\t\t Judging the summary located in the directory: {file_dir}")                   
                judge_dir = os.path.join(output_directory, f'evaluation_REVISED{llm_judge}.txt')           
                print(f"judge file directory: {judge_dir}")
                print("JUDGING PROMPT:", judge_prompt)        
                row = result_script(judge_dir)
                # print(f"row: {row}")
                # doc.append(row)
                # if len(doc) != 0:
                #     print(len(doc))
                #     print(f"list is: {doc}")
                result_dir = os.path.join(root_dir, f"result_category6_H_autoskln.csv")
                write_csv(task,dataset_name, sum_llm, sum_prompt, llm_judge, judge_prompt, row , result_dir)     
        
            
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
