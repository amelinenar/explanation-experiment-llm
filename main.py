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


from prompt import explain_process
from dotenv import load_dotenv

def create_fit_classifier(task_name,X_train,y_train, X_test,y_test,target_column):
    if task_name == 'regression':
          
        from alpha_automl import AutoMLRegressor
        automl = AutoMLRegressor(time_bound=1)
        # Perform the search
        automl.fit(X_train, y_train)
        
    elif task_name == 'classification':
        
        from alpha_automl import AutoMLClassifier
        automl = AutoMLClassifier(time_bound=1, verbose=True)
        automl.fit(X_train, y_train)

    elif task_name == 'timeseries_forcasting':
        
        from alpha_automl import AutoMLTimeSeries
        automl = AutoMLTimeSeries(time_bound=1, date_column='Date', target_column=target_column)
        automl.fit(X_train, y_train)

    elif task_name == 'semi_supervised_classification':
        
        from alpha_automl import AutoMLSemiSupervisedClassifier
        automl = AutoMLSemiSupervisedClassifier(time_bound=1, start_mode='spawn')
        automl.fit(X_train, y_train)
 
 
 
########################## main ########################

#change this directory for your machine
load_dotenv()
root_dir = os.getenv('ROOT_DIR')


if __name__ == "__main__":
    
    path_doc = os.path.join(root_dir, 'results.csv')
    if os.path.exists(path_doc):
        os.remove(path_doc)
            

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
                
                for llm in LLMs:
                    print('\t\tllm: ', llm)
                
                    output_directory = tmp_output_directory + dataset_name + '/' + llm + '/'
                    create_directory(output_directory)
                    file_dir =  os.path.join(output_directory , 'summary_result.txt')
         
                    explain_process('full_log_MainProcess.txt', llm, file_dir)
                    print(" ")
                    
                    generate_results(root_dir, task_name, dataset_name, llm)
                
                
                    print(f"DONE SUMMARIZATION: output stored in {file_dir}")



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
    
            explain_process('full_log_MainProcess.txt', llm, file_dir)
            
            generate_results(root_dir, task_name, dataset_name, llm)
            
            print(" ")
            print('DONE SUMMARIZATION')
