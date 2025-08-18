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

def create_fit_classifier(task_name,X_train,y_train, X_test,y_test):
    if task_name == 'regression':
        
        from alpha_automl import AutoMLRegressor
        # Add settings
        automl = AutoMLRegressor(time_bound=1)

        # Perform the search
        automl.fit(X_train, y_train)
        
    
    else: 
        #task_name == 'classifiction':
        from alpha_automl import AutoMLClassifier
        
        # Add settings
        automl = AutoMLClassifier(time_bound=1, verbose=True)

        # Perform the search
        automl.fit(X_train, y_train)

    
########################## main ########################

#change this directory for your machine
load_dotenv()
root_dir = os.getenv('ROOT_DIR')

if sys.argv[1] == 'run_all':
    for task_name in TASK:
        print('task_name', task_name)
        
        # for llm in LLMs:
        #     print('\llm', llm)
    
        datasets_dict = read_all_dataset(root_dir, task_name)
        
        

        tmp_output_directory = root_dir + '/results/' + task_name + '/' 

        for dataset_name in dataset_names_for_task[task_name]:
            print('\tdataset_name: ', dataset_name)
            
            x_train = datasets_dict[dataset_name][0]
            y_train = datasets_dict[dataset_name][1]
            x_test = datasets_dict[dataset_name][2]
            y_test = datasets_dict[dataset_name][3] 


            
            datasets_dict = read_all_dataset(root_dir, task_name)

            create_fit_classifier(task_name,x_train,y_train,x_test,y_test)
            
            for llm in LLMs:
                print('\t\tllm', llm)
            
                output_directory = tmp_output_directory + dataset_name + '/' + llm + '/'

                create_directory(output_directory)
                
                explain_process('full_log.txt', llm)
                
                
                generate_results('result.csv', root_dir, task_name, dataset_name, llm)
            
            
                print('t\t\tDONE')

                # the creation of this directory means
                create_directory(output_directory + '/DONE')


elif sys.argv[1] == '':
    pass

elif sys.argv[1] == 'generate_results':
    
    res = generate_results('results.csv', root_dir)
    print(res.to_string())
    


else:
    # this is the code to launch an experiment on a dataset
    task_name = sys.argv[1]
    dataset_name = sys.argv[2]
    llm = sys.argv[3]
    
    output_directory  = root_dir + '/results/' + task_name + '/' + dataset_name+ '/' + llm + '/'
    
    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', task_name, dataset_name, llm)
    
    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, task_name, dataset_name)
        
        x_train = datasets_dict[dataset_name][0]
        y_train = datasets_dict[dataset_name][1]
        x_test = datasets_dict[dataset_name][2]
        y_test = datasets_dict[dataset_name][3] 

        create_fit_classifier(task_name,x_train,y_train,x_test,y_test)
        
        generate_results('result.csv', root_dir, task_name, dataset_name, llm)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')
    
    