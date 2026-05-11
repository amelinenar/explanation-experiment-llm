import os
import re
import pandas as pd
import numpy as np
from itertools import product
from dotenv import load_dotenv

from utils.constant import CLASSIFICATION_DATASET,TIMESERIES_DATASET
from utils.constant import REGRESSION_DATASET,SEMI_SUPERVISED_DATASET
from utils.constant import TASK, LLMs

from utils.constant import dataset_names_for_task
from utils.constant import SUMMARIZATION_FLAT_PROMPT,JUDGING_PROMPT, SUMMARIZATION_HIERARCHICAL_PROMPT,AUTOML,PROMPT_STRATEGY

load_dotenv()
root_dir = os.getenv('ROOT_DIR')


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory



def read_dataset(root_dir, task_name, dataset_name):
    dataset_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    
    if task_name.lower() == 'regression':
        root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
        train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
        # test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
        
        if dataset_name == '196_autoMpg':
            target_column = 'class'
        elif dataset_name == '26_radon_seed_dataset':
            target_column = 'log_radon'
        elif dataset_name == '534_cps_85_wages':
            target_column = 'WAGE'
        elif dataset_name == 'advertising_dataset':
            target_column = 'Sales'
            
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]
        # X_test = test_dataset.drop(columns=[target_column])
        # y_test = test_dataset[[target_column]] 
        
        dataset_dict[dataset_name] = (X_train, y_train, target_column, date_column)
        
    elif task_name.lower() == 'classification':
        root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
        train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
        # test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
        
        if dataset_name == '185_baseball_dataset':
            target_column = 'Hall_of_Fame'
        if dataset_name == '1491_one_hundred_plants':
            target_column = 'Class'
        if dataset_name == '1567_poker_hand' :
            target_column = 'Class'
        if dataset_name == '299_libras_move':
            target_column = 'class'
        
  
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]
        # X_test = test_dataset.drop(columns=[target_column])
        # y_test = test_dataset[[target_column]] 
        
        dataset_dict[dataset_name] =  (X_train, y_train, target_column, date_column)
        
    elif task_name.lower() == 'semisupervised':
        root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
        train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
        # test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
        
        
        if dataset_name == 'LL0_1053_jm1':
            target_column = 'defects'
        if dataset_name == 'SEMI_155_pokerhand_dataset':
            target_column = 'class'
        if dataset_name == 'SEMI_1217_click_dataset' :
            target_column = 'click'
        if dataset_name == 'SEMI_1459_artificial_characters_dataset':
            target_column = 'Class'
    
        
        y_train = train_dataset[[target_column]]
        X_train = train_dataset.drop(columns=[target_column])
        
        # X_test = test_dataset.drop(columns=[target_column])
        # y_test = test_dataset[[target_column]] 
        
        dataset_dict[dataset_name] = (X_train, y_train, target_column, date_column)
        
    
    elif task_name.lower() == 'time_series_forecast':
        root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
        train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
        # test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
        
        if dataset_name == '56_sunspots_monthly':
            target_column = 'sunspots'
            date_column = 'year-month'
        if dataset_name == 'LL1_crime_chicago' :
            target_column = 'location1'
            date_column = 'time'
        if dataset_name == 'stock_market':
            target_column = 'Close'
            date_column = 'Date'
        if dataset_name == '57_sunspot':
            target_column = 'sunspots'
            date_column = 'year'
            
    
            
        
        
        X_train = train_dataset
        y_train = train_dataset[[target_column]]
        # X_test = test_dataset
        # y_test = test_dataset[[target_column]] 
        
        dataset_dict[dataset_name] = (X_train, y_train, target_column,date_column)

    return dataset_dict

def read_all_dataset(root_dir, task_name):
    dataset_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []
    date_column = ''
    
    if task_name.lower() == 'regression':
        
        for dataset_name in REGRESSION_DATASET:
            root_dir_dataset = cur_root_dir + '/dataset/' + task_name 
            
            train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
            # test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')  
            
            if dataset_name == '196_autoMpg':
                target_column = 'class'
            elif dataset_name == '26_radon_seed_dataset':
                target_column = 'log_radon'
            elif dataset_name == '534_cps_85_wages':
                target_column = 'WAGE'
            elif dataset_name == 'advertising_dataset':
                target_column = 'Sales'
                
            # if dataset_name == '196_autoMpg':
                # target_column = 'class'
             
            print("Target column:", target_column)
            print("Available columns:", train_dataset.columns.tolist())
    
            X_train = train_dataset.drop(columns=[target_column])
            y_train = train_dataset[[target_column]]
            # X_test = test_dataset.drop(columns=[target_column])
            # y_test = test_dataset[[target_column]] 
            
            dataset_dict[dataset_name] = (X_train, y_train,target_column,date_column)
                
    
    elif task_name.lower() == 'classification':
        
        for dataset_name in CLASSIFICATION_DATASET:
            root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
            
            train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
            test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
            
            if dataset_name == '185_baseball_dataset':
                target_column = 'Hall_of_Fame'
            if dataset_name == '1491_one_hundred_plants':
                target_column = 'Class'
            if dataset_name == '1567_poker_hand' :
                target_column = 'Class'
            if dataset_name == '299_libras_move':
                target_column = 'class'
          
            X_train = train_dataset.drop(columns=[target_column])
            y_train = train_dataset[[target_column]]
            # X_test = test_dataset.drop(columns=[target_column])
            # y_test = test_dataset[[target_column]] 
            
            dataset_dict[dataset_name] = (X_train, y_train, target_column,date_column)
            
    
    elif task_name.lower() == 'semisupervised':
    
        for dataset_name in SEMI_SUPERVISED_DATASET:
            root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
            
            train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
            test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
            
            if dataset_name == 'LL0_1053_jm1':
                target_column = 'defects'
            if dataset_name == 'SEMI_155_pokerhand_dataset':
                target_column = 'class'
            if dataset_name == 'SEMI_1217_click_dataset' :
                target_column = 'click'
            if dataset_name == 'SEMI_1459_artificial_characters_dataset':
                target_column = 'Class'

            X_train = train_dataset.drop(columns=[target_column])
            y_train = train_dataset[[target_column]]
            # X_test = test_dataset.drop(columns=[target_column])
            # y_test = test_dataset[[target_column]] 
            
            dataset_dict[dataset_name] = (X_train, y_train, target_column, date_column)
        
            
    elif task_name.lower() == 'time_series_forecast':
    
        for dataset_name in TIMESERIES_DATASET:
            root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
            
            train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
            test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
                
            if dataset_name == '56_sunspots_monthly':
                target_column = 'sunspots'
                date_column = 'year-month'
            # if dataset_name == 'LL1_336_MS_Geolife':
            #     target_column = 'Transportation'
            #     date_column = 'date_time'
            if dataset_name == 'LL1_crime_chicago' :
                target_column = 'location1'
                date_column = 'time'
            if dataset_name == 'stock_market':
                target_column = 'Close'
                date_column = 'Date'
            if dataset_name == '57_sunspot':
                target_column = 'sunspots'
                date_column = 'year'
                
        
            X_train = train_dataset
            y_train = train_dataset[[target_column]]
            # X_test = test_dataset
            # y_test = test_dataset[[target_column]] 
            
            dataset_dict[dataset_name] = (X_train, y_train,target_column,date_column)
    
           
    return dataset_dict     


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
    



def iterate_loop(prompt_strategy,automl):
    jobs = []
    for task in TASK:
        for dataset_name, sum_llm, sum_prompt in product(dataset_names_for_task[task], LLMs, SUMMARIZATION_FLAT_PROMPT):                
            
            # for automl in AUTOML:
            
            if automl == "ALPHA-AUTOML":
                logs_path = os.path.join(root_dir, 'results', automl, task, dataset_name, 'filter_logs.txt')
            if automl == "AUTOSKLEARN":
                logs_path = os.path.join( root_dir,  "autosklearn_logs",  task, dataset_name,  "full_log_MainProcess.txt")
                    
            
            if ( task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN"):
                print( f"⚠️  Skipping unsupported combination: "f"{automl} + {task}" )
                continue 

                
            # for prompt_strategy in PROMPT_STRATEGY:
                                
            if prompt_strategy == "FLAT_PROMPTING":                 
                output_directory = root_dir + '/results/'+ automl + '/' + task + '/' + dataset_name + '/' + sum_llm + '/' + sum_prompt + '/'
            elif prompt_strategy == "HIERARCHICAL_PROMPTING":
                output_directory = root_dir + '/results_Hierarchical_Prompting/'+ automl + '/' + task + '/' + dataset_name + '/' + sum_llm + '/'
                    

            create_directory(output_directory)
            summary_dir =  os.path.join(output_directory , 'summary_result.txt')

            # if automl == "":
            # logs_path = os.path.join(root_dir, 'results', automl, task, dataset_name, 'filter_logs.txt')
            # if automl == "":
                
                    
            jobs.append((logs_path, summary_dir, sum_prompt, output_directory,sum_llm,task,dataset_name,automl))

    return jobs













def generate_results(root_dir, task_name, dataset_name, llm):
    res = pd.DataFrame(data=np.zeros((0,3), dtype=np.float16), index=[],
                        columns=['task', 'dataset', 'llm' ])
    
    df_metrics = pd.DataFrame([{
        'task': task_name,
        'dataset': dataset_name,
        'llm': llm
    }])
    
    df_metrics['task'] = task_name
    df_metrics['dataset'] = dataset_name
    df_metrics['llm'] = llm
    #df_metrics['metric'] = metric  
    #df_metrics['llm'] = llm
 
    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(os.path.join(root_dir, 'results.csv'), index=False, mode='a', header= not os.path.exists(os.path.join(root_dir, 'results.csv')))
    
    return res


    
def filter_automl_logs(log_text_path):
    
    with open(log_text_path, "r", encoding="utf-8") as f:
        log_text = f.read()
        
    # 1. Remove timestamps
    log_text = re.sub(r'^\d{4}-\d{2}-\d{2} [\d:,]+ - ', '', log_text, flags=re.MULTILINE)

    # 2. Remove MCTS simulation spam
    log_text = re.sub(r'.*MCTS SIMULATION \d+.*\n?', '', log_text)

    # 3. Remove MOVE ACTION lines
    log_text = re.sub(r'.*MOVE ACTION:.*\n?', '', log_text)

    # 4. Deduplicate identical lines
    lines = log_text.splitlines()
    seen = set()
    filtered = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            filtered.append(line)

    return "\n".join(filtered)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    
                
                
        
        

# file_name = cur_root_dir + '/task/' + task_name + '/' + dataset_name + '/'
# # If running it in Windows or CUDA environment, Alpha-AutoML should be used inside of "if __name__ == '__main__':"
# # Read the datasets
# train_dataset = pd.read_csv(join(dirname(__file__), 'datasets/196_autoMpg/train_data.csv'))
# test_dataset = pd.read_csv(join(dirname(__file__), 'datasets/196_autoMpg/test_data.csv'))

# target_column = 'class'
# X_train = train_dataset.drop(columns=[target_column])
# y_train = train_dataset[[target_column]]
# X_test = test_dataset.drop(columns=[target_column])
# y_test = test_dataset[[target_column]]