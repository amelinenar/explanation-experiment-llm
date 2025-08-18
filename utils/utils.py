import os
import pandas as pd
import numpy as np

from utils.constant import CLASSIFICATION_DATASET
from utils.constant import REGRESSION_DATASET
from utils.constant import TASK
from utils.constant import LLMs




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
    
    if task_name == 'regression':
        root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
        train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
        test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
        
        target_column = 'class'
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]
        X_test = test_dataset.drop(columns=[target_column])
        y_test = test_dataset[[target_column]] 
        
        dataset_dict[dataset_name] = (X_train, y_train, X_test, y_test)
        
    elif task_name == 'classification':
        root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
        train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
        test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
        
        target_column = 'class'
        X_train = train_dataset.drop(columns=[target_column])
        y_train = train_dataset[[target_column]]
        X_test = test_dataset.drop(columns=[target_column])
        y_test = test_dataset[[target_column]] 
        
        dataset_dict[dataset_name] = (X_train, y_train, X_test, y_test)
        
    return dataset_dict

def read_all_dataset(root_dir, task_name):
    dataset_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []
    
    if task_name == 'regression':
        
        for dataset_name in REGRESSION_DATASET:
            root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
            
            train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
            test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
            
            target_column = 'class'
            X_train = train_dataset.drop(columns=[target_column])
            y_train = train_dataset[[target_column]]
            X_test = test_dataset.drop(columns=[target_column])
            y_test = test_dataset[[target_column]] 
            
            dataset_dict[dataset_name] = (X_train, y_train, X_test, y_test)
                
    
    elif task_name == 'classification':
        
        for dataset_name in CLASSIFICATION_DATASET:
            root_dir_dataset = cur_root_dir + '/dataset/' + task_name + '/' 
            
            train_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'train_data.csv')  
            test_dataset = pd.read_csv(root_dir_dataset + '/' + dataset_name + '/' + 'test_data.csv')   
            
            target_column = 'class'
            X_train = train_dataset.drop(columns=[target_column])
            y_train = train_dataset[[target_column]]
            X_test = test_dataset.drop(columns=[target_column])
            y_test = test_dataset[[target_column]] 
            
            dataset_dict[dataset_name] = (X_train, y_train, X_test, y_test)
           
    return dataset_dict     
          
        
def calculate_metrics():
    pass

def generate_results(file, root_dir, task_name, dataset_name, llm):
    
    res = pd.DataFrame(data=np.zeros((0,3), dtype=np.float16), index=[],
                        columns=['task', 'dataset', 'llm' ])
    # df_metrics = pd.DataFrame()
    
    df_metrics = pd.DataFrame([{
        'task': task_name,
        'dataset': dataset_name,
        'llm': llm
    }])
    
    # df_metrics = pd.read_csv(output_dir)
    # print(output_dir)
    df_metrics['task'] = task_name
    df_metrics['dataset'] = dataset_name
    df_metrics['llm'] = llm
    #df_metrics['metric'] = metric  
    #df_metrics['llm'] = llm
    res = pd.concat((res, df_metrics), axis=0, sort=False)
    
    res.to_csv(root_dir + '/' + 'results.csv', index=False, mode='a', header=not os.path.exists(root_dir + '/' + 'results.csv'))
    
    # res = generate_results('results.csv', root_dir)
    
    print(res.to_string())
    
    return res


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                    
                
                
            
        
        

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