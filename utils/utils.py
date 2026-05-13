import os
import re
import pandas as pd
import numpy as np
from itertools import product
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import seaborn as sns

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
    



# ============================================================
# CONFIGURATION
# ============================================================

METRICS_OF_INTEREST = [
    "Accuracy",
    "Completeness",
    "Clarity",
    "Relevance"
]



def generate_graph(csv_file):
    
    output_dir = os.path.join(root_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(csv_file)

    # Convert score to percentage
    df["score_per"] = ((df["score"] - 1) / 3) * 100

    # Remove unwanted datasets/tasks
    df_common = df[
        (df["task"] != "REGRESSION") &
        (df["dataset"] != "185_baseball_dataset")
    ]

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    tasks = df_common["task"].unique()
    datasets = df_common["dataset"].unique()
    llms = df_common["llm_summarizer"].unique()
    prompts = df_common["summarization_prompt"].unique()
    metrics = df_common["metric"].unique()

    # ========================================================
    # RQ1 — OVERALL PROMPT PERFORMANCE
    # ========================================================

    print("\n================================================")
    print("RQ1 — Overall Prompt Performance")
    print("================================================")

    metric_scores = (
        df_common
        .groupby(["summarization_prompt", "metric"])["score_per"]
        .mean()
        .unstack()
    )

    print(metric_scores)

#     for metric in metrics:

#         subset = df_common[df_common["metric"] == metric]

#         if subset.empty:
#             continue

#         data = []
#         labels = []

#         for prompt in prompts:

#             prompt_scores = subset[
#                 subset["summarization_prompt"] == prompt
#             ]["score_per"]

#             if len(prompt_scores) > 0:
#                 data.append(prompt_scores)
#                 labels.append(prompt)

#         if not data:
#             continue

#         plt.figure(figsize=(6, 5))
#         plt.boxplot(data)
#         plt.xticks( range(1, len(labels) + 1), labels,rotation=45)
#         plt.ylabel("Score (%)")

#         plt.title(f"Score Distribution per Prompt\nMetric: {metric}" )
#         plt.tight_layout()
#         plt.savefig(
#     os.path.join(output_dir, f"overall_prompt_performance_{metric}.pdf"),
#     bbox_inches="tight",
#     pad_inches=0
# )



    # ========================================================
    # PROMPT × TASK INTERACTION
    # ========================================================

    print("\n================================================")
    print("Prompt × Task Interaction")
    print("================================================")

    filtered = df_common[
        df_common["metric"].isin(METRICS_OF_INTEREST)
    ]

    interaction_scores = {}

    for metric in METRICS_OF_INTEREST:

        interaction_scores[metric] = (
            filtered[filtered["metric"] == metric]
            .groupby(
                ["summarization_prompt", "task"]
            )["score_per"]
            .mean()
            .unstack()
        )

    for metric, table in interaction_scores.items():

        prompts = table.index.tolist()
        tasks = table.columns.tolist()

        angles = np.linspace(
            0,
            2 * np.pi,
            len(prompts),
            endpoint=False
        ).tolist()

        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.axes(polar=True)

        for task in tasks:

            values = table[task].tolist()
            values += values[:1]

            ax.plot(angles,  values,  marker="o",  linewidth=2,label=task)  
            ax.fill(angles, values, alpha=0.15)

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            prompts
        )

        ax.set_title(
            f"Summary Quality: Prompt × Task ({metric})"
        )

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1)
        )
        
        plt.savefig(
    os.path.join(output_dir, f"prompt_performance_task_{metric}.pdf"),
    bbox_inches="tight",
    pad_inches=0
)

     

    # ========================================================
    # PROMPT PERFORMANCE PER DATASET
    # ========================================================

    print("\n================================================")
    print("Prompt Performance per Dataset")
    print("================================================")

    dataset_tables = {}

    for dataset in datasets:

        dataset_df = df_common[
            (df_common["dataset"] == dataset) &
            (df_common["metric"].isin(METRICS_OF_INTEREST))
        ]

        table = (
            dataset_df
            .groupby(
                ["metric", "summarization_prompt"]
            )["score_per"]
            .mean()
            .unstack()
        )

        dataset_tables[dataset] = table

    # --------------------------------------------------------
    # RADAR GRID
    # --------------------------------------------------------

    n_datasets = len(dataset_tables)

    n_cols = 4
    n_rows = int(np.ceil(n_datasets / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        subplot_kw=dict(polar=True),
        figsize=(22, 18)
    )

    axes = axes.flatten()

    legend_handles = None

    for i, (ax, (dataset, table)) in enumerate(
        zip(axes, dataset_tables.items())
    ):

        prompts = table.columns.tolist()
        metrics = table.index.tolist()

        angles = np.linspace(
            0,
            2 * np.pi,
            len(prompts),
            endpoint=False
        ).tolist()

        angles += angles[:1]

        for metric in metrics:

            values = table.loc[metric].tolist()
            values += values[:1]

            ax.plot(
                angles,
                values,
                linewidth=2,
                label=metric
            )

            ax.fill(angles, values, alpha=0.08)

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            prompts
        )

        ax.set_title(dataset, fontsize=10)

        ax.set_ylim(0, 100)

        if i == 0:
            legend_handles = ax.get_legend_handles_labels()

    # Hide unused axes
    for i in range(len(dataset_tables), len(axes)):
        fig.delaxes(axes[i])

    # --------------------------------------------------------
    # GLOBAL LEGEND
    # --------------------------------------------------------

    handles, labels = legend_handles

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.tight_layout()

    plt.subplots_adjust(
        bottom=0.08,
        wspace=0.15,
        hspace=0.25
    )

    plt.savefig(
    os.path.join(output_dir, "prompt_per_dataset_without_regression.pdf"),
    bbox_inches="tight",
    pad_inches=0
)
    
    # ========================================================
    # PROMPT × LLM INTERACTION
    # ========================================================

    print("\n================================================")
    print("Prompt × LLM Interaction")
    print("================================================")

    filtered = df_common[
        df_common["metric"].isin(METRICS_OF_INTEREST)
    ]

    interaction_scores = {}

    for metric in METRICS_OF_INTEREST:

        interaction_scores[metric] = (
            filtered[filtered["metric"] == metric]
            .groupby(
                ["summarization_prompt", "llm_summarizer"]
            )["score_per"]
            .mean()
            .unstack()
        )

    print(interaction_scores["Accuracy"])

    for metric, table in interaction_scores.items():

        prompts = table.index.tolist()
        llms = table.columns.tolist()

        angles = np.linspace(
            0,
            2 * np.pi,
            len(prompts),
            endpoint=False
        ).tolist()

        angles += angles[:1]

        plt.figure(figsize=(8, 8))

        ax = plt.axes(polar=True)

        for llm in llms:

            values = table[llm].tolist()
            values += values[:1]

            ax.plot(
                angles,
                values,
                marker="o",
                linewidth=2,
                label=llm
            )

            ax.fill(angles, values, alpha=0.15)

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            prompts
        )

        ax.set_title(
            f"Interaction Effect: Prompt × LLM ({metric})"
        )

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1)
        )
        
        
        plt.savefig(
    os.path.join(output_dir, f"prompt_llm_interaction_{metric}.pdf"),
    bbox_inches="tight",
    pad_inches=0
)

    
    
    
    #------------------------------------
    #  Prompt performance  accross AutoML 
    #-------------------------------------

    task_of_interest = df["task"].unique()

    filtered = df[df["metric"].isin(METRICS_OF_INTEREST)]

    interaction_scores = {}

    for metric in METRICS_OF_INTEREST:
        interaction_scores[metric] = (
            filtered[filtered["metric"] == metric]
            .groupby(["summarization_prompt", "automl"])["score_per"]
            .mean()
            .unstack()
        )

    # # Example
    # print(interaction_scores["Accuracy"])


    for metric, table in interaction_scores.items():

        prompts = table.index.tolist()
        tasks = table.columns.tolist()

        angles = np.linspace(0, 2 * np.pi, len(prompts), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.axes(polar=True)

        for task in tasks:
            values = table[task].tolist()
            values += values[:1]

            ax.plot(angles, values, marker='o', linewidth=2, label=task)
            ax.fill(angles, values, alpha=0.15)
            


        ax.set_thetagrids(np.degrees(angles[:-1]), prompts)
        # ax.set_title(f"Summary quality wrt : Prompt × AutoMl ({metric})")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        


        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)

        # fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

        plt.savefig(
             os.path.join(output_dir, f"prompt_ROBUSTNESS_{metric}.pdf"),
            bbox_inches='tight',
            pad_inches=0
        )
        
    
    
    #------------------------------#
    #-----overall distribution of the evaluation per llm-----#
    #------------------------------#


    print("\n===============================================")
    print("overall distribution of the evaluation per llm")
    print("================================================")


    for metric in metrics:

        subset = df[
            (df["metric"] == metric)
            # & (df_2task_3llm["task"] == task)
            # & (df_2task_3llm["dataset"] == dataset)
            # & (df_2task_3llm["summarization_prompt"] == prompt)
        ]

        if subset.empty:
            continue

        
        # summary = (
        #     subset
        #     .groupby("judging_llm")["score_per"]
        #     .agg(["count", "mean", "median", "std", "min", "max"])
        #     .round(2)
        # )

        # print(f"\nMetric: {metric}")
        # print(summary)
        
        
        summary_all = (
        df
        .groupby(["metric", "judging_llm"])["score_per"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    print(summary_all)
        
        
    print(f"The graphs are stored in the folder {output_dir}")
        
       





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