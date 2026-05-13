import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_graph(csv_file):
    
    df = pd.read_csv("/home/nguenang/Master_thesis/experiment_setup/result.csv")
    df["score_per"] = (df["score"]-1)/3 * 100

    df_common = df[(df["task"] != "REGRESSION") & (df["dataset"] != "185_baseball_dataset")]


    """RQ1: How does the choice of the prompt variant affected the quality of the summary 
        Did the prompt variant affected the quality of the summary ?? """
        
    # Mean score per metric
    # ----------------------------

    # Compute mean score for each prompt × metric
    metric_scores = (
        df_common
    
        .groupby(["summarization_prompt", "metric"])["score_per"]
        .mean()
        .unstack()
    )

    print(metric_scores)
    
    tasks = df_common['task'].unique()
    datasets = df_common['dataset'].unique()
    llms = df_common['llm_summarizer'].unique()
    prompts = df_common['summarization_prompt'].unique()
    metrics = df_common['metric'].unique()
    
    
    #------------------------------
#  the overall performance of prompt 
#-------------------------------------


    for metric in metrics:

        subset = df_common[
            (df_common["metric"] == metric)
            # & (df_2task_3llm["task"] == task)
            # & (df_2task_3llm["dataset"] == dataset)
            # & (df_2task_3llm["summarization_prompt"] == prompt)
        ]

        if subset.empty:
            continue
        
        data = []
        labels = []

        for prompt in prompts:
            prompts_scores = subset[subset["summarization_prompt"] == prompt]["score_per"]

            if len(prompts_scores) > 0:
                data.append(prompts_scores)
                labels.append(prompt)

        if not data:
            continue

        plt.figure(figsize=(6,5))
        plt.boxplot(data)
        plt.xticks(range(1, len(labels)+1), labels, rotation=45)
        plt.ylabel("Score %")
        plt.title(f"Score Distribution per Prompt\nMetric: {metric}")
        plt.tight_layout()
        plt.savefig(
        f"Overall Prompt Performance_{metric}.pdf",
        bbox_inches='tight',
        pad_inches=0
    )
        plt.show()
        
        
    
        
        
    #-------------------------------------------#
    #  How the prompt performance varied by task#
    #-------------------------------------------#

    metrics_of_interest = [
        "Accuracy",
        "Completeness",
        "Clarity",
        "Relevance"
    ]

    task_of_interest = df_common["task"].unique()

    filtered = df_common[df_common["metric"].isin(metrics_of_interest)]

    interaction_scores = {}

    for metric in metrics_of_interest:
        interaction_scores[metric] = (
            filtered[filtered["metric"] == metric]
            .groupby(["summarization_prompt", "task"])["score_per"]
            .mean()
            .unstack()
        )



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
        ax.set_title(f"Summary quality wrt : Prompt × Task ({metric})")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
        plt.savefig(
    f"prompt performance per task {metric}.pdf",
    bbox_inches='tight',
    pad_inches=0
)
        plt.show() 
        
        

    metrics_of_interest = [
        "Accuracy",
        "Completeness",
        "Clarity",
        "Relevance"
    ]

    datasets_of_interest = df_common["dataset"].unique()

    dataset_tables = {}

    # ---------------------------------
    # Build tables (metric x prompt)
    # ---------------------------------
    for dataset in datasets_of_interest:
        
        dataset_df = df_common[
            (df_common["dataset"] == dataset) &
            (df_common["metric"].isin(metrics_of_interest))
        ]
        print(dataset_df)
        table = (
            dataset_df
            .groupby(["metric", "summarization_prompt"])["score_per"]
            .mean()
            .unstack()
        )
        
        dataset_tables[dataset] = table

    print(dataset_tables[dataset])



    # ---------------------------------
    # Radar Grid (4 per row)
    # ---------------------------------

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

    for i, (ax, (dataset, table)) in enumerate(zip(axes, dataset_tables.items())):

        prompts = table.columns.tolist()
        metrics = table.index.tolist()

        angles = np.linspace(0, 2 * np.pi, len(prompts), endpoint=False).tolist()
        angles += angles[:1]

        for metric in metrics:
            values = table.loc[metric].tolist()
            values += values[:1]

            ax.plot(angles, values, linewidth=2, label=metric)
            ax.fill(angles, values, alpha=0.08)

        ax.set_thetagrids(np.degrees(angles[:-1]), prompts)
        ax.set_title(dataset, fontsize=10)
        ax.set_ylim(0, 100)

        # Collect legend handles only once (after full plotting)
        if i == 0:
            legend_handles = ax.get_legend_handles_labels()





    # Hide unused axes
    for i in range(len(dataset_tables), len(axes)):
        fig.delaxes(axes[i])


    # ---------------------------------
    # Global Legend (only once)
    # ---------------------------------

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
    plt.subplots_adjust(bottom=0.08)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    plt.savefig(
        "prompt_per_dataset_without_regression.pdf",
        bbox_inches='tight',
        pad_inches=0
    )
    plt.show()
    
    
    
    #-----------------------------------------------#
    #------- Summary quality wrt prompt x llm------#
    #----------------------------------------------#
    metrics_of_interest = [
        "Accuracy",
        "Completeness",
        "Clarity",
        "Relevance"
    ]

    filtered = df_common[df_common["metric"].isin(metrics_of_interest)]

    interaction_scores = {}

    for metric in metrics_of_interest:
        interaction_scores[metric] = (
            filtered[filtered["metric"] == metric]
            .groupby(["summarization_prompt", "llm_summarizer"])["score"]
            .mean()
            .unstack()
        )

    # Example
    print(interaction_scores["Accuracy"])

    metrics_of_interest = [
        "Accuracy",
        "Completeness",
        "Clarity",
        "Relevance"
    ]

    filtered = df_common[df_common["metric"].isin(metrics_of_interest)]

    interaction_scores = {}

    for metric in metrics_of_interest:
        interaction_scores[metric] = (
            filtered[filtered["metric"] == metric]
            .groupby(["summarization_prompt", "llm_summarizer"])["score"]
            .mean()
            .unstack()
        )

    # Example
    print(interaction_scores["Accuracy"])

    import numpy as np
    import matplotlib.pyplot as plt

    for metric, table in interaction_scores.items():

        prompts = table.index.tolist()
        llms = table.columns.tolist()

        angles = np.linspace(0, 2 * np.pi, len(prompts), endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.axes(polar=True)

        for llm in llms:
            values = table[llm].tolist()
            values += values[:1]

            ax.plot(angles, values, marker='o', linewidth=2, label=llm)
            ax.fill(angles, values, alpha=0.15)

        ax.set_thetagrids(np.degrees(angles[:-1]), prompts)
        ax.set_title(f"Interaction Effect: Prompt × LLM ({metric})")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

        plt.show()



