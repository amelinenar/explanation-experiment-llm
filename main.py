import pstats
import sys
import os
import time
import pandas as pd
import numpy as np

from itertools import product
from utils.utils import create_directory, create_fit_classifier,  read_dataset, read_all_dataset, generate_results, filter_automl_logs,iterate_loop,generate_graph


from utils.constant import TASK
from utils.constant import LLMs, LLMs_judge
from utils.constant import dataset_names_for_task
from utils.constant import SUMMARIZATION_FLAT_PROMPT,JUDGING_PROMPT, SUMMARIZATION_HIERARCHICAL_PROMPT,AUTOML,PROMPT_STRATEGY


from prompt import explain_process,judging_explanation
from dotenv import load_dotenv
from result_script import write_csv, result_script
from pipeline.Hierarchical_pipeline import phase_segmentation, micro_summarization,extract_pure_json,macro_summarization,verification,revised_summary,regex_filtering,fact_aggregation
from pipeline.Hierarchical_pipeline import fact_extraction
from filtering_logs import filtering_logs, remove_consecutive_duplicates

from concurrent.futures import ThreadPoolExecutor
import gc



########################## main ########################
load_dotenv()
root_dir = os.getenv('ROOT_DIR')


if __name__ == "__main__":

    results = []

    if sys.argv[1] == 'run_all_FlatPrompting':
        
        start_time = time.time()

        print("\n" + "=" * 80)
        print(" STARTING MODEL TRAINING PIPELINE")
        print("=" * 80)

        for task_name in TASK:

            # Skip unsupported task
            if task_name == "TIME_SERIES_FORECAST":
                continue

            print(f"\n TASK: {task_name}")
            print("-" * 80)

            dataset_list = dataset_names_for_task[task_name]
            print(f" Datasets: {dataset_list}")

            # Load all datasets for current task
            datasets_dict = read_all_dataset(root_dir, task_name)

            base_output_dir = os.path.join(root_dir,"results", "ALPHA-AUTOML", task_name )

            for dataset_name in dataset_list:

                print("\n" + "─" * 80)
                print(f" DATASET: {dataset_name}")
                print("─" * 80)

                # Extract dataset components
                x_train = datasets_dict[dataset_name][0]
                y_train = datasets_dict[dataset_name][1]
                target_column = datasets_dict[dataset_name][2]
                date_column = datasets_dict[dataset_name][3]

                # Output paths
                output_directory = os.path.join(base_output_dir, dataset_name)
                output_dir = output_directory

                # Optional: create output directory
                # create_directory(output_directory)

                print("⚙️  Training model...")
                fit_start = time.time()

                create_fit_classifier(task_name, x_train,y_train,target_column,output_directory,date_column, output_dir)

                fit_end = time.time()
                
               
                print(f"Training completed in {fit_end - fit_start:.2f} seconds")

                # ------------------------------------------------------------------
                # Log Processing
                # ------------------------------------------------------------------
                print("\n Filtering logs...")

                logs_path = os.path.join(output_directory,"full_log_MainProcess.txt")

                temp_filtered_log = os.path.join(output_directory, "fil_tmp.txt")
                final_filtered_log = os.path.join(output_directory, "filter_logs.txt")

                filtering_logs(logs_path, temp_filtered_log)
                remove_consecutive_duplicates(temp_filtered_log, final_filtered_log)

                print("Logs filtered successfully")

        # --------------------------------------------------------------------------
        # Total Runtime Summary
        # --------------------------------------------------------------------------
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" PIPELINE COMPLETED")
        print("=" * 80)

        print(f"  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)
            
                        
        
        start_time = time.time()

        print("\n" + "=" * 80)
        print(" STARTING FLAT SUMMARIZATION PIPELINE")
        print("=" * 80)

        # ==========================================================================
        # AUTOSKLEARN SUMMARIZATION
        # ==========================================================================
        print("\n SUMMARIZING AUTOSKLEARN RESULTS")
        print("-" * 80)

        for task in ["CLASSIFICATION","REGRESSION"]:

            print(f"\n TASK: {task}")

            for dataset_name, sum_llm, sum_prompt in product(
                dataset_names_for_task[task],
                LLMs,
                SUMMARIZATION_FLAT_PROMPT
            ):

                print("\n" + "─" * 80)
                print(f" DATASET : {dataset_name}")
                print(f" LLM     : {sum_llm}")
                print(f" PROMPT  : {sum_prompt}")
                print("─" * 80)

                # Output directory
                output_directory = os.path.join( root_dir, "results", "AUTOSKLEARN", task, dataset_name, sum_llm, sum_prompt)

                create_directory(output_directory)

                # Summary output file
                summary_dir = os.path.join( output_directory, "summary_result.txt" )

                # Logs path
                logs_path = os.path.join(root_dir,"autosklearn_logs",task,dataset_name, "full_log_MainProcess.txt" )

                print(f" Logs Path   : {logs_path}")
                print(f" Output File : {summary_dir}")

                # ------------------------------------------------------------------
                # Generate Summary
                # ------------------------------------------------------------------
                print("\n⚙️  Generating summary...")

                summary_start = time.time()

                explain_process( logs_path,sum_llm,summary_dir,sum_prompt )

                summary_end = time.time()

                print(
                    f"Summary generated in "
                    f"{summary_end - summary_start:.2f} seconds"
                )

        # ==========================================================================
        # ALPHA-AUTOML SUMMARIZATION
        # ==========================================================================
        print("\n" + "=" * 80)
        print(" SUMMARIZING ALPHA-AUTOML RESULTS")
        print("=" * 80)

        jobs = iterate_loop(prompt_strategy="FLAT_PROMPTING", automl="ALPHA-AUTOML")

        for (logs_path,file_dir,sum_prompt,output_directory,sum_llm, _, _,_) in jobs:

            print("\n" + "─" * 80)
            print(f" LLM     : {sum_llm}")
            print(f" PROMPT  : {sum_prompt}")
            print("─" * 80)

            print(f" Logs Path   : {logs_path}")
            print(f" Output File : {file_dir}")

            # ----------------------------------------------------------------------
            # Generate Summary
            # ----------------------------------------------------------------------
            print("\n⚙️  Generating summary...")

            summary_start = time.time()
            explain_process(logs_path,sum_llm,file_dir, sum_prompt)
            summary_end = time.time()
            print(
                f" Summary generated in "
                f"{summary_end - summary_start:.2f} seconds"
            )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("FLAT SUMMARIZATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)       
        
        
        
        start_time = time.time()

        print("\n" + "=" * 80)
        print("⚖️  STARTING FLAT SUMMARY EVALUATION")
        print("=" * 80)

        # ==========================================================================
        # MAIN LOOP
        # ==========================================================================
        for task in TASK:

            print(f"\n TASK: {task}")
            print("-" * 80)

            for dataset_name, sum_llm, sum_prompt in product(dataset_names_for_task[task],LLMs, SUMMARIZATION_FLAT_PROMPT):

                for automl in AUTOML:

                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # --------------------------------------------------------------
                    if (task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN" ):
                        print( f"⚠️  Skipping unsupported combination: " f"{automl} + {task}" )
                        continue

                    print("\n" + "─" * 80)
                    print(f" AutoML   : {automl}")
                    print(f" Dataset  : {dataset_name}")
                    print(f" Summary LLM : {sum_llm}")
                    print("─" * 80)

                    # --------------------------------------------------------------
                    # Output Directory
                    # --------------------------------------------------------------
                    output_directory = os.path.join(root_dir,"results", automl, task,dataset_name,sum_llm,sum_prompt )

                    create_directory(output_directory)

                    # --------------------------------------------------------------
                    # Summary File
                    # --------------------------------------------------------------
                    summary_file = os.path.join( output_directory,"summary_result.txt")

                    # --------------------------------------------------------------
                    # Logs Path
                    # --------------------------------------------------------------
                    if automl == "ALPHA-AUTOML":

                        logs_path = os.path.join(  root_dir, "results",automl, task,dataset_name, "filter_logs.txt"  )

                    elif automl == "AUTOSKLEARN":

                        logs_path = os.path.join( root_dir, "autosklearn_logs", task,dataset_name,  "full_log_MainProcess.txt"   )

                    print(f" Logs Path    : {logs_path}")
                    print(f" Summary File : {summary_file}")

                    # ==============================================================
                    # JUDGING LOOP
                    # ==============================================================
                    for llm_judge, judge_prompt in product(LLMs_judge,JUDGING_PROMPT):

                        print("\n" + "·" * 80)
                        print(f"⚖️  Judge LLM : {llm_judge}")
                        print(f" Prompt     : {judge_prompt}")
                        print("·" * 80)

                        # ----------------------------------------------------------
                        # Judge Output Directory
                        # ----------------------------------------------------------
                        final_dir = os.path.join( output_directory, judge_prompt  )
                        create_directory(final_dir)
                        judge_file = os.path.join(  final_dir,f"evaluation_{llm_judge}.txt"  )
                        print(f" Evaluation Output : {judge_file}")

                        # ----------------------------------------------------------
                        # Run Evaluation
                        # ----------------------------------------------------------
                        print("\n⚙️  Evaluating summary quality...")

                        eval_start = time.time()

                        judging_explanation(   logs_path, summary_file, llm_judge,judge_file,  judge_prompt)

                        eval_end = time.time()

                        print(
                            f" Evaluation completed in "
                            f"{eval_end - eval_start:.2f} seconds"
                        )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" FLAT SUMMARY EVALUATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)  
        
     
    if sys.argv[1] == 'run_all_HierarchicalPrompting':
        
        start_time = time.time()

        print("\n" + "=" * 80)
        print(" STARTING HIERARCHICAL SUMMARIZATION PIPELINE")
        print("=" * 80)

        # ==========================================================================
        # MAIN LOOP
        # ==========================================================================
        for task in TASK:

            print(f"\n TASK: {task}")
            print("-" * 80)

            for dataset_name, sum_llm in product(dataset_names_for_task[task], LLMs ):

                for automl in AUTOML:

                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # --------------------------------------------------------------
                    if ( task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN"):
                        print( f"⚠️  Skipping unsupported combination: "f"{automl} + {task}" )
                        continue

                    print("\n" + "─" * 80)
                    print(f" AutoML   : {automl}")
                    print(f" Dataset  : {dataset_name}")
                    print(f" LLM      : {sum_llm}")
                    print("─" * 80)

                    # --------------------------------------------------------------
                    # Output Directory
                    # --------------------------------------------------------------
                    output_directory = os.path.join(root_dir,"results_Hierarchical_Prompting",automl,task,dataset_name,sum_llm)

                    create_directory(output_directory)

                    # --------------------------------------------------------------
                    # File Paths
                    # --------------------------------------------------------------
                    fact_file = os.path.join(output_directory, "fact.txt")
                    global_summary_file = os.path.join(output_directory, "global_summary_T.txt")
                    verification_file = os.path.join(output_directory,"verification.json" )
                    revised_summary_file = os.path.join(output_directory, "global_summary_revised.txt")

                    # --------------------------------------------------------------
                    # Logs Path
                    # --------------------------------------------------------------
                    if automl == "ALPHA-AUTOML":

                        logs_path = os.path.join(root_dir,"results",automl,task,dataset_name, "filter_logs.txt")

                    elif automl == "AUTOSKLEARN":

                        logs_path = os.path.join( root_dir,"autosklearn_logs",task,dataset_name, "full_log_MainProcess.txt")

                    print(f" Logs Path : {logs_path}")
                    print(f"⚙️  Applying Hierarchical Prompting")

                    # ==============================================================
                    # STEP 1 — FACT EXTRACTION
                    # ==============================================================
                    print("\n Step 1: Phase Segmentation")

                    step_start = time.time()

                    fact_extraction(logs_path,sum_llm,fact_file)

                    print(
                        f"  Phase Segmentation completed in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

                    # ==============================================================
                    # STEP 2 — MACRO SUMMARIZATION
                    # ==============================================================
                    print("\n Step 2: Macro Summarization")

                    step_start = time.time()

                    macro_summarization(fact_file,sum_llm,global_summary_file,logs_path)

                    print(
                        f" Macro summarization completed in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

                    # ==============================================================
                    # STEP 3 — VERIFICATION
                    # ==============================================================
                    print("\n✔️  Step 3: Verification")

                    step_start = time.time()

                    verification(global_summary_file, logs_path,sum_llm,verification_file)

                    print(
                        f" Verification completed in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

                    # ==============================================================
                    # STEP 4 — REVISED SUMMARY
                    # ==============================================================
                    print("\n♻️  Step 4: Revised Summary Generation")

                    step_start = time.time()

                    revised_summary(global_summary_file,  logs_path,verification_file,sum_llm, revised_summary_file )

                    print(
                        f" Revised summary generated in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" HIERARCHICAL SUMMARIZATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)
        
        
        start_time = time.time()

        print("\n" + "=" * 80)
        print("⚖️  STARTING HIERARCHICAL SUMMARY EVALUATION")
        print("=" * 80)

        # ==========================================================================
        # MAIN LOOP
        # ==========================================================================
        for task in TASK:

            print(f"\n TASK: {task}")
            print("-" * 80)

            for dataset_name, sum_llm in product(dataset_names_for_task[task],LLMs):

                for automl in AUTOML:

                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # --------------------------------------------------------------
                    if (task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN" ):
                        print( f"⚠️  Skipping unsupported combination: " f"{automl} + {task}" )
                        continue

                    print("\n" + "─" * 80)
                    print(f" AutoML   : {automl}")
                    print(f" Dataset  : {dataset_name}")
                    print(f" Summary LLM : {sum_llm}")
                    print("─" * 80)

                    # --------------------------------------------------------------
                    # Output Directory
                    # --------------------------------------------------------------
                    output_directory = os.path.join(root_dir,"results_Hierarchical_Prompting", automl, task,dataset_name,sum_llm )

                    create_directory(output_directory)

                    # --------------------------------------------------------------
                    # Summary File
                    # --------------------------------------------------------------
                    summary_file = os.path.join( output_directory,"global_summary_revised.txt"   )

                    # --------------------------------------------------------------
                    # Logs Path
                    # --------------------------------------------------------------
                    if automl == "ALPHA-AUTOML":

                        logs_path = os.path.join(  root_dir, "results",automl, task,dataset_name, "filter_logs.txt"  )

                    elif automl == "AUTOSKLEARN":

                        logs_path = os.path.join( root_dir, "autosklearn_logs", task,dataset_name,  "full_log_MainProcess.txt"   )

                    print(f" Logs Path    : {logs_path}")
                    print(f" Summary File : {summary_file}")

                    # ==============================================================
                    # JUDGING LOOP
                    # ==============================================================
                    for llm_judge, judge_prompt in product(LLMs_judge,JUDGING_PROMPT):

                        print("\n" + "·" * 80)
                        print(f"⚖️  Judge LLM : {llm_judge}")
                        print(f" Prompt     : {judge_prompt}")
                        print("·" * 80)

                        # ----------------------------------------------------------
                        # Judge Output Directory
                        # ----------------------------------------------------------
                        final_dir = os.path.join( output_directory, judge_prompt  )
                        create_directory(final_dir)
                        judge_file = os.path.join(  final_dir,f"evaluation_{llm_judge}.txt"  )
                        print(f" Evaluation Output : {judge_file}")

                        # ----------------------------------------------------------
                        # Run Evaluation
                        # ----------------------------------------------------------
                        print("\n⚙️  Evaluating summary quality...")

                        eval_start = time.time()

                        judging_explanation(   logs_path, summary_file, llm_judge,judge_file,   judge_prompt    )

                        eval_end = time.time()

                        print(
                            f" Evaluation completed in "
                            f"{eval_end - eval_start:.2f} seconds"
                        )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" HIERARCHICAL SUMMARY EVALUATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)
      
    
    elif sys.argv[1] == "fit":

        start_time = time.time()

        print("\n" + "=" * 80)
        print(" STARTING MODEL TRAINING PIPELINE")
        print("=" * 80)

        for task_name in TASK:

            # Skip unsupported task
            if task_name == "TIME_SERIES_FORECAST":
                continue

            print(f"\n TASK: {task_name}")
            print("-" * 80)

            dataset_list = dataset_names_for_task[task_name]
            print(f" Datasets: {dataset_list}")

            # Load all datasets for current task
            datasets_dict = read_all_dataset(root_dir, task_name)

            base_output_dir = os.path.join(root_dir,"results", "ALPHA-AUTOML", task_name )

            for dataset_name in dataset_list:

                print("\n" + "─" * 80)
                print(f" DATASET: {dataset_name}")
                print("─" * 80)

                # Extract dataset components
                x_train = datasets_dict[dataset_name][0]
                y_train = datasets_dict[dataset_name][1]
                target_column = datasets_dict[dataset_name][2]
                date_column = datasets_dict[dataset_name][3]

                # Output paths
                output_directory = os.path.join(base_output_dir, dataset_name)
                output_dir = output_directory

                # Optional: create output directory
                # create_directory(output_directory)

                print("⚙️  Training model...")
                fit_start = time.time()

                create_fit_classifier(task_name, x_train,y_train,target_column,output_directory,date_column, output_dir)

                fit_end = time.time()
                
               
                print(f"Training completed in {fit_end - fit_start:.2f} seconds")

                # ------------------------------------------------------------------
                # Log Processing
                # ------------------------------------------------------------------
                print("\n Filtering logs...")

                logs_path = os.path.join(output_directory,"full_log_MainProcess.txt")

                temp_filtered_log = os.path.join(output_directory, "fil_tmp.txt")
                final_filtered_log = os.path.join(output_directory, "filter_logs.txt")

                filtering_logs(logs_path, temp_filtered_log)
                remove_consecutive_duplicates(temp_filtered_log, final_filtered_log)

                print("Logs filtered successfully")

        # --------------------------------------------------------------------------
        # Total Runtime Summary
        # --------------------------------------------------------------------------
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" PIPELINE COMPLETED")
        print("=" * 80)

        print(f"  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)
        
       
       
    elif sys.argv[1] == "flat_summarization":

        start_time = time.time()

        print("\n" + "=" * 80)
        print(" STARTING FLAT SUMMARIZATION PIPELINE")
        print("=" * 80)

        # ==========================================================================
        # AUTOSKLEARN SUMMARIZATION
        # ==========================================================================
        print("\n SUMMARIZING AUTOSKLEARN RESULTS")
        print("-" * 80)

        for task in ["CLASSIFICATION"]:

            print(f"\n TASK: {task}")

            for dataset_name, sum_llm, sum_prompt in product(
                dataset_names_for_task[task],
                LLMs,
                SUMMARIZATION_FLAT_PROMPT
            ):

                print("\n" + "─" * 80)
                print(f" DATASET : {dataset_name}")
                print(f" LLM     : {sum_llm}")
                print(f" PROMPT  : {sum_prompt}")
                print("─" * 80)

                # Output directory
                output_directory = os.path.join( root_dir, "results", "AUTOSKLEARN", task, dataset_name, sum_llm, sum_prompt)

                create_directory(output_directory)

                # Summary output file
                summary_dir = os.path.join( output_directory, "summary_result.txt" )

                # Logs path
                logs_path = os.path.join(root_dir,"autosklearn_logs",task,dataset_name, "full_log_MainProcess.txt" )

                print(f" Logs Path   : {logs_path}")
                print(f" Output File : {summary_dir}")

                # ------------------------------------------------------------------
                # Generate Summary
                # ------------------------------------------------------------------
                print("\n⚙️  Generating summary...")

                summary_start = time.time()

                explain_process( logs_path,sum_llm,summary_dir,sum_prompt )

                summary_end = time.time()

                print(
                    f"Summary generated in "
                    f"{summary_end - summary_start:.2f} seconds"
                )

        # ==========================================================================
        # ALPHA-AUTOML SUMMARIZATION
        # ==========================================================================
        print("\n" + "=" * 80)
        print(" SUMMARIZING ALPHA-AUTOML RESULTS")
        print("=" * 80)

        jobs = iterate_loop(prompt_strategy="FLAT_PROMPTING", automl="ALPHA-AUTOML")

        for (logs_path,file_dir,sum_prompt,output_directory,sum_llm, _, _,_) in jobs:

            print("\n" + "─" * 80)
            print(f" LLM     : {sum_llm}")
            print(f" PROMPT  : {sum_prompt}")
            print("─" * 80)

            print(f" Logs Path   : {logs_path}")
            print(f" Output File : {file_dir}")

            # ----------------------------------------------------------------------
            # Generate Summary
            # ----------------------------------------------------------------------
            print("\n⚙️  Generating summary...")

            summary_start = time.time()
            explain_process(logs_path,sum_llm,file_dir, sum_prompt)
            summary_end = time.time()
            print(
                f" Summary generated in "
                f"{summary_end - summary_start:.2f} seconds"
            )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("FLAT SUMMARIZATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)         
    
        
 
    ### summarization using Hierachical prompt modelling
    
    elif sys.argv[1] == "Hierachical_summarization":

        start_time = time.time()

        print("\n" + "=" * 80)
        print(" STARTING HIERARCHICAL SUMMARIZATION PIPELINE")
        print("=" * 80)

        # ==========================================================================
        # MAIN LOOP
        # ==========================================================================
        for task in TASK:

            print(f"\n TASK: {task}")
            print("-" * 80)

            for dataset_name, sum_llm in product(dataset_names_for_task[task], LLMs ):

                for automl in AUTOML:

                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # --------------------------------------------------------------
                    if ( task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN"):
                        print( f"⚠️  Skipping unsupported combination: "f"{automl} + {task}" )
                        continue

                    print("\n" + "─" * 80)
                    print(f" AutoML   : {automl}")
                    print(f" Dataset  : {dataset_name}")
                    print(f" LLM      : {sum_llm}")
                    print("─" * 80)

                    # --------------------------------------------------------------
                    # Output Directory
                    # --------------------------------------------------------------
                    output_directory = os.path.join(root_dir,"results_Hierarchical_Prompting",automl,task,dataset_name,sum_llm)

                    create_directory(output_directory)

                    # --------------------------------------------------------------
                    # File Paths
                    # --------------------------------------------------------------
                    fact_file = os.path.join(output_directory, "fact.txt")
                    global_summary_file = os.path.join(output_directory, "global_summary_T.txt")
                    verification_file = os.path.join(output_directory,"verification.json" )
                    revised_summary_file = os.path.join(output_directory, "global_summary_revised.txt")

                    # --------------------------------------------------------------
                    # Logs Path
                    # --------------------------------------------------------------
                    if automl == "ALPHA-AUTOML":

                        logs_path = os.path.join(root_dir,"results",automl,task,dataset_name, "filter_logs.txt")

                    elif automl == "AUTOSKLEARN":

                        logs_path = os.path.join( root_dir,"autosklearn_logs",task,dataset_name, "full_log_MainProcess.txt")

                    print(f" Logs Path : {logs_path}")
                    print(f"⚙️  Applying Hierarchical Prompting")

                    # ==============================================================
                    # STEP 1 — FACT EXTRACTION
                    # ==============================================================
                    print("\n Step 1: Phase Segmentation")

                    step_start = time.time()

                    fact_extraction(logs_path,sum_llm,fact_file)

                    print(
                        f"  Phase Segmentation completed in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

                    # ==============================================================
                    # STEP 2 — MACRO SUMMARIZATION
                    # ==============================================================
                    print("\n Step 2: Macro Summarization")

                    step_start = time.time()

                    macro_summarization(fact_file,sum_llm,global_summary_file,logs_path)

                    print(
                        f" Macro summarization completed in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

                    # ==============================================================
                    # STEP 3 — VERIFICATION
                    # ==============================================================
                    print("\n✔️  Step 3: Verification")
                    time.sleep(60)
                    step_start = time.time()

                    verification(global_summary_file, logs_path,sum_llm,verification_file)

                    print(
                        f" Verification completed in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

                    # ==============================================================
                    # STEP 4 — REVISED SUMMARY
                    # ==============================================================
                    print("\n♻️  Step 4: Revised Summary Generation")

                    step_start = time.time()

                    revised_summary(global_summary_file,  logs_path,verification_file,sum_llm, revised_summary_file )

                    print(
                        f" Revised summary generated in "
                        f"{time.time() - step_start:.2f} seconds"
                    )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" HIERARCHICAL SUMMARIZATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)

    
     
    elif sys.argv[1] == "judge_H":

        start_time = time.time()

        print("\n" + "=" * 80)
        print("⚖️  STARTING HIERARCHICAL SUMMARY EVALUATION")
        print("=" * 80)

        # ==========================================================================
        # MAIN LOOP
        # ==========================================================================
        for task in TASK:

            print(f"\n TASK: {task}")
            print("-" * 80)

            for dataset_name, sum_llm in product(dataset_names_for_task[task],LLMs):

                for automl in AUTOML:

                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # --------------------------------------------------------------
                    if (task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN" ):
                        print( f"⚠️  Skipping unsupported combination: " f"{automl} + {task}" )
                        continue

                    print("\n" + "─" * 80)
                    print(f" AutoML   : {automl}")
                    print(f" Dataset  : {dataset_name}")
                    print(f" Summary LLM : {sum_llm}")
                    print("─" * 80)

                    # --------------------------------------------------------------
                    # Output Directory
                    # --------------------------------------------------------------
                    output_directory = os.path.join(root_dir,"results_Hierarchical_Prompting", automl, task,dataset_name,sum_llm )

                    create_directory(output_directory)

                    # --------------------------------------------------------------
                    # Summary File
                    # --------------------------------------------------------------
                    summary_file = os.path.join( output_directory,"global_summary_revised.txt"   )

                    # --------------------------------------------------------------
                    # Logs Path
                    # --------------------------------------------------------------
                    if automl == "ALPHA-AUTOML":

                        logs_path = os.path.join(  root_dir, "results",automl, task,dataset_name, "filter_logs.txt"  )

                    elif automl == "AUTOSKLEARN":

                        logs_path = os.path.join( root_dir, "autosklearn_logs", task,dataset_name,  "full_log_MainProcess.txt"   )

                    print(f" Logs Path    : {logs_path}")
                    print(f" Summary File : {summary_file}")

                    # ==============================================================
                    # JUDGING LOOP
                    # ==============================================================
                    for llm_judge, judge_prompt in product(LLMs_judge,JUDGING_PROMPT):

                        print("\n" + "·" * 80)
                        print(f"⚖️  Judge LLM : {llm_judge}")
                        print(f" Prompt     : {judge_prompt}")
                        print("·" * 80)

                        # ----------------------------------------------------------
                        # Judge Output Directory
                        # ----------------------------------------------------------
                        final_dir = os.path.join( output_directory, judge_prompt  )
                        create_directory(final_dir)
                        judge_file = os.path.join(  final_dir,f"evaluation_{llm_judge}.txt"  )
                        print(f" Evaluation Output : {judge_file}")

                        # ----------------------------------------------------------
                        # Run Evaluation
                        # ----------------------------------------------------------
                        print("\n⚙️  Evaluating summary quality...")

                        eval_start = time.time()

                        judging_explanation(   logs_path, summary_file, llm_judge,judge_file,   judge_prompt    )

                        eval_end = time.time()

                        print(
                            f" Evaluation completed in "
                            f"{eval_end - eval_start:.2f} seconds"
                        )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" HIERARCHICAL SUMMARY EVALUATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")

        print("=" * 80)
      
         
      
    elif sys.argv[1] == "judge":
        
        
        start_time = time.time()

        print("\n" + "=" * 80)
        print("⚖️  STARTING FLAT SUMMARY EVALUATION")
        print("=" * 80)

        # ==========================================================================
        # MAIN LOOP
        # ==========================================================================
        for task in TASK:

            print(f"\n TASK: {task}")
            print("-" * 80)

            for dataset_name, sum_llm, sum_prompt in product(dataset_names_for_task[task],LLMs, SUMMARIZATION_FLAT_PROMPT):

                for automl in AUTOML:

                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # --------------------------------------------------------------
                    if (task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN" ):
                        print( f"⚠️  Skipping unsupported combination: " f"{automl} + {task}" )
                        continue

                    print("\n" + "─" * 80)
                    print(f" AutoML   : {automl}")
                    print(f" Dataset  : {dataset_name}")
                    print(f" Summary LLM : {sum_llm}")
                    print("─" * 80)

                    # --------------------------------------------------------------
                    # Output Directory
                    # --------------------------------------------------------------
                    output_directory = os.path.join(root_dir,"results", automl, task,dataset_name,sum_llm,sum_prompt )

                    create_directory(output_directory)

                    # --------------------------------------------------------------
                    # Summary File
                    # --------------------------------------------------------------
                    summary_file = os.path.join( output_directory,"summary_result.txt")

                    # --------------------------------------------------------------
                    # Logs Path
                    # --------------------------------------------------------------
                    if automl == "ALPHA-AUTOML":

                        logs_path = os.path.join(  root_dir, "results",automl, task,dataset_name, "filter_logs.txt"  )

                    elif automl == "AUTOSKLEARN":

                        logs_path = os.path.join( root_dir, "autosklearn_logs", task,dataset_name,  "full_log_MainProcess.txt"   )

                    print(f" Logs Path    : {logs_path}")
                    print(f" Summary File : {summary_file}")

                    # ==============================================================
                    # JUDGING LOOP
                    # ==============================================================
                    for llm_judge, judge_prompt in product(LLMs_judge,JUDGING_PROMPT):

                        print("\n" + "·" * 80)
                        print(f"⚖️  Judge LLM : {llm_judge}")
                        print(f" Prompt     : {judge_prompt}")
                        print("·" * 80)

                        # ----------------------------------------------------------
                        # Judge Output Directory
                        # ----------------------------------------------------------
                        final_dir = os.path.join( output_directory, judge_prompt  )
                        create_directory(final_dir)
                        judge_file = os.path.join(  final_dir,f"evaluation_{llm_judge}.txt"  )
                        print(f" Evaluation Output : {judge_file}")

                        # ----------------------------------------------------------
                        # Run Evaluation
                        # ----------------------------------------------------------
                        print("\n⚙️  Evaluating summary quality...")

                        eval_start = time.time()

                        judging_explanation(   logs_path, summary_file, llm_judge,judge_file,  judge_prompt)

                        eval_end = time.time()

                        print(
                            f" Evaluation completed in "
                            f"{eval_end - eval_start:.2f} seconds"
                        )

        # ==========================================================================
        # FINAL RUNTIME SUMMARY
        # ==========================================================================
        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(" FLAT SUMMARY EVALUATION COMPLETED")
        print("=" * 80)

        print("  Total Runtime:")
        print(f"   • Seconds : {total_time:.2f}s")
        print(f"   • Minutes : {total_time / 60:.2f} min")
        print(f"   • Hours   : {total_time / 3600:.2f} hr")
    
        print("=" * 80)
    
    
    
    
    
    # if sys.argv[1] == 'generate_csv_file':

    #     csv_df = []
    #     doc = []

    #     # prompt_strategy = "FLAT_PROMPTING"

    #     for prompt_strategy in PROMPT_STRATEGY:
    #         for automl in AUTOML:
    #             # if prompt_strategy == "HIERARCHICAL_PROMPTING":
    #             #     sum_prompt = "Hierachical prompt"

    #             jobs = iterate_loop(
    #                 prompt_strategy=prompt_strategy,
    #                 automl=automl
    #             )

    #             for (
    #                 logs_path,
    #                 summary_dir,
    #                 sum_prompt,
    #                 output_directory,
    #                 sum_llm,
    #                 task,
    #                 dataset_name,
    #                 automl
    #             ) in jobs:

    #                 if prompt_strategy == "HIERARCHICAL_PROMPTING":
    #                     sum_prompt = "Hierachical prompt"

    #                 for llm_judge in LLMs_judge:

    #                     judge_prompt = "zeroshot_judging"

    #                     judge_dir = os.path.join(
    #                         output_directory,
    #                         f'evaluation_{llm_judge}.txt'
    #                     )

    #                     print(f"judge file directory: {judge_dir}")
    #                     print("JUDGING PROMPT:", judge_prompt)

    #                     row = result_script(judge_dir)

    #                     result_dir = os.path.join(root_dir, "result.csv")

    #                     write_csv(
    #                         prompt_strategy,
    #                         automl,
    #                         task,
    #                         dataset_name,
    #                         sum_llm,
    #                         sum_prompt,
    #                         llm_judge,
    #                         judge_prompt,
    #                         row,
    #                         result_dir
    #                     )    
                
    
    



    elif sys.argv[1] == 'generate_csv_file':
        csv_df = [ ]
        doc = []
        # prompt_strategy = "FLAT_PROMPTING"
        for prompt_strategy in PROMPT_STRATEGY:
            print("OUTER LOOP:", prompt_strategy)
            for automl in AUTOML:
                jobs = iterate_loop(prompt_strategy=prompt_strategy, automl=automl)
                for logs_path, summary_dir, sum_prompt, output_directory, sum_llm, task, dataset_name, automl in jobs:
                    if prompt_strategy == "HIERARCHICAL_PROMPTING":
                        sum_prompt = "Hierarchical_prompt"
                    # --------------------------------------------------------------
                    # Skip unsupported combinations
                    # # --------------------------------------------------------------
                    # if (task in ["SEMISUPERVISED", "TIME_SERIES_FORECAST"]and automl == "AUTOSKLEARN" ):
                    #     print( f"⚠️  Skipping unsupported combination: " f"{automl} + {task}" )
                    #     continue
                    
                    for  llm_judge, judge_prompt in product(LLMs_judge,JUDGING_PROMPT ):

                        judge_dir = os.path.join(output_directory, judge_prompt, f'evaluation_{llm_judge}.txt')
                        if not os.path.exists(judge_dir):
                            continue    
                        # print(f"judge file directory: {judge_dir}")
                        # print("JUDGING PROMPT:", judge_prompt)        
                        row = result_script(judge_dir)
                        result_dir = os.path.join(root_dir, f"result.csv")
                        write_csv(prompt_strategy, automl,task,dataset_name, sum_llm, sum_prompt, llm_judge, judge_prompt, row , result_dir)     
            
                    
    elif sys.argv[1] == 'generate_graph':
        result_dir = os.path.join(root_dir, f"result.csv")
    
     
        generate_graph(result_dir)

    elif sys.argv[1] == "fit_1":

        if len(sys.argv) < 4:
            print("Usage: python script.py fit_1 <task_name> <dataset_name>")
            sys.exit(1)

        task_name = sys.argv[2]
        dataset_name = sys.argv[3]

        output_directory = os.path.join(
            root_dir,
            "results",
            "ALPHA-AUTOML",
            task_name,
            dataset_name
        )

        print("\n")
        print(f"TASK NAME: {task_name} | DATASET NAME: {dataset_name}")
        print("\n")

        create_directory(output_directory)
        
        datasets_dict = read_dataset(root_dir, task_name, dataset_name)      
        
        x_train = datasets_dict[dataset_name][0]
        y_train = datasets_dict[dataset_name][1]
        # x_test = datasets_dict[dataset_name][2]
        # y_test = datasets_dict[dataset_name][3] 
        # target_column = datasets_dict[dataset_name][4]
        
        target_column = datasets_dict[dataset_name][2]
        date_column = datasets_dict[dataset_name][3]


        print("-----------------START FITTING--------------")
        
        create_fit_classifier(task_name,x_train,y_train,target_column, output_directory, date_column, output_directory)

        print("\n")
        print("-----------------DONE FITTING---------------")
    
            
            
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
              
            #create_fit_classifier(task_name,x_train,y_train,target_column, output_directory, date_column, output_dir)
            print(" ")
            print('-----------------DONE FITTING---------------')

    
            file_dir =  os.path.join(output_directory , 'summary_result.txt')
    
        #  explain_process('full_log_MainProcess.txt', llm, file_dir)
            
        #  generate_results(root_dir, task_name, dataset_name, llm)
            
            print(" ")
            print('DONE SUMMARIZATION')
