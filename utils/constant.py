CLASSIFICATION_DATASET = ['299_libras_move']
REGRESSION_DATASET = ['196_autoMpg']
SEMI_SUPERVISED_DATASET = ['LL0_1053_jm1']
TIMESERIES_DATASET = ['stock_market']
TASK = ['REGRESSION', 'CLASSIFICATION', 'SEMISUPERVISED' , 'TIME_SERIES_FORECAST']
METRICS = []
ARCHIVE_NAMES = ['dataset']
dataset_names_for_task = {'REGRESSION': REGRESSION_DATASET,'CLASSIFICATION':CLASSIFICATION_DATASET, 'SEMISUPERVISED':SEMI_SUPERVISED_DATASET,'TIME_SERIES_FORECAST':TIMESERIES_DATASET}
#LLMs = { 'deepseek-r1:14b','llama4'}
LLMs = {'llama4'}
#PROMPT = ['summarization_prompt', 'judging_prompt']
SUMMARIZATION_PROMPT = ['zeroshot', 'zeroshot_instruction', 'fewshot', 'chain_of_thought','zeroshot_CoT']
#SUMMARIZATION_PROMPT = ['zero_shot']
JUDGING_PROMPT = ['test_judge']
#JUDGING_PROMPT = ['judge_zeroshot', 'judge_zeroshotInstruction', 'test_judge', 'judging_prompt_zeroShot']
#prompt_for_promptType ={'summarization'}
LLMs_judge = {'deepseek-r1:14b'}

