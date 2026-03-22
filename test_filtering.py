import re,os,json
from collections import defaultdict
from utils.utils import create_directory

def parse_logs(full_logs):
    pipeline_pattern = re.compile(r"PIPELINE:\s*(.+)")
    column_proc_pattern = re.compile(r"Processing column \d+ '([^']+)'")
    column_type_pattern = re.compile(r"Column type ([^\[]+)")
    profiling_summary_pattern = re.compile(r"Results of profiling data:\s*(.+)")

    games = []
    current_game = []

    profiling = {"columns": [], "summary": None}
    last_column = None

    for line in full_logs.splitlines():

        # ---------------- PIPELINE ----------------
        pipe_match = pipeline_pattern.search(line)
        if pipe_match:
            pipeline = pipe_match.group(1).strip()
            pipeline_clean = pipeline.replace("|", ",")
            current_game.append(pipeline_clean)
            continue

        # ---------------- PROFILING ----------------
        col_match = column_proc_pattern.search(line)
        if col_match:
            last_column = col_match.group(1)
            profiling["columns"].append({
                "name": last_column,
                "type": None
            })
            continue

        type_match = column_type_pattern.search(line)
        if type_match and last_column:
            profiling["columns"][-1]["type"] = type_match.group(1).strip()
            continue

        summary_match = profiling_summary_pattern.search(line)
        if summary_match:
            profiling["summary"] = summary_match.group(1)
            continue

    if current_game:
        games.append(current_game)

    # format output
    formatted_output = []
    for i, game in enumerate(games):
        formatted_output.append(f"Game {i}")
        formatted_output.append("S")
        formatted_output.extend(game)

    return {
        "pipeline_steps": "\n".join(formatted_output),
        "profiling": profiling
    }


 
def save_result(destination_path):
    
    log_path = "/home/nguenang/Master_thesis/experiment_setup/results/CLASSIFICATION/299_libras_move/full_log_MainProcess.txt"
    res = parse_logs(log_path)
    print(res)
    # with open(destination_path, "w", encoding="utf-8") as f:
    #     f.write(res)
    try:
        with open(destination_path, "w") as f:
            json.dump(res, f, indent=2)
    except FileExistsError:
        print("File does not exist")
    


#Create document


output_directory = "/home/nguenang/Master_thesis/experiment_setup/"
create_directory(output_directory)
file_path =  os.path.join(output_directory , 'test_log_analysis.txt')

save_result(file_path)
 
