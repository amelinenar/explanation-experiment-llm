import re, os
from collections import defaultdict
from utils.utils import create_directory

with open("/home/nguenang/Master_thesis/experiment_setup/results/CLASSIFICATION/299_libras_move/full_log_MainProcess.txt") as f:
   log_text =  f.read()
   

# ================= REGEX =================
ITER_RE = re.compile(r'ITER\s+(\d+)')
SIM_RE = re.compile(r'MCTS SIMULATION\s+(\d+)')
PIPE_RE = re.compile(r'PIPELINE:\s*(.*)')
COACH_RE = re.compile(r'COACH ACTION\s+(\d+)')
PIPELINE_CREATED_RE = re.compile(r'New pipelined created:')
SCORE_RE = re.compile(r'Score:\s*([0-9.]+)')
SUCCESS_RE = re.compile(r'Pipeline scored successfully')
FAIL_RE = re.compile(r'findwin\s+-1')

# Profiling
PROFILING_START_RE = re.compile(r'Identifying types,\s*(\d+)\s*columns')
COLUMN_PROCESS_RE = re.compile(r"Processing column\s+(\d+)\s+'([^']+)'")
COLUMN_TYPE_RE = re.compile(r'Column type\s+(.+)')
PROFILING_SUMMARY_RE = re.compile(r'Results of profiling data:\s*(.*)')

def parse_logs(log_text):
    data = defaultdict(lambda: {
        "simulations": defaultdict(lambda: {
            "pipelines": [],
            "score": None,
            "status": None
        }),
        "coach_actions": []
    })

    current_iter = None
    current_sim = None

    for line in log_text.splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            continue

        sim_match = SIM_RE.search(line)
        if sim_match:
            current_sim = int(sim_match.group(1))
            continue

        pipe_match = PIPE_RE.search(line)
        if pipe_match and current_iter and current_sim:
            data[current_iter]["simulations"][current_sim]["pipelines"].append(pipe_match.group(1))

        coach_match = COACH_RE.search(line)
        if coach_match and current_iter:
            data[current_iter]["coach_actions"].append(int(coach_match.group(1)))

        if PIPELINE_CREATED_RE.search(line) and current_iter and current_sim:
            data[current_iter]["simulations"][current_sim]["status"] = "created"

        score_match = SCORE_RE.search(line)
        if score_match and current_iter and current_sim:
            data[current_iter]["simulations"][current_sim]["score"] = float(score_match.group(1))

        if SUCCESS_RE.search(line) and current_iter and current_sim:
            data[current_iter]["simulations"][current_sim]["status"] = "success"

        if FAIL_RE.search(line) and current_iter and current_sim:
            data[current_iter]["simulations"][current_sim]["status"] = "failed"

    return data

parsed = parse_logs(log_text)

#Create document


output_directory = "/home/nguenang/Master_thesis/experiment_setup/"
create_directory(output_directory)
file_path =  os.path.join(output_directory , 'log_analysis.txt')
output_file = os.path.join(output_directory , 'log_analysis_out.txt')





with open(file_path, "w", encoding="utf-8") as f:
    f.write("LOG ANALYSIS REPORT\n")
    f.write("=" * 50 + "\n\n")

    for iter_id, iter_data in parsed.items():
        f.write(f"ITERATION {iter_id}\n")
        f.write("-" * 40 + "\n")

        f.write(f"Coach Actions: {iter_data['coach_actions']}\n\n")

        for sim_id, sim_data in iter_data["simulations"].items():
            f.write(f"  Simulation {sim_id}\n")
            f.write(f"    Status: {sim_data['status']}\n")
            f.write(f"    Score: {sim_data['score']}\n")

            f.write("    Pipelines:\n")
            for p in sim_data["pipelines"]:
                f.write(f"      - {p}\n")

            f.write("\n")  # spacing between simulations

        f.write("\n")  # spacing between iterations
        

def remove_consecutive_duplicates(file_path, output_file):
    prev_line = None

    with open(file_path, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            if line != prev_line:
                fout.write(line)
                prev_line = line


print(f"Saved to {file_path}")

# usage
remove_consecutive_duplicates(file_path, output_file)





    