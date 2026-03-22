import re
import os
from utils.utils import create_directory

# ================= INPUT =================
input_file = "/home/nguenang/Master_thesis/experiment_setup/results/CLASSIFICATION/185_baseball_dataset/full_log_MainProcess.txt"

with open(input_file, "r", encoding="utf-8") as f:
    log_text = f.read()


# ================= REGEX =================
ROUND_RE = re.compile(r'ITER\s+(\d+)')
GAME_RE = re.compile(r'MCTS SIMULATION\s+(\d+)')
PIPELINE_RE = re.compile(r'PIPELINE:\s*(.*)')


# ================= MAIN PARSER =================
def build_clean_structure(log_text):
    output_lines = []

    current_round = None
    current_game = None
    first_pipeline_in_game = False

    prev_pipeline = None  # for deduplication

    for line in log_text.splitlines():
        line = line.strip()

        # -------- ROUND --------
        round_match = ROUND_RE.search(line)
        if round_match:
            current_round = int(round_match.group(1))
            output_lines.append(f"Round {current_round}")
            continue

        # -------- GAME --------
        game_match = GAME_RE.search(line)
        if game_match:
            current_game = int(game_match.group(1))

            output_lines.append(f"Game {current_game}")
            output_lines.append("S")  # always start with S

            first_pipeline_in_game = True
            prev_pipeline = None
            continue

        # -------- PIPELINE --------
        pipe_match = PIPELINE_RE.search(line)
        if pipe_match:
            pipeline = pipe_match.group(1).strip()

            # convert format
            pipeline = pipeline.replace("|", ",")

            # remove consecutive duplicates
            if pipeline == prev_pipeline:
                continue

            output_lines.append(pipeline)
            prev_pipeline = pipeline

    return output_lines


# ================= BUILD =================
clean_lines = build_clean_structure(log_text)


# ================= OUTPUT =================
output_directory = "/home/nguenang/Master_thesis/experiment_setup/"
create_directory(output_directory)

output_file = os.path.join(output_directory, "clean_pipeline_dataset.txt")

with open(output_file, "w", encoding="utf-8") as f:
    for line in clean_lines:
        f.write(line + "\n")

print(f"✅ Clean structured file saved to: {output_file}")