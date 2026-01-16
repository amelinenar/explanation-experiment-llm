# import re
# import csv
# from pathlib import Path
# def result_script(path):
    
#     text = Path(path).read_text(encoding="utf-8")
    
#     # document_path = path
#     # try:
#     #     with open(document_path) as f:
#     #         text = f.read()
#     # except OSError:
#     #     print("File not find")
        
#     #Regrex pattern to match lines like:
#     #1. Accuracy - Score: 4 (Good)
    
#     patterns = [
        
#         re.compile(
#       r"\d+\.\s+(?P<metric>[A-Za-z]+)\s+–\s+Score:\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)", re.I
#         ),
#         re.compile( r"\d+\.\s*\*{0,2}(?P<metric>[A-Za-z ]+)\s*[-–—]\s*Score:\s*(?P<score>\d+)\s*\((?P<label>[^)]+)\)\*{0,2}"
# , re.I ),
        
#    re.compile(
#       r"\s+###\d+\.\s+(?P<metric>[A-Za-z]+)\s+–\s+Score:\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)", re.I
#         ),
   
#      re.compile(
#       r"\s*\*{0,2}\d+\.\s+(?P<metric>[A-Za-z]+)\s+:\s *\*{0,2}\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)", re.I
#         ),
   
        
#     ]
    
#     results = []
    
#     for match in patterns.finditer(text):
#         results.append({
            
#             # "task": task,
#             # "dataset": dataset,
#             # "llm_summarizer": llm_summarizer,
#             # "summarization_prompt": summarization_prompt,
#             # "judging_llm": judging_llm,
#             # "judging_prompt": judging_prompt,
#             "metric": match.group("metric"),
#             "score": int(match.group("score")),
#             "label": match.group("label")
#         })
        
#     return results

# def write_csv(task, dataset, llm_summarizer, summarization_prompt, judging_llm, judging_prompt,results, csv_path):
#     """
#     writing the results to a csv document
#     """
    
#     fieldnames = ["task", "dataset", "llm_summarizer", "summarization_prompt", "judging_llm", "judging_prompt", "llm", "metric", "score","label"]
    
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in results:
#             row["task"] = task
#             row["dataset"] = dataset
#             row["llm_summarizer"] = llm_summarizer
#             row["summarization_prompt"] = summarization_prompt
#             row["judging_llm"] = judging_llm
#             row["judging_prompt"] = judging_prompt
       
#             #row["llm"] = llm_name
#             writer.writerow(row)    
            
            
            
import re
import csv
from pathlib import Path


def result_script(path):
    text = Path(path).read_text(encoding="utf-8")

    # Regex patterns to match different formats
    patterns = [
        re.compile(
            r"\d+\.\s+(?P<metric>[A-Za-z ]+)\s+[-–—]\s+Score:\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)",
            re.I,
        ),
        re.compile(
            r"\d+\.\s*\*{0,2}(?P<metric>[A-Za-z ]+)\s*[-–—]\s*Score:\s*(?P<score>\d+)\s*\((?P<label>[^)]+)\)\*{0,2}",
            re.I,
        ),
        re.compile(
            r"\s*###\s*\d+\.\s+(?P<metric>[A-Za-z ]+)\s+[-–—]\s+Score:\s+(?P<score>\d+)\s+\((?P<label>[^)]+)\)",
            re.I,
        ),
        re.compile(
            r"\s*\*{0,2}\d+\.\s+(?P<metric>[A-Za-z ]+)\s*:\s*\*{0,2}\s*(?P<score>\d+)\s+\((?P<label>[^)]+)\)",
            re.I,
        ),
    ]

    results = []

    for pattern in patterns:
        for match in pattern.finditer(text):
            results.append({
                "metric": match.group("metric").strip(),
                "score": int(match.group("score")),
                "label": match.group("label").strip(),
            })

    return results


def write_csv(
    task,
    dataset,
    llm_summarizer,
    summarization_prompt,
    judging_llm,
    judging_prompt,
    results,
    csv_path,
):
    """
    Write the results to a CSV file
    """

    fieldnames = [
        "task",
        "dataset",
        "llm_summarizer",
        "summarization_prompt",
        "judging_llm",
        "judging_prompt",
        "metric",
        "score",
        "label",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            
            row["task"] = task
            row["dataset"] = dataset
            row["llm_summarizer"] = llm_summarizer
            row["summarization_prompt"] = summarization_prompt
            row["judging_llm"] = judging_llm
            row["judging_prompt"] = judging_prompt
       
            # row.update({
            #     "task": task,
            #     "dataset": dataset,
            #     "llm_summarizer": llm_summarizer,
            #     "summarization_prompt": summarization_prompt,
            #     "judging_llm": judging_llm,
            #     "judging_prompt": judging_prompt,
            # })
            writer.writerow(row)
            
        # for row in results:
#             row["task"] = task
#             row["dataset"] = dataset
#             row["llm_summarizer"] = llm_summarizer
#             row["summarization_prompt"] = summarization_prompt
#             row["judging_llm"] = judging_llm
#             row["judging_prompt"] = judging_prompt
       
#             #row["llm"] = llm_name
#             writer.writerow(row)    
            
