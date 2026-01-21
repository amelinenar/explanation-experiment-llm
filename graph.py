import os
from pandas import read_csv 

current_dir = os.getcwd()
path = os.path.join(current_dir, "metric.csv")
document = read_csv(path)

print()