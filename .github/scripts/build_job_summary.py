"""
This function builds a summary from the pytest output in markdown to be used by GITHUB_STEP_SUMMARY.
The input file is the output of the following command:
pytest -vv --durations=0 --durations-min=0.001 > report.txt
"""
import re
from pathlib import Path
import pandas as pd
import sys


file_path = Path(sys.argv[1])
assert file_path.is_file(), "Input file with testing times not found"
file_path_out = Path(file_path).with_suffix(".md")

all_text = file_path.read_text()
all_lines = all_text.splitlines()

# find the line with the line with slowest in text
start_line = next(line for line in all_lines if "slowest" in line)
start_index = all_lines.index(start_line) + 1

last_index = next(index for index, line in enumerate(all_lines[start_index:]) if "===" in line) + start_index

# Pytest 8.4+ appends a blank line and a "(N durations < Xs hidden.)" footer
# inside the slowest-durations block. Keep only true duration rows shaped like
# "0.123s call test_x.py::test_name".
duration_line_re = re.compile(r"^\d+\.\d+s\s+(call|setup|teardown)\s")
timing_info = [line for line in all_lines[start_index:last_index] if duration_line_re.match(line)]
timing_column = [float(line.split("s")[0].rstrip()) for line in timing_info]
type = [line.split("s")[1].rstrip() for line in timing_info]
short_name = [line.rpartition('::')[2] for line in timing_info]
long_name = [line.rpartition('/')[2] for line in timing_info]

data_frame = pd.DataFrame(dict(test_time=timing_column, test_name=short_name, long_name=long_name, type=type))
total_test_time = data_frame["test_time"].sum()
data_frame["%of_total_time"] = (100 * data_frame["test_time"] / total_test_time).round(2)
data_frame["%cum_total_time"] = (100 *  data_frame["test_time"].cumsum() / total_test_time).round(2)

# Build markdown strings
data_frame_to_display = data_frame[["test_name", "type", "test_time", "%of_total_time", "%cum_total_time", "long_name"]]
data_frame_header_markdown = data_frame_to_display.head(10).to_markdown()
data_frame_markdown = data_frame_to_display.to_markdown()

# Build GITHUB_STEP_SUMMARY markdown file
sys.stdout.write("## Pytest summary")
sys.stdout.write("\n \n")
sys.stdout.write(all_lines[-1])
sys.stdout.write("\n \n")
sys.stdout.write("## Disaggregated information")
sys.stdout.write("\n \n")
sys.stdout.write(f"\t Total running time (sum(test_time)) = {total_test_time:.2f} s")
sys.stdout.write("\n \n")
sys.stdout.write("### Top 10 slowest tests")
sys.stdout.write("\n \n")
sys.stdout.write(data_frame_header_markdown)
sys.stdout.write("\n \n")
sys.stdout.write("\n \n")
sys.stdout.write("\n \n")
sys.stdout.write("\n \n")
sys.stdout.write("<details>")
sys.stdout.write("<summary> click here to see the full table </summary>")
sys.stdout.write("\n \n")
sys.stdout.write(data_frame_markdown)
sys.stdout.write("\n \n")
sys.stdout.write("</details>")
sys.stdout.write("\n \n")
