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
# last_index = all_lines.index(next(line for line in all_lines if line in ['\n', '\r\n']))
# last_index = next(index for index, line in enumerate(all_lines[start_index:]) if len(line)==0) + start_index

last_index = next(index for index, line in enumerate(all_lines[start_index:]) if "===" in line) + start_index
#last_index = all_lines.index(last_line)

timing_info = all_lines[start_index:last_index]
timing_column = [float(line.split("s")[0].rstrip()) for line in timing_info]
short_name_column = [line.rpartition('::')[2] for line in timing_info]

data_frame = pd.DataFrame(dict(test_time=timing_column, test_name=short_name_column))
total_test_time = data_frame["test_time"].sum()
data_frame["%of_total_time"] = (100 * data_frame["test_time"] / total_test_time).round(2)
data_frame["%cum_total_time"] = (100 *  data_frame["test_time"].cumsum() / total_test_time).round(2)

# Build markdown string
data_frame_to_display = data_frame[["test_name", "test_time", "%of_total_time", "%cum_total_time"]]
markdown_string = data_frame_to_display.to_markdown()
    
# Output sort test summary info
sys.stdout.write("## Pytest summary of tests")
sys.stdout.write("\n")
sys.stdout.write(all_lines[-1])
sys.stdout.write("\n")
sys.stdout.write("## Disaggregated information")
sys.stdout.write("\n")
sys.stdout.write(f"\t Total running time (sum(test_time)) = {total_test_time}")
sys.stdout.write("\n")
sys.stdout.write(markdown_string)