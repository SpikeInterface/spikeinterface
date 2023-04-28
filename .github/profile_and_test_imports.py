import subprocess
import math

import_statement_list = [
    "import spikeinterface",
    "import spikeinterface.core",
    "import spikeinterface.extractors",
    "import spikeinterface.qualitymetrics",
    "import spikeinterface.preprocessing",
    "import spikeinterface.comparison",
    "import spikeinterface.postprocessing",
    "import spikeinterface.sortingcomponents",
    "import spikeinterface.curation",
]

markdown_output = "## Import Profiling\n\n| Module | Time (seconds) | Standard Deviation |\n| ------ | -------------- | ------------------ |\n"
n_samples = 10

exceptions = []

for import_statement in import_statement_list:
    time_taken_list = []
    for _ in range(n_samples):
        script_to_execute = (
                f"import timeit \n"
                f"import_statement = '{import_statement}' \n"
                f"time_taken = timeit.timeit(import_statement, number=1) \n"
                f"print(time_taken) \n"
               ) 

        result = subprocess.run(["python", "-c", script_to_execute], capture_output=True, text=True)

        if result.returncode != 0:
            error_message  = (
                f"Error when running {import_statement} \n"
                f"Error in subprocess: {result.stderr.strip()}\n"
            )
            exceptions.append(error_message)
            break

        time_taken = float(result.stdout.strip())
        time_taken_list.append(time_taken)

    if time_taken_list:
        avg_time_taken = sum(time_taken_list) / len(time_taken_list)
        std_dev_time_taken = math.sqrt(sum((x - avg_time_taken) ** 2 for x in time_taken_list) / len(time_taken_list))
        markdown_output += f"| {import_statement} | {avg_time_taken:.2f} | {std_dev_time_taken:.2f} |\n"

if exceptions:
    raise Exception("\n".join(exceptions))

# This is displayed to GithubSummary
print(markdown_output)