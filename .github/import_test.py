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
    "import spikeinterface.exporters",
    "import spikeinterface.widgets",
    "import spikeinterface.full",
]

n_samples = 10
# Note that the symbols at the end are for centering the table
markdown_output = f"## \n\n| Imported Module ({n_samples=}) | Importing Time (seconds) | Standard Deviation (seconds) | Times List (seconds) |\n| :--: | :--------------: | :------------------: | :-------------: |\n"

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

    for time in time_taken_list:
        if time > 2.5:
            exceptions.append(f"Importing {import_statement} took too long: {time:.2f} seconds")
            break

    if time_taken_list:
        avg_time_taken = sum(time_taken_list) / len(time_taken_list)
        std_dev_time_taken = math.sqrt(sum((x - avg_time_taken) ** 2 for x in time_taken_list) / len(time_taken_list))
        times_list_str = ", ".join(f"{time:.2f}" for time in time_taken_list)
        markdown_output += f"| `{import_statement}` | {avg_time_taken:.2f} | {std_dev_time_taken:.2f} | {times_list_str} |\n"

if exceptions:
    raise Exception("\n".join(exceptions))

# This is displayed to GITHUB_STEP_SUMMARY
print(markdown_output)
