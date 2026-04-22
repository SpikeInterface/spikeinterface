import subprocess
import math

import_statement_list = [
    "import spikeinterface",
    "import spikeinterface.core",
    "import spikeinterface.extractors",
    "import spikeinterface.metrics",
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
    # Warm-up subprocess. Discarded from the timing data because the first
    # `python -c "import X"` after a fresh install pays cold-start cost
    # (filesystem cache priming, DLL loading, antivirus scanning a new
    # package) that has nothing to do with spikeinterface's actual import
    # weight. On Windows this can be 10x+ the steady-state and dominate
    # the average.
    warmup_script = (
        f"import timeit \n"
        f"import_statement = '{import_statement}' \n"
        f"timeit.timeit(import_statement, number=1) \n"
    )
    subprocess.run(["python", "-c", warmup_script], capture_output=True, text=True)

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
            error_message = (
                f"Error when running {import_statement} \n" f"Error in subprocess: {result.stderr.strip()}\n"
            )
            exceptions.append(error_message)
            break

        time_taken = float(result.stdout.strip())
        time_taken_list.append(time_taken)

    for time in time_taken_list:
        import_time_threshold = 3.0
        if time >= import_time_threshold:
            exceptions.append(
                f"Importing {import_statement} took: {time:.2f} s. Should be <: {import_time_threshold} s."
            )
            break

    if time_taken_list:
        avg_time = sum(time_taken_list) / len(time_taken_list)
        std_time = math.sqrt(sum((x - avg_time) ** 2 for x in time_taken_list) / len(time_taken_list))
        times_list_str = ", ".join(f"{time:.2f}" for time in time_taken_list)
        markdown_output += f"| `{import_statement}` | {avg_time:.2f} | {std_time:.2f} | {times_list_str} |\n"

        import_time_threshold = 2.0
        if avg_time > import_time_threshold:
            exceptions.append(
                f"Importing {import_statement} took: {avg_time:.2f} s in average. Should be <: {import_time_threshold} s."
            )

# This is displayed to GITHUB_STEP_SUMMARY. Print it before raising so the
# per-sample table is available even when the average threshold is exceeded.
print(markdown_output)

if exceptions:
    raise Exception("\n".join(exceptions))
