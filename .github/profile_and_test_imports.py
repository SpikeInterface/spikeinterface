import subprocess
import numpy as np

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

for import_statement in import_statement_list:
    time_taken_list = []
    for _ in range(n_samples):
        code = f"""
                import timeit
                import_statement = "{import_statement}"
                time_taken = timeit.timeit(import_statement, number=1)
                print(time_taken)
                """

        result = subprocess.run(["python", "-c", code], capture_output=True, text=True)

        if result.returncode != 0:
            error_message  = (
                f"Error when running {import_statement} \n"
                f"Error in subprocess: {result.stderr.strip()}\n"
            )
            raise ImportError(error_message)

        time_taken = float(result.stdout.strip())
        time_taken_list.append(time_taken)

    avg_time_taken = np.mean(time_taken_list)
    std_dev_time_taken = np.std(time_taken_list)
    markdown_output += f"| {import_statement} | {avg_time_taken:.2f} | {std_dev_time_taken:.2f} |\n"

print(markdown_output)