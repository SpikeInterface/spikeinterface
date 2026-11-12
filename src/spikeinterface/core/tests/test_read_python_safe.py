"""Class guard for the spikeinterface read_python() exec RCE (CWE-95).

1. read_python must NOT execute code from a config file (a side-effecting line
   produces no side effect; literal assignments still parse).
2. Lint: read_python's source contains no exec(/eval( of file contents.
Run: pytest test_read_python_safe.py -v
"""

import re
import inspect
from spikeinterface.core.core_tools import read_python


def test_read_python_does_not_execute_config(tmp_path):
    marker = tmp_path / "pwned.txt"
    p = tmp_path / "params.py"
    p.write_text(f"sample_rate = 30000\ndtype = 'int16'\nimport os\nos.system('echo x> {marker.as_posix()}')\n")
    md = read_python(str(p))
    assert not marker.exists(), "read_python executed code from the config file (CWE-95)"
    assert md.get("sample_rate") == 30000  # literal assignments still parse


def test_read_python_source_has_no_exec_eval():
    src = inspect.getsource(read_python)
    assert not re.search(r"(?<![\w.])exec\(", src), "read_python still uses exec()"
    assert not re.search(r"(?<![\w.])eval\(", src), "read_python still uses eval()"
