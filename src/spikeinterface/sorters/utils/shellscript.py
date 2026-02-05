from __future__ import annotations

import subprocess
import tempfile
import shutil
import signal
import os
from pathlib import Path
import time
import sys
from typing import Optional, List, Any, Union

PathType = Union[str, Path]


class ShellScript:
    def __init__(
        self,
        script: str,
        script_path: Optional[PathType] = None,
        log_path: Optional[PathType] = None,
        keep_temp_files: bool = False,
        verbose: bool = False,
    ):
        lines = script.splitlines()
        lines = self._remove_initial_blank_lines(lines)
        if len(lines) > 0:
            num_initial_spaces = self._get_num_initial_spaces(lines[0])
            for ii, line in enumerate(lines):
                if len(line.strip()) > 0:
                    n = self._get_num_initial_spaces(line)
                    if n < num_initial_spaces:
                        print(script)
                        raise Exception("Problem in script. First line must not be indented relative to others")
                    lines[ii] = lines[ii][num_initial_spaces:]
        self._script = "\n".join(lines)
        self._script_path = script_path
        self._log_path = log_path
        self._keep_temp_files = keep_temp_files
        self._process: Optional[subprocess.Popen] = None
        self._files_to_remove: List[str] = []
        self._dirs_to_remove: List[str] = []
        self._start_time: Optional[float] = None
        self._verbose = verbose

    def __del__(self):
        self.cleanup()

    def substitute(self, old: str, new: Any) -> None:
        self._script = self._script.replace(old, "{}".format(new))

    def write(self, script_path: Optional[str] = None) -> None:
        if script_path is None:
            script_path = self._script_path
        if script_path is None:
            raise Exception("Cannot write script. No path specified")
        with open(script_path, "w") as f:
            f.write(self._script)
        os.chmod(script_path, 0o744)

    def start(self) -> None:
        if self._script_path is not None:
            script_path = Path(self._script_path)
            if script_path.suffix == "":
                if "win" in sys.platform and sys.platform != "darwin":
                    script_path = script_path.parent / (script_path.name + ".bat")
                else:
                    script_path = script_path.parent / (script_path.name + ".sh")
        else:
            tempdir = Path(tempfile.mkdtemp(prefix="tmp_shellscript"))
            if "win" in sys.platform and sys.platform != "darwin":
                script_path = tempdir / "script.bat"
            else:
                script_path = tempdir / "script.sh"
            self._dirs_to_remove.append(tempdir)

        if self._log_path is None:
            script_log_path = script_path.parent / "spike_sorters_log.txt"
        else:
            script_log_path = Path(self._log_path)
            if script_path.suffix == "":
                script_log_path = script_log_path.parent / (script_log_path.name + ".txt")

        self.write(script_path)
        cmd = str(script_path)
        if self._verbose:
            print("RUNNING SHELL SCRIPT: " + cmd)
        self._start_time = time.time()
        encoding = sys.getdefaultencoding()
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, encoding=encoding
        )
        with open(script_log_path, "w+") as script_log_file:
            for line in self._process.stdout:
                script_log_file.write(line)
                if self._verbose:
                    # Print onto console depending on the verbose property passed on from the sorter class
                    print(line)

    def wait(self, timeout=None) -> Optional[int]:
        if not self.isRunning():
            return self.returnCode()
        assert self._process is not None, "Unexpected self._process is None even though it is running."
        try:
            retcode = self._process.wait(timeout=timeout)
            return retcode
        except:
            return None

    def cleanup(self) -> None:
        if self._keep_temp_files:
            return
        for dirpath in self._dirs_to_remove:
            _rmdir_with_retries(str(dirpath), num_retries=5)
        if self._process is not None:
            if self._process.stdout:
                self._process.stdout.close()
            if self._process.stderr:
                self._process.stderr.close()
            self._process.kill()

    def stop(self) -> None:
        if not self.isRunning():
            return
        assert self._process is not None, "Unexpected self._process is None even though it is running."

        signals = [signal.SIGINT] * 10 + [signal.SIGTERM] * 10 + [signal.SIGKILL] * 10

        for signal0 in signals:
            self._process.send_signal(signal0)
            try:
                self._process.wait(timeout=0.02)
                return
            except:
                pass

    def kill(self) -> None:
        if not self.isRunning():
            return

        assert self._process is not None, "Unexpected self._process is None even though it is running."
        self._process.send_signal(signal.SIGKILL)
        try:
            self._process.wait(timeout=1)
        except:
            print("WARNING: unable to kill shell script.")
            pass

    def stopWithSignal(self, sig, timeout) -> bool:
        if not self.isRunning():
            return True

        assert self._process is not None, "Unexpected self._process is None even though it is running."
        self._process.send_signal(sig)
        try:
            self._process.wait(timeout=timeout)
            return True
        except:
            return False

    def elapsedTimeSinceStart(self) -> Optional[float]:
        if self._start_time is None:
            return None

        return time.time() - self._start_time

    def isRunning(self) -> bool:
        if not self._process:
            return False
        retcode = self._process.poll()
        if retcode is None:
            return True
        return False

    def isFinished(self) -> bool:
        if not self._process:
            return False
        return not self.isRunning()

    def returnCode(self) -> Optional[int]:
        if not self.isFinished():
            raise Exception("Cannot get return code before process is finished.")
        assert self._process is not None, "Unexpected self._process is None even though it is finished."
        return self._process.returncode

    def scriptPath(self) -> Optional[str]:
        return self._script_path

    def _remove_initial_blank_lines(self, lines: List[str]) -> List[str]:
        ii = 0
        while ii < len(lines) and len(lines[ii].strip()) == 0:
            ii = ii + 1
        return lines[ii:]

    def _get_num_initial_spaces(self, line: str) -> int:
        ii = 0
        while ii < len(line) and line[ii] == " ":
            ii = ii + 1
        return ii


def _rmdir_with_retries(dirname, num_retries, delay_between_tries=1):
    for retry_num in range(1, num_retries + 1):
        if not os.path.exists(dirname):
            return
        try:
            shutil.rmtree(dirname)
            break
        except:
            if retry_num < num_retries:
                print("Retrying to remove directory: {}".format(dirname))
                time.sleep(delay_between_tries)
            else:
                raise Exception("Unable to remove directory after {} tries: {}".format(num_retries, dirname))
