"""
Some utils function to run command.
"""
from spikeinterface.core import load_extractor

from subprocess import Popen, PIPE, CalledProcessError, call, check_output
import shlex
import sys


def _run_command_and_print_output(command):
    command_list = shlex.split(command, posix="win" not in sys.platform)
    with Popen(command_list, stdout=PIPE, stderr=PIPE) as process:
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc


def _run_command_and_print_output_split(command_list):
    with Popen(command_list, stdout=PIPE, stderr=PIPE) as process:
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc


def _call_command(command):
    command_list = shlex.split(command, posix="win" not in sys.platform)
    try:
        call(command_list)
    except CalledProcessError as e:
        raise Exception(e.output)


def _call_command_split(command_list):
    try:
        call(command_list)
    except CalledProcessError as e:
        raise Exception(e.output)


def get_git_commit(git_folder, shorten=True):
    if git_folder is None:
        return None
    try:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        if shorten:
            commit = commit[:12]
    except:
        commit = None
    return commit


def recover_recording(rec_arg):
    if isinstance(rec_arg, dict):
        recording = load_extractor(rec_arg)
    else:
        recording = rec_arg
    return recording


class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""
