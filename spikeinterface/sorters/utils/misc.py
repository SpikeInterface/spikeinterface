
class SpikeSortingError(RuntimeError):
    """Raised whenever spike sorting fails"""
    
def get_git_commit(git_folder, shorten=True):
    """
    Get commit to generate sorters version.
    """
    if git_folder is None:
        return None
    try:
        commit = check_output(['git', 'rev-parse', 'HEAD'], cwd=git_folder).decode('utf8').strip()
        if shorten:
            commit = commit[:12]
    except:
        commit = None
    return commit


# Alessio : do we need this anymore ?
"""
def recover_recording(rec_arg):
    if isinstance(rec_arg, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(rec_arg)
    else:
        recording = rec_arg
    return recording
"""
