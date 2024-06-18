from __future__ import annotations
import os
import warnings
from pathlib import Path
from typing import Callable, Optional

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .tf_utils import import_tf
from ...core import BaseRecording


global train_func


def train_deepinterpolation(
    recordings: BaseRecording | list[BaseRecording],
    model_folder: str | Path,
    model_name: str,
    desired_shape: tuple[int, int],
    train_start_s: Optional[float] = None,
    train_end_s: Optional[float] = None,
    train_duration_s: Optional[float] = None,
    test_start_s: Optional[float] = None,
    test_end_s: Optional[float] = None,
    test_duration_s: Optional[float] = None,
    test_recordings: Optional[BaseRecording | list[BaseRecording]] = None,
    pre_frame: int = 30,
    post_frame: int = 30,
    pre_post_omission: int = 1,
    existing_model_path: Optional[str | Path] = None,
    verbose: bool = True,
    nb_gpus: int = 1,
    steps_per_epoch: int = 10,
    period_save: int = 100,
    apply_learning_decay: int = 0,
    nb_times_through_data: int = 1,
    learning_rate: float = 0.0001,
    loss: str = "mean_squared_error",
    nb_workers: int = -1,
    caching_validation: bool = False,
    run_uid: str = "si",
    network: Callable | None = None,
    use_gpu: bool = True,
    disable_tf_logger: bool = True,
    memory_gpu: Optional[int] = None,
):
    """
    Train a deepinterpolation model from a recording extractor.

    Parameters
    ----------
    recordings : BaseRecording | list[BaseRecording]
        The recording(s) to be deepinteprolated. If a list is given, the recordings are concatenated
        and samples at the border of the recordings are omitted.
    model_folder : str | Path
        Path to the folder where the model will be saved
    model_name : str
        Name of the model to be used
    train_start_s : float or None, default: None
        Start time of the training in seconds. If None, the training starts at the beginning of the recording
    train_end_s : float or None, default: None
        End time of the training in seconds. If None, the training ends at the end of the recording
    train_duration_s : float, default: None
        Duration of the training in seconds. If None, the entire [train_start_s, train_end_s] is used for training
    test_start_s : float or None, default: None
        Start time of the testing in seconds. If None, the testing starts at the beginning of the recording
    test_end_s : float or None, default: None
        End time of the testing in seconds. If None, the testing ends at the end of the recording
    test_duration_s : float or None, default: None
        Duration of the testing in seconds, If None, the entire [test_start_s, test_end_s] is used for testing (not recommended)
    test_recordings : BaseRecording | list[BaseRecording] | None, default: None
        The recording(s) used for testing. If None, the training recording (or recordings) is used for testing
    desired_shape : tuple
        Shape of the input to the network
    pre_frame : int
        Number of frames before the frame to be predicted
    post_frame : int
        Number of frames after the frame to be predicted
    pre_post_omission : int
        Number of frames to be omitted before and after the frame to be predicted
    existing_model_path : str | Path | None, default: None
        Path to an existing model to be used for transfer learning
    verbose : bool, default: True
        Whether to print the progress of the training
    steps_per_epoch : int, default: 10
        Number of steps per epoch
    period_save : int, default: 100
        Period of saving the model
    apply_learning_decay : int, default: 0
        Whether to use a learning scheduler during training
    nb_times_through_data : int, default: 1
        Number of times the data is repeated during training
    learning_rate : float, default: 0.0001
        Learning rate
    loss : str, default: "mean_squared_error"
        Loss function to be used
    nb_workers : int, default: -1
        Number of workers to be used for the training
    caching_validation : bool, default: False
        Whether to cache the validation data
    run_uid : str, default: "si"
        Unique identifier for the training
    network : Callable or None, default: None
        Name deepinterpolation network to use. If None, the "unet_single_ephys_1024" network is used.
        The network should be a callable that takes a dictionary as input and returns a deepinterpolation network.
        See deepinterpolation.network_collection for examples.
    use_gpu : bool, default: True
        Whether to use GPU
    disable_tf_logger : bool, default: True
        Whether to disable the tensorflow logger
    memory_gpu : int, default: None
        Amount of memory to be used by the GPU

    Returns
    -------
    model_path : Path
        Path to the model
    """

    if nb_workers == -1:
        nb_workers = os.cpu_count()

    args = (
        recordings,
        model_folder,
        model_name,
        desired_shape,
        train_start_s,
        train_end_s,
        train_duration_s,
        test_start_s,
        test_end_s,
        test_duration_s,
        test_recordings,
        pre_frame,
        post_frame,
        pre_post_omission,
        existing_model_path,
        verbose,
        nb_gpus,
        steps_per_epoch,
        period_save,
        apply_learning_decay,
        nb_times_through_data,
        learning_rate,
        loss,
        nb_workers,
        caching_validation,
        run_uid,
        network,
        use_gpu,
        disable_tf_logger,
        memory_gpu,
    )
    global train_func
    train_func = train_deepinterpolation_process
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=nb_workers) as executor:
        f = executor.submit(train_func, *args)
        model_path = f.result()
    return model_path


def train_deepinterpolation_process(
    recordings: BaseRecording | list[BaseRecording],
    model_folder: str | Path,
    model_name: str,
    desired_shape: tuple[int, int],
    train_start_s: float,
    train_end_s: float,
    train_duration_s: float | None,
    test_start_s: float,
    test_end_s: float,
    test_duration_s: float | None,
    test_recordings: Optional[BaseRecording | list[BaseRecording]] = None,
    pre_frame: int = 30,
    post_frame: int = 30,
    pre_post_omission: int = 1,
    existing_model_path: Optional[str | Path] = None,
    verbose: bool = True,
    nb_gpus: int = 1,
    steps_per_epoch: int = 10,
    period_save: int = 100,
    apply_learning_decay: int = 0,
    nb_times_through_data: int = 1,
    learning_rate: float = 0.0001,
    loss: str = "mean_absolute_error",
    nb_workers: int = -1,
    caching_validation: bool = False,
    run_uid: str = "training",
    network: Callable | None = None,
    use_gpu: bool = True,
    disable_tf_logger: bool = True,
    memory_gpu: Optional[int] = None,
):
    from deepinterpolation.trainor_collection import core_trainer
    from .generators import SpikeInterfaceRecordingGenerator

    # initialize TF
    _ = import_tf(use_gpu, disable_tf_logger, memory_gpu=memory_gpu)

    trained_model_folder = Path(model_folder)
    trained_model_folder.mkdir(exist_ok=True, parents=True)

    # check if roughly zscored
    if not isinstance(recordings, list):
        recordings = [recordings]
    fs = recordings[0].sampling_frequency

    ### Define params
    start_frame_training = int(train_start_s * fs)
    end_frame_training = int(train_end_s * fs)
    if train_duration_s is not None:
        total_samples_training = int(train_duration_s * fs)
    else:
        total_samples_training = -1
    start_frame_test = int(test_start_s * fs)
    end_frame_test = int(test_end_s * fs)
    if test_duration_s is not None:
        total_samples_testing = int(test_duration_s * fs)
    else:
        total_samples_testing = -1

    if test_recordings is None:
        test_recordings = recordings
        if (start_frame_training <= start_frame_test < end_frame_training) or (
            start_frame_training < end_frame_test <= end_frame_training
        ):
            warnings.warn("Training and testing overlap. This is not recommended.")

    # # Those are parameters used for the network topology
    training_params = dict()
    training_params["output_dir"] = str(trained_model_folder)
    # We pass on the uid
    training_params["run_uid"] = run_uid

    # We convert to old schema
    training_params["nb_gpus"] = nb_gpus
    training_params["type"] = "trainer"
    training_params["steps_per_epoch"] = steps_per_epoch
    training_params["period_save"] = period_save
    training_params["apply_learning_decay"] = apply_learning_decay
    training_params["nb_times_through_data"] = nb_times_through_data
    training_params["learning_rate"] = learning_rate
    training_params["loss"] = loss
    training_params["nb_workers"] = nb_workers
    training_params["caching_validation"] = caching_validation
    training_params["model_string"] = model_name
    if existing_model_path:
        training_params["model_path"] = str(existing_model_path)

    # Training (from core_trainor class)
    training_data_generator = SpikeInterfaceRecordingGenerator(
        recordings,
        pre_frame=pre_frame,
        post_frame=post_frame,
        pre_post_omission=pre_post_omission,
        start_frame=start_frame_training,
        end_frame=end_frame_training,
        desired_shape=desired_shape,
        total_samples=total_samples_training,
    )
    test_data_generator = SpikeInterfaceRecordingGenerator(
        test_recordings,
        pre_frame=pre_frame,
        post_frame=post_frame,
        pre_post_omission=pre_post_omission,
        start_frame=start_frame_test,
        end_frame=end_frame_test,
        steps_per_epoch=-1,
        desired_shape=desired_shape,
        total_samples=total_samples_testing,
    )

    if network is None:
        from deepinterpolation.network_collection import unet_single_ephys_1024

        network_obj = unet_single_ephys_1024({})
    else:
        network_obj = network({})

    training_class = core_trainer(training_data_generator, test_data_generator, network_obj, training_params)

    if verbose:
        print("Created objects for training. Running training job")
    training_class.run()

    if verbose:
        print("Training job finished - finalizing output model")
    training_class.finalize()

    # Re-load model from output folder
    model_path = trained_model_folder / f"{training_params['run_uid']}_{training_params['model_string']}_model.h5"

    return model_path
