from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .tf_utils import import_tf
from .generators import define_recording_generator_class

from ...core import BaseRecording


global train_func


def train_deepinterpolation(
    recording: BaseRecording,
    model_folder: str | Path,
    model_name: str,
    train_start_s: float,
    train_end_s: float,
    test_start_s: float,
    test_end_s: float,
    desired_shape: tuple[int, int],
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
    run_uid: str = "si",
    network_name: str = "unet_single_ephys_1024",
    use_gpu: bool = True,
    disable_tf_logger: bool = True,
    memory_gpu: Optional[int] = None,
):
    """
    Train a deepinterpolation model from a recording extractor.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor to be deepinteprolated
    model_folder : str | Path
        Path to the folder where the model will be saved
    model_name : str
        Name of the model to be used
    train_start_s : float
        Start time of the training in seconds
    train_end_s : float
        End time of the training in seconds
    test_start_s : float
        Start time of the testing in seconds
    test_end_s : float
        End time of the testing in seconds
    desired_shape : tuple
        Shape of the input to the network
    pre_frame : int
        Number of frames before the frame to be predicted
    post_frame : int
        Number of frames after the frame to be predicted
    pre_post_omission : int
        Number of frames to be omitted before and after the frame to be predicted
    existing_model_path : str | Path
        Path to an existing model to be used for transfer learning, default is None
    verbose : bool
        Whether to print the progress of the training, default is True
    steps_per_epoch : int
        Number of steps per epoch
    period_save : int
        Period of saving the model
    apply_learning_decay : int
        Whether to use a learning scheduler during training
    nb_times_through_data : int
        Number of times the data is repeated during training
    learning_rate : float
        Learning rate
    loss : str
        Loss function to be used
    nb_workers : int
        Number of workers to be used for the training
    run_uid : str
        Unique identifier for the training
    network_name : str
        Name of the network to be used, default is None
    use_gpu : bool
        Whether to use GPU, default is True
    disable_tf_logger : bool
        Whether to disable the tensorflow logger, default is True
    memory_gpu : int
        Amount of memory to be used by the GPU, default is None

    Returns
    -------
    model_path : Path
        Path to the model
    """

    # q: can you list all argument of the train_deepinterpolation function in a tuple?
    #   a: yes, see below
    # q: can you list all argument of the train_deepinterpolation function in a tuple?

    args = (
        recording,
        model_folder,
        model_name,
        train_start_s,
        train_end_s,
        test_start_s,
        test_end_s,
        pre_frame,
        post_frame,
        pre_post_omission,
        desired_shape,
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
        run_uid,
        network_name,
        use_gpu,
        disable_tf_logger,
        memory_gpu,
    )
    global train_func
    train_func = train_deepinterpolation_process
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        f = executor.submit(train_func, *args)
        model_path = f.result()
    return model_path


def train_deepinterpolation_process(
    recording: BaseRecording,
    model_folder: str | Path,
    model_name: str,
    train_start_s: float,
    train_end_s: float,
    test_start_s: float,
    test_end_s: float,
    pre_frame: int,
    post_frame: int,
    pre_post_omission: int,
    desired_shape: tuple[int, int],
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
    run_uid: str = "training",
    network_name: str = "unet_single_ephys_1024",
    use_gpu: bool = True,
    disable_tf_logger: bool = True,
    memory_gpu: Optional[int] = None,
):
    from deepinterpolation.trainor_collection import core_trainer
    from deepinterpolation.generic import ClassLoader

    # initialize TF
    _ = import_tf(use_gpu, disable_tf_logger, memory_gpu=memory_gpu)

    recording_generator = define_recording_generator_class()

    trained_model_folder = Path(model_folder)
    trained_model_folder.mkdir(exist_ok=True)

    # check if roughly zscored
    fs = recording.sampling_frequency

    ### Define params
    start_frame_training = int(train_start_s * fs)
    end_frame_training = int(train_end_s * fs)
    start_frame_test = int(test_start_s * fs)
    end_frame_test = int(test_end_s * fs)

    # Those are parameters used for the network topology
    network_params = dict()
    network_params["type"] = "network"
    # Name of network topology in the collection
    network_params["name"] = network_name if network_name is not None else "unet_single_ephys_1024"
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
    training_params["caching_validation"] = False
    training_params["model_string"] = model_name
    if existing_model_path:
        training_params["existing_model_path"] = str(existing_model_path)

    # Training (from core_trainor class)
    training_data_generator = recording_generator(
        recording,
        pre_frame=pre_frame,
        post_frame=post_frame,
        pre_post_omission=pre_post_omission,
        start_frame=start_frame_training,
        end_frame=end_frame_training,
        desired_shape=desired_shape,
    )
    test_data_generator = recording_generator(
        recording,
        pre_frame=pre_frame,
        post_frame=post_frame,
        pre_post_omission=pre_post_omission,
        start_frame=start_frame_test,
        end_frame=end_frame_test,
        steps_per_epoch=-1,
        desired_shape=desired_shape,
    )

    network_json_path = trained_model_folder / "network_params.json"
    with open(network_json_path, "w") as f:
        json.dump(network_params, f)

    network_obj = ClassLoader(network_json_path)
    data_network = network_obj.find_and_build()(network_json_path)

    training_json_path = trained_model_folder / "training_params.json"
    with open(training_json_path, "w") as f:
        json.dump(training_params, f)

    training_class = core_trainer(training_data_generator, test_data_generator, data_network, training_json_path)

    if verbose:
        print("Created objects for training. Running training job")
    training_class.run()

    if verbose:
        print("Training job finished - finalizing output model")
    training_class.finalize()

    # Re-load model from output folder
    model_path = trained_model_folder / f"{training_params['run_uid']}_{training_params['model_string']}_model.h5"

    return model_path
