import os
import importlib.util


def has_tf(use_gpu=True, disable_tf_logger=True, memory_gpu=None):
    # first check if package even in environment so we don't just go to
    # the try-except. This should speed things up in the failure case, but
    # still provide the protection in case of rarer failure modes
    if importlib.util.find_spec("tensorflow") is None:
        return False
    else:
        try:
            import_tf(use_gpu, disable_tf_logger, memory_gpu)
            return True
        except ImportError:
            return False


def import_tf(use_gpu=True, disable_tf_logger=True, memory_gpu=None):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if disable_tf_logger:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import tensorflow as tf

    if disable_tf_logger:
        tf.get_logger().setLevel("ERROR")

    tf.compat.v1.disable_eager_execution()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and use_gpu:
        if memory_gpu is None:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        else:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=memory_gpu)]
                )
    return tf
