import importlib.metadata

package_name = "spikeinterface"
version = importlib.metadata.version(package_name)
if version.endswith("dev0"):
    print(True)
