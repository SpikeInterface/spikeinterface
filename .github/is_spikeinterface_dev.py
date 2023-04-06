import spikeinterface

version = spikeinterface.__version__
if version.endswith("dev0"):
    print(True)