# Contributors guide to spikeinterface

Before uploading committing any changes, please make sure to run `black` and `isort`.

You can do so through `pre-commit`:

```shell
pip install .[dev]
pre-commit install
pre-commit run --all-files
```

The first line installs `pre-commit` in the current environment—you don't need to install it yourself.
