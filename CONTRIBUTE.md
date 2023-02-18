# Contributors guide to spikeinterface

Before uploading committing any changes, please make sure to run [`black`](https://github.com/psf/black) and [`isort`](https://github.com/PyCQA/isort).

You can do so through `pre-commit`. You can install it in your environment by running the following:
```shell
pip install .[dev]
pre-commit install
```
You can now perform a commit, where `pre-commit` will try to `black` and `isort` all the files for you. You can also run it manually:
```shell
pre-commit run --all-files
```
