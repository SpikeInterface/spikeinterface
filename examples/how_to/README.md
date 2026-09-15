# Build documentation examples

In this examples folder, we only keep the light-weight .py files that will appear in the docs.

The `tutorials` folder will be built with `sphinx-gallery` (all examples starting with `plot_`) will be run in
Read The Docs and added to the documentation at run-time.

The `how_to` examples, instead, are manually run and added to the documentation.

To do so, we use `jupytext` to sync the .py files with an associated notebook, and then convert the notebook to .rst
with `nbconvert`. Here are the steps (in this example for the `get_started`):

1. Create a notebook from the .py file:

```
>>>  jupytext --to notebook get_started.py
>>>  jupytext --set-formats ipynb,py get_started.ipynb
```

2. Run the notebook

3. Sync the run notebook to the .py file:

```
>>> jupytext --sync get_started.ipynb
```

4. Convert the notebook to .rst

```
>>>  jupyter nbconvert get_started.ipynb --to rst
```

5. Move the .rst and associated folder (e.g. `get_started.rst` and `get_started_files` folder) to the `doc/how_to`.
