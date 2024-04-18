from __future__ import annotations

import numpy as np

from ..core.core_tools import check_json


def make_serializable(*args):
    dict_to_serialize = {int(i): a for i, a in enumerate(args)}
    serializable_dict = check_json(dict_to_serialize)
    returns = ()
    for i in range(len(args)):
        returns += (serializable_dict[str(i)],)
    if len(returns) == 1:
        returns = returns[0]
    return returns


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def handle_display_and_url(widget, view, **backend_kwargs):
    url = None
    # TODO: put this back when figurl-jupyter is working again
    # if is_notebook() and backend_kwargs["display"]:
    #     display(view.jupyter(height=backend_kwargs["height"]))
    if backend_kwargs["generate_url"]:
        figlabel = backend_kwargs.get("figlabel")
        if figlabel is None:
            # figlabel = widget.default_label
            figlabel = ""
        url = view.url(label=figlabel)
        print(url)

    return url


def generate_unit_table_view(sorting, unit_properties=None, similarity_scores=None):
    import sortingview.views as vv

    if unit_properties is None:
        ut_columns = []
        ut_rows = [vv.UnitsTableRow(unit_id=u, values={}) for u in sorting.unit_ids]
    else:
        ut_columns = []
        ut_rows = []
        values = {}
        valid_unit_properties = []
        for prop_name in unit_properties:
            property_values = sorting.get_property(prop_name)
            # make dtype available
            val0 = np.array(property_values[0])
            if val0.dtype.kind in ("i", "u"):
                dtype = "int"
            elif val0.dtype.kind in ("U", "S"):
                dtype = "str"
            elif val0.dtype.kind == "f":
                dtype = "float"
            elif val0.dtype.kind == "b":
                dtype = "bool"
            else:
                print(f"Unsupported dtype {val0.dtype} for property {prop_name}. Skipping")
                continue
            ut_columns.append(vv.UnitsTableColumn(key=prop_name, label=prop_name, dtype=dtype))
            valid_unit_properties.append(prop_name)

        for ui, unit in enumerate(sorting.unit_ids):
            for prop_name in valid_unit_properties:
                property_values = sorting.get_property(prop_name)
                val0 = np.array(property_values[0])
                if val0.dtype.kind == "f":
                    if np.isnan(property_values[ui]):
                        continue
                values[prop_name] = property_values[ui]
            ut_rows.append(vv.UnitsTableRow(unit_id=unit, values=check_json(values)))

    v_units_table = vv.UnitsTable(rows=ut_rows, columns=ut_columns, similarity_scores=similarity_scores)
    return v_units_table
