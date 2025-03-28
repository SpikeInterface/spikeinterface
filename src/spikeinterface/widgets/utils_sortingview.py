from __future__ import annotations

from warnings import warn

import numpy as np

from spikeinterface.core import SortingAnalyzer, BaseSorting
from spikeinterface.core.core_tools import check_json
from .utils import make_units_table_from_sorting, make_units_table_from_analyzer


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


def generate_unit_table_view(
    sorting_or_sorting_analyzer: SortingAnalyzer | BaseSorting,
    unit_properties: list[str] | None = None,
    similarity_scores: np.ndarray | None = None,
    extra_unit_properties: dict | None = None,
):
    import sortingview.views as vv

    if isinstance(sorting_or_sorting_analyzer, SortingAnalyzer):
        analyzer = sorting_or_sorting_analyzer
        units_tables = make_units_table_from_analyzer(analyzer, extra_properties=extra_unit_properties)
        sorting = analyzer.sorting
    else:
        sorting = sorting_or_sorting_analyzer
        units_tables = make_units_table_from_sorting(sorting)
        # analyzer = None

    if unit_properties is None:
        ut_columns = []
        ut_rows = [vv.UnitsTableRow(unit_id=u, values={}) for u in sorting.unit_ids]
    else:
        # keep only selected columns
        unit_properties = np.array(unit_properties)
        keep = np.isin(unit_properties, units_tables.columns)
        if sum(keep) < len(unit_properties):
            warn(f"Some unit properties are not in the sorting: {unit_properties[~keep]}")
        unit_properties = unit_properties[keep]
        units_tables = units_tables.loc[:, unit_properties]

        dtype_convertor = {"i": "int", "u": "int", "f": "float", "U": "str", "S": "str", "b": "bool"}
        # we add "O": "str" because pandas automatically converts strings to Object dtype
        dtype_convertor["O"] = "str"

        ut_columns = []
        for col in unit_properties:
            values = units_tables[col].to_numpy()
            if values.dtype.kind in dtype_convertor:
                txt_dtype = dtype_convertor[values.dtype.kind]
                ut_columns.append(vv.UnitsTableColumn(key=col, label=col, dtype=txt_dtype))

        ut_rows = []
        for unit_index, unit_id in enumerate(sorting.unit_ids):
            row_values = {}
            for col in unit_properties:
                values = units_tables[col].to_numpy()
                if values.dtype.kind in dtype_convertor:
                    value = values[unit_index]
                    if values.dtype.kind == "f":
                        # Check for NaN values and round floats
                        if np.isnan(values[unit_index]):
                            continue
                        value = np.format_float_positional(value, precision=4, fractional=False)
                    row_values[col] = value
            ut_rows.append(vv.UnitsTableRow(unit_id=unit_id, values=check_json(row_values)))

    v_units_table = vv.UnitsTable(rows=ut_rows, columns=ut_columns, similarity_scores=similarity_scores)

    return v_units_table
