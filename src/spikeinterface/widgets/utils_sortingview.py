from __future__ import annotations

import numpy as np

from ..core import SortingAnalyzer, BaseSorting
from ..core.core_tools import check_json
from warnings import warn


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
    similarity_scores: npndarray | None = None,
):
    import sortingview.views as vv

    if isinstance(sorting_or_sorting_analyzer, SortingAnalyzer):
        analyzer = sorting_or_sorting_analyzer
        sorting = analyzer.sorting
    else:
        sorting = sorting_or_sorting_analyzer
        analyzer = None

    # Find available unit properties from all sources
    sorting_props = list(sorting.get_property_keys())
    if analyzer is not None:
        if analyzer.get_extension("quality_metrics") is not None:
            qm_props = list(analyzer.get_extension("quality_metrics").get_data().columns)
            qm_data = analyzer.get_extension("quality_metrics").get_data()
        else:
            qm_props = []
        if analyzer.get_extension("template_metrics") is not None:
            tm_props = list(analyzer.get_extension("template_metrics").get_data().columns)
            tm_data = analyzer.get_extension("template_metrics").get_data()
        else:
            tm_props = []
        # Check for any overlaps and warn user if any
        all_props = sorting_props + qm_props + tm_props
    else:
        all_props = sorting_props
        qm_props = []
        tm_props = []
        qm_data = None
        tm_data = None

    overlap_props = [prop for prop in all_props if all_props.count(prop) > 1]
    if len(overlap_props) > 0:
        warn(
            f"Warning: Overlapping properties found in sorting, quality_metrics, and template_metrics: {overlap_props}"
        )

    # Get unit properties
    if unit_properties is None:
        ut_columns = []
        ut_rows = [vv.UnitsTableRow(unit_id=u, values={}) for u in sorting.unit_ids]
    else:
        ut_columns = []
        ut_rows = []
        values = {}
        valid_unit_properties = []

        # Create columns for each property
        for prop_name in unit_properties:

            # Get property values from correct location
            if prop_name in sorting_props:
                property_values = sorting.get_property(prop_name)
            elif prop_name in qm_props:
                property_values = qm_data[prop_name].to_numpy()
            elif prop_name in tm_props:
                property_values = tm_data[prop_name].to_numpy()
            else:
                warn(f"Property '{prop_name}' not found in sorting, quality_metrics, or template_metrics")
                continue

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
                warn(f"Unsupported dtype {val0.dtype} for property {prop_name}. Skipping")
                continue
            ut_columns.append(vv.UnitsTableColumn(key=prop_name, label=prop_name, dtype=dtype))
            valid_unit_properties.append(prop_name)

        # Create rows for each unit
        for ui, unit in enumerate(sorting.unit_ids):
            for prop_name in valid_unit_properties:

                # Get property values from correct location
                if prop_name in sorting_props:
                    property_values = sorting.get_property(prop_name)
                elif prop_name in qm_props:
                    property_values = qm_data[prop_name].to_numpy()
                elif prop_name in tm_props:
                    property_values = tm_data[prop_name].to_numpy()

                # Check for NaN values and round floats
                val0 = np.array(property_values[0])
                if val0.dtype.kind == "f":
                    if np.isnan(property_values[ui]):
                        continue
                    property_values[ui] = np.format_float_positional(property_values[ui], precision=4, fractional=False)
                values[prop_name] = property_values[ui]
            ut_rows.append(vv.UnitsTableRow(unit_id=unit, values=check_json(values)))

    v_units_table = vv.UnitsTable(rows=ut_rows, columns=ut_columns, similarity_scores=similarity_scores)
    return v_units_table
