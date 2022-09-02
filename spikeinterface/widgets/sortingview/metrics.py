import numpy as np
from ..base import to_attr
from .base_sortingview import SortingviewPlotter, generate_unit_table_view


class MetricsPlotter(SortingviewPlotter):

    def do_plot(self, data_plot, **backend_kwargs):
        import sortingview.views as vv

        dp = to_attr(data_plot)
        backend_kwargs = self.update_backend_kwargs(**backend_kwargs)

        metrics = dp.metrics

        if dp.unit_ids is None:
            unit_ids = metrics.index.values
        else:
            unit_ids = dp.uniit_ids
        unit_ids = self.make_serializable(unit_ids)

        metrics_sv = []
        for col in metrics.columns:
            dtype = metrics.iloc[0][col].dtype
            
            metric = vv.UnitMetricsGraphMetric(
                            key=col,
                            label=col,
                            dtype=dtype.str
                        )
            metrics_sv.append(metric)

        units_m = []
        for unit_id in unit_ids:
            values = metrics.loc[unit_id].to_dict()
            # make sure values are json serializable
            values_ser = {}
            for key, val in values.items():
                # skip nans
                if np.isnan(val):
                    continue
                dtype = type(val)
                if np.dtype(dtype) == np.int64:
                    values_ser[key] = int(val)
                elif np.dtype(dtype) == np.float64:
                    values_ser[key] = float(val)
                else:
                    values_ser[key] = val
                    
            units_m.append(
                vv.UnitMetricsGraphUnit(
                    unit_id=unit_id,
                    values=values_ser
                )
            )
        v_metrics = vv.UnitMetricsGraph(
                units=units_m,
                metrics=metrics_sv
            )

        if not dp.hide_unit_selector:
            v_units_table = generate_unit_table_view(unit_ids)

            view = vv.Box(
                direction="horizontal",
                items=[
                    vv.LayoutItem(v_units_table, max_size=150),
                    vv.LayoutItem(v_metrics),
                ],
            )
        else:
            view = v_metrics

        self.set_view(view)

        if backend_kwargs["generate_url"]:
            if backend_kwargs.get("figlabel") is None:
                label = "SpikeInterface - Metrics"
            url = view.url(label=label)
            print(url)
        return view
