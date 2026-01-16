"""
Observers for the cosym procedure.
"""

from __future__ import annotations

import json

from jinja2 import ChoiceLoader, Environment, PackageLoader

from dials.algorithms.clustering.observers import UnitCellAnalysisObserver
from dials.algorithms.symmetry.cosym import SymmetryAnalysis
from dials.algorithms.symmetry.cosym.plots import plot_coords, plot_rij_histogram
from dials.util.observer import Observer, singleton


def register_default_cosym_observers(script):
    """Register the standard observers to the cosym script."""
    script.cosym_analysis.register_observer(
        event="analysed_symmetry", observer=SymmetryAnalysisObserver()
    )
    script.cosym_analysis.register_observer(
        event="optimised", observer=CosymClusterAnalysisObserver()
    )
    script.register_observer(event="run_cosym", observer=UnitCellAnalysisObserver())
    script.register_observer(
        event="run_cosym", observer=CosymHTMLGenerator(), callback="make_html"
    )
    script.register_observer(
        event="run_cosym", observer=CosymJSONGenerator(), callback="make_json"
    )


@singleton
class CosymHTMLGenerator(Observer):
    """
    Observer to make a html report
    """

    def make_html(self, cosym_script):
        """Collect data from the individual observers and write the html."""
        filename = cosym_script.params.output.html
        if not filename:
            return
        self.data.update(CosymClusterAnalysisObserver().make_plots())
        self.data.update(UnitCellAnalysisObserver().make_plots())
        self.data.update(SymmetryAnalysisObserver().make_tables())
        axes_labels = {
            str(i): f"PC {i + 1} ({var:.1f}%)"
            for i, var in enumerate(cosym_script.cosym_analysis.explained_variance_ratio * 100)
        }
        import numpy as np
        cob_mat = np.linalg.inv(np.transpose(cosym_script.cosym_analysis.pca_components))
        
        from dials.algorithms.correlation.plots import plot_pca_coords
        coords = np.array([])
        for data in cosym_script.cosym_analysis.coords:
            rot_coord = np.matmul(cob_mat, data)
            if coords.size == 0:
                coords = rot_coord
            else:
                coords = np.vstack([coords, rot_coord])
        if coords.shape[1] > 6:
            coords = coords[:, 0:6]
            dim_list = list(range(0, 6))
        else:
            dim_list = list(range(0, cosym_script.cosym_analysis.target.dim))

        from dials.algorithms.correlation.cluster import ClusterInfo
        cluster_labels = [0]*len(coords)
        clusters = [ClusterInfo(0,cluster_labels, 1, 1, (1,1,1,90,90,90), 1)]
        
        pca_plot = plot_pca_coords(
            coords,
            axes_labels,
            cluster_labels,
            dim_list,
            clusters,
            [0],
        )
        print(f"Writing html report to: {filename}")
        loader = ChoiceLoader(
            [
                PackageLoader("dials", "templates"),
                PackageLoader("dials", "static", encoding="utf-8"),
            ]
        )
        env = Environment(loader=loader)
        template = env.get_template("cosym_report.html")
        html = template.render(
            page_title="DIALS cosym report",
            cosym_graphs=self.data["cosym_graphs"],
            unit_cell_graphs=self.data["unit_cell_graphs"],
            symmetry_analysis=self.data["symmetry_analysis"],
            pca_plot=pca_plot,
        )
        with open(filename, "wb") as f:
            f.write(html.encode("utf-8", "xmlcharrefreplace"))


@singleton
class CosymJSONGenerator(Observer):
    """
    Observer to make a html report
    """

    def make_json(self, cosym_script):
        """Collect data from the individual observers and write the html."""
        filename = cosym_script.params.output.json
        if not filename:
            return
        self.data.update(CosymClusterAnalysisObserver().make_plots())
        self.data.update(UnitCellAnalysisObserver().make_plots())
        self.data.update(SymmetryAnalysisObserver().get_data())
        print(f"Writing json to: {filename}")
        with open(filename, "w") as f:
            json.dump(self.data, f)


@singleton
class CosymClusterAnalysisObserver(Observer):
    """
    Observer to record cosym cluster analysis data and make model plots.
    """

    def update(self, cosym):
        """Update the data in the observer."""
        self.data["coordinates"] = cosym.coords
        self.data["rij_matrix"] = cosym.target.rij_matrix

    def make_plots(self):
        """Generate cosym cluster analysis plot data."""
        d = plot_rij_histogram(self.data["rij_matrix"])
        d.update(plot_coords(self.data["coordinates"]))
        graphs = {"cosym_graphs": d}
        return graphs


@singleton
class SymmetryAnalysisObserver(Observer):
    """
    Observer to record symmetry analysis data and make tables.
    """

    def update(self, cosym):
        if cosym._symmetry_analysis is not None:
            self.data.update(cosym._symmetry_analysis.as_dict())

    def make_tables(self):
        """Generate symmetry analysis tables."""
        d = {"symmetry_analysis": {}}
        if self.data:
            d["symmetry_analysis"].update(
                {
                    "sym_ops_table": SymmetryAnalysis.sym_ops_table(self.data),
                    "subgroups_table": SymmetryAnalysis.subgroups_table(self.data),
                    "summary_table": SymmetryAnalysis.summary_table(self.data),
                }
            )
        return d

    def get_data(self):
        return self.data
