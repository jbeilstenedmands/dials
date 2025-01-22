from __future__ import annotations

import logging

from dxtbx.model.experiment_list import ExperimentList

from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.indexer import Indexer
from dials.algorithms.indexing.known_orientation import IndexerKnownOrientation
from dials.algorithms.indexing.lattice_search import BasisVectorSearch, LatticeSearch
from dials.array_family import flex
from dials.util.multi_dataset_handling import generate_experiment_identifiers

logger = logging.getLogger(__name__)


class SSXIndexer(Indexer):
    "Keep it very simple"

    def index(self):
        logger.info("Using SSX indexer")
        experiments = ExperimentList()
        # max_lattices = self.params.multiple_lattice_search.max_lattices
        max_lattices = 1
        while True:
            if max_lattices is not None and len(experiments.crystals()) >= max_lattices:
                break

            n_lattices_previous_cycle = len(experiments)

            # index multiple lattices per shot
            if len(experiments) == 0:
                new = self.find_lattices()
                generate_experiment_identifiers(new)
                experiments.extend(new)
                if len(experiments) == 0:
                    raise DialsIndexError("No suitable lattice could be found.")
            else:
                try:
                    new = self.find_lattices()
                    generate_experiment_identifiers(new)
                    experiments.extend(new)
                except Exception as e:
                    logger.info("Indexing remaining reflections failed")
                    logger.debug(
                        "Indexing remaining reflections failed, exception:\n" + str(e)
                    )

            # reset reflection lattice flags
            # the lattice a given reflection belongs to: a value of -1 indicates
            # that a reflection doesn't belong to any lattice so far
            self.reflections["id"] = flex.int(len(self.reflections), -1)

            self.index_reflections(experiments, self.reflections)

            if len(experiments) == n_lattices_previous_cycle:
                # no more lattices found
                break

            if self.params.known_symmetry.space_group is not None:
                self._apply_symmetry_post_indexing(
                    experiments, self.reflections, n_lattices_previous_cycle
                )

            logger.info("\nIndexed crystal models:")
            self.show_experiments(experiments, self.reflections, d_min=self.d_min)

            self.indexed_reflections = self.reflections["id"] > -1
            if self.d_min is None:
                sel = self.reflections["id"] <= -1
            else:
                sel = flex.bool(len(self.reflections), False)
                lengths = 1 / self.reflections["rlp"].norms()
                isel = (lengths >= self.d_min).iselection()
                sel.set_selected(isel, True)
                sel.set_selected(self.reflections["id"] > -1, False)

            unindexed_reflections = self.reflections.select(sel)
            reflections_for_refinement = self.reflections.select(
                self.indexed_reflections
            )

            self.refined_reflections = reflections_for_refinement
            # id can be -1 if some were determined as outliers in self.refine
            sel = self.refined_reflections["id"] < 0
            if sel.count(True):
                self.refined_reflections.unset_flags(
                    sel,
                    self.refined_reflections.flags.indexed,
                )
                self.refined_reflections["miller_index"].set_selected(sel, (0, 0, 0))
                unindexed_reflections.extend(self.refined_reflections.select(sel))
                self.refined_reflections = self.refined_reflections.select(~sel)
            self.refined_reflections.clean_experiment_identifiers_map()
            self.unindexed_reflections = unindexed_reflections
            self.refined_experiments = experiments

            from dials.algorithms.refinement.prediction.managed_predictors import (
                ExperimentsPredictorFactory,
            )

            ref_predictor = ExperimentsPredictorFactory.from_experiments(
                experiments,
                force_stills=True,
                spherical_relp=self.all_params.refinement.parameterisation.spherical_relp_model,
            )
            ref_predictor(self.refined_reflections)
            self.refined_reflections["delpsical2"] = (
                self.refined_reflections["delpsical.rad"] ** 2
            )

            if "xyzcal.mm" in self.refined_reflections:
                self._xyzcal_mm_to_px(
                    self.refined_experiments, self.refined_reflections
                )


class SSXIndexerKnownOrientation(IndexerKnownOrientation, SSXIndexer):
    pass


class SSXIndexerBasisVectorSearch(SSXIndexer, BasisVectorSearch):
    pass


class SSXIndexerLatticeSearch(SSXIndexer, LatticeSearch):
    pass
