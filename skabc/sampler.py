import numbers
import numpy as np
from sklearn.neighbors._base import KNeighborsMixin, NeighborsBase
from sklearn.utils import check_array
from sklearn.utils._param_validation import Interval


class NearestNeighborSampler(KNeighborsMixin, NeighborsBase):
    """
    Approximate posterior sampler based on k-nearest neighbors.

    Args:
        n_neighbors: Number of neighbors to sample from the reference table.
        frac_neighbors: Fraction of neighbors to sample from the reference table.
        algorithm: Algorithm used to compute the nearest neighbors.

            - :code:`ball_tree` will use :class:`~sklearn.neighbors.BallTree`
            - :code:`kd_tree` will use :class:`~sklearn.neighbors.KDTree`
            - :code:`brute` will use a brute-force search.
            - :code:`auto` will attempt to decide the most appropriate algorithm based
              on the values passed to :meth:`fit` method.
        leaf_size: Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
            :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
            construction and query, as well as the memory required to store the tree.
            The optimal value depends on the nature of the problem.
        p: Power parameter for the Minkowski metric. When :code:`p = 1`, this is
            equivalent to using :code:`manhattan_distance` (:math:`l_1`), and
            :code:`euclidean_distance` (:math:`l_2`) for :code:`p = 2`. For arbitrary
            :code:`p`, :code:`minkowski_distances` (:math:`l_p`) is used.
        metric: Metric to use for distance computation. Default is :code:`minkowski`,
            which results in the standard Euclidean distance when :code:`p = 2`. See the
            documentation of :mod:`scipy.spatial.distance` and the metrics listed in
            :func:`~sklearn.metrics.pairwise.distance_metrics` for valid metric values.

            If metric is :code:`precomputed`, :code:`X` is assumed to be a distance
            matrix and must be square during :meth:`fit`. :code:`X` may be a sparse
            graph, in which case only "nonzero" elements may be considered neighbors.

            If :code:`metric` is a callable function, it takes two arrays representing
            1D vectors as inputs and must return one value indicating the distance
            between those vectors. This works for Scipy's metrics, but is less
            efficient than passing the metric name as a string.

            If metric is a :class:`~sklearn.metrics.DistanceMetric` object, it will be
            passed directly to the underlying computation routines.
        metric_params: Additional keyword arguments for the :code:`metric` function.
        n_jobs: Number of parallel jobs to run for neighbors search. :code:`None`
            means 1 unless in a :code:`joblib.parallel_backend` context. :code:`-1`
            means using all processors. Doesn't affect :meth:`fit` method.

    Attributes:
        effective_metric_: Distance metric to use. It will be same as the :code:`metric`
            parameter or a synonym of it, e.g. :code:`euclidean` if the :code:`metric`
            parameter is set to :code:`minkowski` and :code:`p = 2`.
        effective_metric_params_: Additional keyword arguments for the metric function.
            For most metrics will be same with :code:`metric_params` parameter, but may
            also contain the :code:`p` parameter value if the :code:`effective_metric_`
            attribute is set to :code:`minkowski`.
        n_features_in_: Number of features seen during :meth:`fit`.
        feature_names_in_: Names of features seen during :meth:`fit`. Defined only when
            :code:`X` has feature names that are all strings.
        n_samples_fit_: Number of samples in the fitted data.
    """

    _parameter_constraints = {
        **NeighborsBase._parameter_constraints,
        "frac_neighbors": [Interval(numbers.Real, 0, 1, closed="neither"), None],
    }

    def __init__(
        self,
        *,
        n_neighbors: None | int = None,
        frac_neighbors: None | float = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: None | int = 2,
        metric_params=None,
        n_jobs: None | int = None,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.frac_neighbors = frac_neighbors

    def _more_tags(self):
        return {
            # Required to store the parameter samples. Cf. RegressorMixin.
            "requires_y": True,
            # Required because each sample should have a parameter vector.
            "multioutput_only": True,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestNeighborSampler":
        """
        Fit the k-nearest neighbors sampler to the reference table.

        Args:
            X: Summary statistics of the reference table with shape
                (n_train, n_summaries).
            y: Parameter values of the reference table with shape (n_train, n_params).

        Returns:
            The :class:`NearestNeighborSampler`.
        """
        if (self.n_neighbors is None) == (self.frac_neighbors is None):
            raise ValueError(
                "Exactly one of `n_neighbors` or `frac_neighbors` must be given."
            )
        check_array(y)
        return self._fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Draw approximate posterior samples.

        Args:
            X: Summary statistics for the target dataset with shape
                (n_queries, n_summaries).

        Returns:
            Posterior samples with shape (n_queries, n_neighbors, n_params).
        """
        n_neighbors = self.n_neighbors or int(self.frac_neighbors * self.n_samples_fit_)
        _, neigh_ind = self.kneighbors(X, n_neighbors)
        return self._y[neigh_ind]
