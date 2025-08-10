import numpy as np
from sklearnex import patch_sklearn, config_context
patch_sklearn()

from sklearn.cluster import DBSCAN

X = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
with config_context(target_offload="gpu:0"):
   clustering = DBSCAN(eps=3, min_samples=2).fit(X)