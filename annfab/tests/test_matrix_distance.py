
import numpy as np

import annfab.distances
import nearpy.distances


def test_init_matrix_distance():
	d = annfab.distances.MatrixCosineDistance()

	assert d is not None


def test_matrix_distance_for_vector_is_distance():
	n = 100
	x = np.random.randn(n)
	y = np.random.randn(n)

	vD = nearpy.distances.CosineDistance()
	vd = vD.distance(x, y)

	mD = annfab.distances.MatrixCosineDistance()
	md = mD.distance(x, y)






