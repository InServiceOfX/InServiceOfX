from MachineLearning.Supervised.KNN import (euclidean_distance, KNN)

import numpy as np
import pytest

def generate_training_example_1():
	x1 = np.array([1.0, 1.0], dtype=np.float64)
	x2 = np.array([-2.0, 0.5], dtype=np.float64)
	x3 = np.array([-3.0, -4.0], dtype=np.float64)
	x4 = np.array([5.0, -1.25], dtype=np.float64)
	y = np.array([1, 2, 3, 4], dtype=np.int64)
	return np.array([x1, x2, x3, x4]), y

def test_euclidean_distance():
	# https://numpy.org/doc/stable/reference/generated/numpy.array.html
	# numpy.array(..) is a function that creates a ndarray.
	x = np.array([5.25, 2.5, 3., 1.125], dtype=np.float64)
	y = np.array([6.125, 3., 4.75, 1.5], dtype=np.float64)
	assert euclidean_distance(x, y) == pytest.approx(2.053959590644373)

	x = np.array([2**52, -2**52], dtype=np.float64)
	y = np.array([0.0, 0.0], dtype=np.float64)

	assert euclidean_distance(x, y) == pytest.approx(np.sqrt(2) * 2**52)

	x = np.array([1**300, -1**300], dtype=np.float64)
	assert euclidean_distance(x, y) == pytest.approx(np.sqrt(2) * 1**300)

def test_KNN__predict():
	X_train, y_train = generate_training_example_1()
	knn = KNN(1)
	knn.fit(X_train, y_train)

	# Test the k=1 nearest neighbor as testing out which quadrant would a test
	# point be considered to be, i.e. imagine the 4 quadrants of a graph and
	# classifying points by which quadrant it would be nearest to, but given the
	# caveat that we're considering proximity to a point in each quadrant.
	assert knn._predict(np.array([2.0, 2.0])) == 1
	assert knn._predict(np.array([-19.0, 2.0])) == 2
	assert knn._predict(np.array([-22.0, -42.0])) == 3
	assert knn._predict(np.array([422.0, -420.0])) == 4
