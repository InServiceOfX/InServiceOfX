"""
https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch/blob/main/01%20KNN/KNN.py

https://youtu.be/rTEtEy5o3X0?si=SEtNb0QfVh9LXlga
"""

import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
	"""
	x1, x2 are expected to be single dimensional arrays, possibly of type
	numpy.ndarray, and that subtraction (-) and powers of 2 (**2) are
	element-wise operations.
	"""
	distance = np.sqrt(np.sum((x1 - x2)**2))
	return distance

class KNN:
	def __init__(self, k=3):
		self.k = k

	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

	def predict(self, X):
		predictions = [self._predict(x) for x in X]
		return predictions

	def _predict(self, x):
		# Compute the distance
		distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

		# get the closest k
		k_indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[i] for i in k_indices]

		# majority vote
		most_common = Counter(k_nearest_labels).most_common()
		return most_common[0][0]