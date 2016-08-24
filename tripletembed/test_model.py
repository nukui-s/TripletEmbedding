from model import TripletEmbedding
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

te = TripletEmbedding(100, 20, 5, 3, [4], learning_rate=10e-2)

subjects = np.random.randint(0, 100, size=1000)
objects = np.random.randint(0, 100, size=1000)
predicates = np.random.randint(0, 20, size=1000)
labels = 2 * (-0.5 + np.random.randint(0, 1, size=1000))

te.fit(subjects, objects, predicates, labels)
