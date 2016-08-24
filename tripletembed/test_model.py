from model import TripletEmbedding
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

te = TripletEmbedding(100, 20, 5, 3, [4], learning_rate=10e-2, sigma=10e-4)

subjects = np.random.randint(0, 100, size=1000)
objects = np.random.randint(0, 100, size=1000)
predicates = np.random.randint(0, 20, size=1000)
labels = 2 * (-0.5 + np.random.randint(0, 2, size=1000))


te.fit(subjects, objects, predicates, labels)

print(te.get_entity_embedding([1,2,3,4,5,6]))
print(te.get_relation_embedding([1,2,3]))

print(te.get_estimated_embedding([1, 3, 5], [2, 4, 6]))
