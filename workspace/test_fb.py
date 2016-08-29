from knowledge import FBManager
from tripletembed.model import TripletEmbedding
import numpy as np
import pandas as pd

fbm = FBManager("data/FB15k/freebase_mtr100_mte100-train.txt", "data/FB15k/freebase_mtr100_mte100-test.txt")

s_train, o_train, p_train = fbm.get_train_data()

n_entity = fbm.n_inst
n_relation = fbm.n_prop

model = TripletEmbedding(n_entity, n_relation, 30, 15, [3], sigma=10e-6, learning_rate=10e-3,
                         w_regularization=10e-2)

model.fit(s_train, o_train, p_train)
