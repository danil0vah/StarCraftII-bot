import numpy as np
import CONSTANTS as CT
import os

train_collection = os.listdir(CT.train_data_path)
for file in train_collection:
    full_path = os.path.join(CT.train_data_path, file)
    data = np.load(full_path, allow_pickle=True)
print(data)
