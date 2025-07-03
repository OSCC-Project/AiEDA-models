"""
Author: juanyu 291701755@qq.com
Description: train DNN model
"""

from sklearn.preprocessing import StandardScaler
from utils.sample_classification import *

design_id_list = ["9t1", "9t2", "9t3", "9t6", "9t7", "9t8", "9t9", "9t10"]

DRV_away, DRV_located, DRV_neighbor = load_data(design_id_list)

# Set the DRV-away samples random extraction ratio
extraction_ratio = 0.1
extraction_num = int(DRV_away.shape[0] * 0.1)

# Randomly select DRV-away samples
random_index = np.arange(DRV_away.shape[0])
np.random.seed(1)
np.random.shuffle(random_index)
extraction_DRV_away = DRV_away[random_index[0:extraction_num]]

# Merge all samples and shuffle the samples
all_data = extraction_DRV_away
all_data = np.concatenate((all_data, DRV_located), axis=0)
all_data = np.concatenate((all_data, DRV_neighbor), axis=0)
np.random.seed(1)
np.random.shuffle(all_data)

# StandardScaler transforms features into a standard normal distribution
scaler = StandardScaler()
scaler.fit(all_data[:, 0:23])
all_data[:, 0:23] = scaler.transform(all_data[:,0:23])

np.save("../data/ispd2019/all_data_" + str(extraction_ratio) + ".npy", all_data)
