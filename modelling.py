import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import typeconv as tp

heart_failure_dataset = pd.read_csv("heart.csv")
dataset_after_conv_cat_val=tp.typeconvo(heart_failure_dataset)

x = dataset_after_conv_cat_val.drop('HeartDisease', axis = 1)
y = dataset_after_conv_cat_val['HeartDisease']

knn = KNeighborsClassifier(n_neighbors=2,metric='euclidean',p=2)
knn.fit(x, y)

pickle.dump(knn, open('predic_model.pkl', 'wb')) 
