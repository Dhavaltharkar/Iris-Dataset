import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pickle  

import warnings; warnings.filterwarnings('ignore')

# Load the Dataset
data = pd.read_csv("https://raw.githubusercontent.com/Dhavaltharkar/TSF_Data_Science_and_Business_Analytics/main/TSF_GRIP_Prediction_Using_Unsupervised_ML/Iris.csv")

x = data.iloc[:, [0, 1, 2, 3]].values

# K-means clustering
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Applying KMeans with optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
iris_k_model = KMeans(n_clusters=3)
iris_k_model.fit(x)

# Save the KMeans model as a pickle file
with open('kmeans.pkl', 'wb') as f:
    pickle.dump(iris_k_model, f)
