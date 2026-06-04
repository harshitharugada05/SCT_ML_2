import pandas as pd
from sklearn.cluster import KMeans
import pickle

data = pd.read_csv("Mall_Customers.csv")

X = data.iloc[:, [3, 4]].values

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

kmeans.fit(X)

pickle.dump(kmeans, open("customer_segmentation.pkl", "wb"))

print("Model Saved Successfully")