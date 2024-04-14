import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.colors import Normalize

# sklearn package for machine learning in Python:
from sklearn.cluster import KMeans

# read data (make sure .csv in folder)
df = pd.read_csv("country_data.csv")

print(df.head(), '\n')

# select the columns
X = df.iloc[:, [4,2,5]].values

# contruct the model (either k-means or mean shift)
model = KMeans(n_clusters = 3, n_init='auto', random_state=2)

#model = MeanShift()
model.fit(X)
cluster_centers = model.cluster_centers_

# print the centre positions of the clusters
centers = model.cluster_centers_

print('Centroids:', centers, '\n')

#Visualise the result in a 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection = '3d')

# store the normalisation of the color encodings based on the number of clusters
nm = Normalize(vmin = 0, vmax = len(centers)-1)

# plot the clustered data
scatter1 = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = model.predict(X), s = 50, cmap = 'plasma', norm = nm)

# plot the centroids using a for loop
for i in range(centers.shape[0]): ax.text(centers[i, 0], centers[i, 1], centers[i, 2],
str(i), c = 'black', bbox=dict(boxstyle="round", facecolor='white', edgecolor='black'))

ax.azim = -60
ax.dist = 11
ax.elev = 10

# make sure you choose the correct column names here!!!
ax.set_xlabel(df.columns[4])
ax.set_ylabel(df.columns[2])
ax.set_zlabel(df.columns[5])

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter1.legend_elements(), loc="center left", title="Clusters")
ax.add_artist(legend1) 

fig.tight_layout(pad=-2.0)
fig.savefig('Cluster_3Dplot.png')