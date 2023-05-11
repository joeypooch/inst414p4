import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform

# API Key for News API
api_key = "1879c06fe71c467c8b39cb6b7255af19"

# Define API endpoint and parameters
url = ('http://newsapi.org/v2/everything?'
       'q=ukraine&'
       'sortBy=popularity&'
       'pageSize=100&'
       f'apiKey={api_key}')

# Send request and get response
response = requests.get(url)
data = response.json()

# Save data to file
with open('ukraine_articles.json', 'w') as f:
    json.dump(data, f)

# Load data from file
with open('ukraine_articles.json', 'r') as f:
    data = json.load(f)

# Extract article titles and descriptions
articles = [(article['title'] + '. ' + article['description']) for article in data['articles']]

# Vectorize articles using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(articles)

# Reduce dimensionality with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Determine optimal number of clusters using elbow method
distortions = []
K = range(1, 10)

for k in K:
    clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=42).fit(X)
    labels = clustering.labels_
    distortion = 0
    for i in range(k):
        cluster_distortion = pdist(X.toarray()[labels == i, :], metric='euclidean').sum() / X.toarray()[labels == i, :].shape[0]
        distortion += cluster_distortion
    distortions.append(distortion / k)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Distance')
plt.title('Elbow Method for Optimal k')
plt.show()


# Cluster articles using Spectral Clustering
clustering = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=42).fit(X)

# Add cluster labels to PCA data
df = pd.DataFrame(X_pca, columns=['x', 'y'])
df['label'] = clustering.labels_

# Generate network graph of articles and their similarity
dist = pdist(X.toarray(), metric='cosine')
dist = 1 - squareform(dist)
G = nx.from_numpy_array(dist)

# Add article titles as node attributes
mapping = {}
for i, article in enumerate(data['articles']):
    mapping[i] = article['title']
nx.relabel_nodes(G, mapping, copy=False)

# Add cluster labels as node attributes
for node in G.nodes:
    G.nodes[node]['cluster'] = clustering.labels_[i]

# Print clusters
for i in range(clustering.n_clusters):
    print(f'Cluster {i}:')
    for j, article in enumerate(data['articles']):
        if clustering.labels_[j] == i:
            print(f"{article['title']} - {article['source']['name']}")
    print()

# Add node size based on number of connections
sizes = [len(G.edges(node)) for node in G.nodes]
nx.set_node_attributes(G, dict(zip(G.nodes, sizes)), 'size')

# Plot network graph
pos = nx.spring_layout(G, k=0.25)
node_colors = clustering.labels_
cmap = plt.cm.get_cmap('viridis', len(set(node_colors)))
node_sizes = [len(G.edges(node)) for node in G.nodes]
nx.draw_networkx(G, pos, node_color=node_colors, cmap=cmap, node_size=node_sizes, with_labels=False)
plt.show()
