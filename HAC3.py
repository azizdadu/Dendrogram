import pandas as pd
from sklearn.cluster import AgglomerativeClustering	
from scipy.cluster.hierarchy import linkage
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


from os import listdir
from os.path import isfile, join
import os, sys


# Init
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Load data
from sklearn.datasets import load_diabetes

# Clustering
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list
from scipy.spatial import distance
#from fastcluster import linkage # You can use SciPy one too


mypath = "C:\\Users\\admin\\Dropbox\\Thesis_Start\\Thesis\\After_Defence\\gensim_fixed\\Datasets\\Doc50\\"
docLabels = [f for f in listdir(mypath) if isfile(join(mypath, f))]

df = pd.read_csv("Doc50_TFIDF_Vecorizer_cos_similarity_matrix_Doc50-copy.csv")
data_matrix = df.as_matrix()

DF_diabetes = pd.DataFrame(data_matrix, columns = [f for f in docLabels])
#print(DF_diabetes)

DF_dism = 1 - np.abs(DF_diabetes.corr())
#print(DF_dism)

'''
def plot_dendrogram(model, **kwargs):
	# Children of hierarchical clustering
	children = model.children_       
	#print (children)
	# Distances between each pair of children
	# Since we don't have this information, we can use a uniform one for plotting
	distance = np.arange(children.shape[0])
	#print(distance)
    # The number of observations contained in each cluster level
	no_of_observations = np.arange(2, children.shape[0]+2)
	#print(no_of_observations)
    # Create linkage matrix and then plot the dendrogram
	linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
	#print(linkage_matrix)
    # Plot the corresponding dendrogram
	dendrogram(linkage_matrix, **kwargs)

'''
model = AgglomerativeClustering(affinity='precomputed', n_clusters=5, linkage='complete').fit(DF_dism)
#model = model.fit(DF_dism)

plt.title('Hierarchical Clustering Dendrogram')

dendrogram(data_link,labels=data.dtype.names)
#dendogram = dendrogram(model, model.labels_)
plt.show()

#plot_dendrogram(model, labels=model.labels_)


#plt.show()




'''
A_dist = distance.squareform(DF_dism.as_matrix())

#print(A_dist)


Z = linkage(A_dist,method="complete")

D = dendrogram(Z=Z, labels=DF_dism.index, color_threshold=0.7, leaf_font_size=12, leaf_rotation=45)
#plt.show()


'''


'''
mypath = "C:\\Users\\admin\\Dropbox\\Thesis_Start\\Thesis\\After_Defence\\gensim_fixed\\Datasets\\Doc50\\"
docLabels = [f for f in listdir(mypath) if isfile(join(mypath, f))]

df = pd.read_csv("Doc50_TFIDF_Vecorizer_cos_similarity_matrix_Doc50-copy.csv")
data_matrix = df.as_matrix()


data_dist = pdist(data_matrix)

linkage_matrix = linkage(data_dist, 'complete')


plt.figure(101)
plt.subplot(1, 2, 1)
plt.title("ascending")
dendrogram(linkage_matrix,
           color_threshold=1,
           truncate_mode='lastp',
           labels=np.array([f for f in docLabels]),
           distance_sort='ascending')

		   
		   
		   
		   

plt.figure(102)
plt.title("five Clusters")

linkage_matrix = linkage(data_dist, 'complete')
print ("five clusters")
#print linkage_matrix

dendrogram(linkage_matrix,
           truncate_mode='lastp',
           color_threshold=5,
           show_leaf_counts=True)

plt.show()  '''
		 
'''

def augmented_dendrogram(model, **kwargs):

    ddata = dendrogram(model, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')

    return ddata
	
model = AgglomerativeClustering(affinity='precomputed', n_clusters=5, linkage='complete').fit(data_matrix)

plt.title('Hierarchical Clustering Dendrogram')


augmented_dendrogram(model, labels=model.labels_)
plt.show()'''