import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from matplotlib import cm

from scipy.spatial import distance

from pickle import load


def load_HDR_data():
    """Load the data for the HDR dataset.
    
    Returns
    -------
        - data: dictionary with fields X, country_names,
                indicator_names and indicator_descriptions
                
    """
    
    fd = open('hdr_data.dat', 'rb')

    hdr_data = load(fd)
    fd.close()
    
    return hdr_data
    

def RGB_color(color_id, nb_colors, cmap='jet'):
    """ Returns a RGB code for a given number of possible colors.
    
    Parameters
    ----------
        - color_id: id of the color (0 <= color_id < nb_colors)
        - nb_colors: total number of possible colors
        - cmap: name of color map (string, optional)
        
    Returns
    -------
        - rgb_code: a valid RGB code, i.e. a 3-uple of values in [0, 1]
    
    For a list of color maps, see http://matplotlib.org/examples/color/colormaps_reference.html
    
    """
    
    cmap = cm.get_cmap('jet')

    return cmap(float(color_id) / nb_colors)
    

def show_annotated_clustering(X, clustering, labels):
    """Displays a clustering where each instance is labelled.
    
    Parameters
    ----------
        - X: 2D coordinates of objects
        - clustering: assignment to clusters
        - labels: text labels of objects
    
    clustering is expected to contain cluster ids between 0 and
    nb_clusters - 1.  Each cluster must contain a least one object.
    
    """
    
    nb_clusters = np.max(clustering)+1
    n, d = X.shape
    
    for i in range(n):
        plt.text(X[i, 0], X[i, 1], labels[i], fontsize=3, bbox=dict(facecolor=RGB_color(clustering[i], nb_clusters), alpha=0.5))
    
    plt.xlim(np.min(X[:, 0])-1, np.max(X[:, 0])+1)
    plt.ylim(np.min(X[:, 1])-1, np.max(X[:, 1])+1)


def find_closest_instances_to_kmeans(X, model):
    """Returns the set of instances which are the closest to the cluster centers in a k-means.
     
     Parameters
     ----------
        - X: instances (which were used to train the k-means model for clustering)
        - model: k-means model for clustering (produced by sklearn.cluster.kmeans)

    Returns
    -------
        - closest_instances: set of closest instances to the cluster centers
        - closest_indices: indice of closest instances in X
        
    """
    
    centroids = model.cluster_centers_
    nb_clusters, d = centroids.shape
    
    closest_instances = np.zeros((nb_clusters, d), dtype=float)
    closest_indices = np.zeros((nb_clusters, ), dtype=int)
    for cluster_id in range(nb_clusters):
        closest_indices[cluster_id] = np.argmin(sp.spatial.distance.cdist(X, centroids[cluster_id, :].reshape((1, -1))))
        closest_instances[cluster_id, :] = X[closest_indices[cluster_id], :]
        
    return closest_instances, closest_indices
