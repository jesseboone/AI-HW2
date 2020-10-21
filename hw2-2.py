# Needed for main program functionality (gmm model)
import numpy as np
from sklearn import mixture
import sys

# Needed for plot functionality
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

# split a string into tokens based on a given character
def parse (str1, char) :
    l = str1.split(char)
    return l

data = []
tmp = []

# Reads in the x and y values from file
f = open('p2-data.csv', 'r')
for line in f.readlines():
	line = parse(line, ',')
	tmp = [float(line[0])]
	tmp.append(float(line[1]))
	data.append(tmp)
f.close()

# turns array into numpy array
data = np.array(data)
size = len(data)

# colors for plot ellipsoids
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

# this plots the results of the model by drawing ellipsoids over the first probability 
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-15., 18.)
    plt.ylim(-15., 18.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(data)

# Uncommment next line to plot models predictions over the data
# plot_results(data, gmm.predict(data), gmm.means_, gmm.covariances_, 0,
#              'Gaussian Mixture')


# writing output to a file
f = open('part2-output.txt', 'w')

# Output the means and covariances of each cluster
for i, (mean, covar) in enumerate(zip(gmm.means_,gmm.covariances_)):
	f.write("Cluster " + str(i) + "\nMean:\n" + str(mean) + "\nCovariance:\n" + str(covar) + "\n\n")

# Output the cluster assignment for each data point
f.write("\nCluster \tData Point\n\n")
predictions = gmm.predict(data)
for i, (p,d) in enumerate(zip(predictions, data)):
	f.write(str(p) + "\t" + str(d) + "\n")
f.close()


# # prints output to terminal instead
# print(gmm.means_)
# print(gmm.covariances_)
# print(gmm.predict(data))

# # Fit a Dirichlet process Gaussian mixture using five components
# dpgmm = mixture.BayesianGaussianMixture(n_components=5,
#                                         covariance_type='full').fit(data)
# plot_results(data, dpgmm.predict(data), dpgmm.means_, dpgmm.covariances_, 1,
#              'Bayesian Gaussian Mixture with a Dirichlet process prior')

# plt.show()