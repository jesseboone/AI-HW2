# Structure of the Project

## Part 1 - Supervised Learning

In this folder is the main file, "hw2-runMLP.py", which will load the model built by "hw2-buildMLP.py" on the training data from "mlpModel.sav", and run it on whatever data is read in from "x.csv" and "y.csv" in the same folder.  It then outputs the predicted values for that input to "z-predicted.csv". 
(hw2-buildMLP.py does not need to be run unless you want to build a new model from the x and y csv files to run on another set of data)

## Details of the model and rationale of its structure

The model is a multilayer perceptron with two layers, the first being 15 neurons and the second being 7.  With a set of 30 input samples for X and Y, this size net was small enough to be able to learn the pattern of the data relatively quickly, while hopefully avoiding overfitting from being too large.  This balance was struck by testing some nets of different sizes, and this seemed to be where the model complexity met good performance, without too much loss in accuracy as the testing size was gradually dialed up.  With this setup, the model is still able to achieve a .922 R^2 score even with a 50% split of training and test data.

This model also uses adam, which is a stochastic gradient-based optimizer.  While this was not my first choice (lbfgs was due to its strength on small datasets), I chose it instead so I could apply early stopping and shuffling to the training.  In addition to these parameters, I also applied inverse scaling to gradually decrease the learning rate, in hopes of further reducing the possibility of overfitting.


## Part 2 - Unsupervised Learning

Also in this folder is the file, "hw2-2.py", which pulls in data from "p2-data.csv" and builds a Gaussian Mixture Model distribution over the data.  I have included (but commented out) the code to plot the graphs and ellipsoids built by this model as well as a competing model (the Dirichlet Process Model, which has the added ability to adjust the components it tries to fit to the data automatically, unlike being locked in on the input as the GMM is).  However, from using the competing approaches, I was able to confidently determine there should be 5 clusters in this data, as this is where they both performed very similarly (and also where the Dirichlet converged even with 6+ components allowed).

The output of this program is the "part2-output.txt" file, which contains the 5 clusters means and covariances, followed by the cluster assignment for each data point in the original "p2-data" file.

### Sources used:

__Part 1__
Scikitlearn Neural Networks (Multi-layer Perceptron):
https://scikit-learn.org/stable/modules/neural_networks_supervised.html
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor

train_test_split and r2_score (to determine accuracy/fit of model to data):
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

Pickle (to save and load the model):
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

__Part 2__
Scikitlearn Gaussian Mixture Models:
https://scikit-learn.org/stable/modules/mixture.html
https://scikit-learn.org/0.16/modules/generated/sklearn.mixture.GMM.html
https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html


