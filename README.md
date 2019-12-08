# ENS-VS
A method for virtual screening based on ensemble learning.
You can start with main.m
Ensemble.m - a number of decoy subsets with the same size as actives are sampled from the original decoys. Each subset of decoys and all of actives compose a subset for training sub-classifier. The final decision is made by combining the sub-classifiers by bagging. 
ThreeClassifier.m - Each subset is trained by three algorithm: SVM, decision tree and Fisher linear discriminant.
CalculatedAUC.m - Calculate the AUC for the validation set and test set.
myKmean.s- the k-means algorithm used for sampling the decoy subsets
pca_row.m - the pca algorithm for the dimension reduction.
