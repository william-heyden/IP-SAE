import numpy as np
import scipy
from scipy import spatial
import load_data as ld
from sae_helper import SAE, acc_gzsl, NormalizeFea
from sklearn.model_selection import train_test_split

# Load data
X_tr, X_te, S_tr, S_te, Y_te, test_cls, Y_tr, S_te_all, att = ld.cub(int_proj=True)

# Generalized setting w/ 20% seen examples
X_train, X_test, y_train, y_test = train_test_split(X_tr, Y_tr, test_size = 0.2)
X_te =  np.concatenate([X_te , X_test], axis=0)
Y_te =  np.concatenate([Y_te , y_test], axis=0)

# Solve for W
lamb  = 500000;
W=SAE(X_tr.T,S_tr.T,lamb).T

# Projecting from semantic to visual space 
S_tr_gt_all=NormalizeFea(att.T,2).T
dist = 1 - spatial.distance.cdist(X_te,S_tr_gt_all.dot(W.T),'cosine')

harm, seen, unseen = acc_gzsl(dist, Y_te, Y_tr)
print('Accuracy harmonic:\t{:.3f} \nAccuracy seen:\t\t{:.3f}\nAccuracy unseen:\t{:.3f}'.format(harm*100, seen*100, unseen*100))
    