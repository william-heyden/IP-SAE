import numpy as np
import scipy
from scipy import spatial
import load_data as ld
from sae_helper import SAE, acc_zsl, NormalizeFae


# Load data
X_tr, X_te, S_tr, S_te, Y_te, test_cls, Y_tr, S_te_all, _ = ld.cub(int_proj=True)

# Solve for W
lamb  = 500000;
W=SAE(X_tr.T,S_tr.T,lamb).T

# Projecting from semantic to visual space 
S_tr_gt_all=NormalizeFea(S_te_all.T,2).T
dist = 1 - spatial.distance.cdist(X_te,S_tr_gt_all.dot(W.T),'cosine')

print('Accuracy: {}'.format(acc_zsl(dist, test_cls, Y_te)*100))
