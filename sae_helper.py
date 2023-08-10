import numpy as np
import argparse
from scipy import io, spatial, linalg
from sklearn import preprocessing
import statistics
from sklearn.metrics import confusion_matrix

# Utilisation function for IP-SAE
# SAE for calculating W using Bertels-Stewart algorithim to solve Sylvester Equation
# Normalization for features

# Accuracy calculation for conventional- (acc_zsl) and joint- (acc_gszl) setting
# Both assuming transductive learning, i.e. semantic information for unseen examples are available at testing time
# Returns classification accuracy of seen examples, unseen examples, and the harmonic mean between. 
# See paper for details of accuracy calculation

def SAE(X,S,lamb):
    """ The implementation of Bartels–Stewart algorithm to solve Sylvester Equation. 
    See Github https://github.com/Lzh566566/SAE-in-python/blob/master/SAE.py
    for more details.
    """
    A=S.dot(S.T)
    B=lamb*(X.dot(X.T))
    C=(1+lamb)*(S.dot(X.T))
    W=linalg.solve_sylvester(A,B,C)
    return W

def NormalizeFea(fea,mode):
    '''
    mode==0,do (X-X_mean)/X_std
    mode==1. do (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    '''
    if mode==0:
        norm_fea=preprocessing.scale(fea)
    elif mode==1:
        norm_fea=np.zeros(fea.shape)
        for i in range(fea.shape[0]):
            max_=np.max(fea[i])
            min_=np.min(fea[i])
            norm_fea[i]=(fea[i]-min_)/(max_-min_)
    elif mode==2:
        nSmp,mFea = fea.shape
        feaNorm=np.sqrt(np.sum(fea*fea,1))
        b=np.zeros((nSmp,mFea))
        for ii in range(mFea):
            b[:,ii]=feaNorm
        norm_fea=fea/b
    return norm_fea

def acc_zsl(distance_matrix, classes, test_labels, top_hit=1):
    """ Accuracy measurments under the conventional setting
    """
    dist = distance_matrix
    te_cl_id = classes
    Y_te = test_labels
    Y_hit =np.zeros((dist.shape[0],top_hit))
    for i in range(dist.shape[0]):
        I=np.argsort(dist[i])[::-1]
        Y_hit[i,:]=te_cl_id[I[0:top_hit]]
    n=0
    for i in range(dist.shape[0]):
        if Y_te[i] in Y_hit[i,:]:
            n=n+1
    zsl_accuracy = n/dist.shape[0]
    return zsl_accuracy

def acc_gzsl(distance_matrix, Yte, Ytr):
    """ Accuracy measurments under the generalised setting. 20 % of the train data is
    included during testing.
    """
    dist = distance_matrix
    top_hit = 1
    trainClasses = np.unique(Ytr)
    tr_cl_id_all = np.concatenate([np.unique(Yte), [Ytr[i] for i in range(len(Ytr)) if Ytr[i] not in np.unique(Yte)]])
    Y_hit5 = np.zeros((dist.shape[0], top_hit))
    tmp1=[]
    for i in range(dist.shape[0]):
        I=np.argsort(dist[i])[::-1]
        Y_hit5[i,:]=tr_cl_id_all[I[0:top_hit]]   
    n1=0
    n2=0  
    n1_count=0
    n2_count=0
    for i in range(dist.shape[0]):
        if Yte[i] in trainClasses:
            n1_count=n1_count+1  
            if Yte[i] in Y_hit5[i,:]:
                n1=n1+1
        else:
            n2_count=n2_count+1         
            if Yte[i] in Y_hit5[i,:]:
                n2=n2+1
    seen_acc = n1/n1_count
    unseen_acc = n2/n2_count
    harm_acc = statistics.harmonic_mean([seen_acc, unseen_acc])
    
    return seen_acc, unseen_acc, harm_acc
