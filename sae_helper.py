import numpy as np
import argparse
from scipy import io, spatial, linalg
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix



def SAE(X,S,lamb):

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
    #Return n/len(n)
    return zsl_accuracy
