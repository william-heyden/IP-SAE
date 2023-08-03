#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:26:45 2023

@author: will
"""
#TODO
#SNIPS data set for ZSL NLP?

import os
import numpy as np
from scipy import io
from sae_helper import NormalizeFea

def cub(norm_data=True, int_proj=False):
    """
    #X_tr: train features [8821x2248]
    #X_te: test features [2967x2248]
    #S_te_pro: test semantics [50x20]
    #Y_te: test labels (indx) [2967x1]
    #te_cl_id: test labels (unique) [50x1]
    """
    
    a =    [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
        93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
        183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
        254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]
        
    res101 = io.loadmat(os.path.expanduser('~') + '/data/CUB/res101.mat')
    att_splits=io.loadmat(os.path.expanduser('~') + '/data/CUB/att_splits.mat')
     
    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'
    
    labels = res101['labels']
    labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
    labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])
     
    train_labels_seen = np.unique(labels_train)
    val_labels_unseen = np.unique(labels_val)
    test_labels_unseen = np.unique(labels_test)
    
    train_labels_seen = np.array([x-1 for x in train_labels_seen])
    
    val_labels_unseen= np.array([x-1 for x in val_labels_unseen])
    
    test_labels_unseen = np.array([x-1 for x in test_labels_unseen])
     
    i=0
    for labels in train_labels_seen:
      labels_train[labels_train == labels] = i    
      i+=1
     
    j=0
    for labels in val_labels_unseen:
      labels_val[labels_val == labels] = j
      j+=1
     
    k=0
    for labels in test_labels_unseen:
      labels_test[labels_test == labels] = k
      k+=1
     
    sig = att_splits['att']
    sig = np.delete(sig, a, axis=0)
    test_sig = sig[:, test_labels_unseen] 
    
    testClasses = test_labels_unseen
    trainClasses =  np.concatenate([train_labels_seen , val_labels_unseen], axis=0)
    
    # Loads features
    X = res101['features'].transpose()
    Y_temp = res101['labels']
    Y_temp=Y_temp.transpose()
    Y_temp=Y_temp[0]
    Y = np.array([x-1 for x in Y_temp])
    Y =Y.astype(np.int32).transpose()
    att =  att_splits['att']
    att = np.delete(att, a, axis=0)
    att=np.transpose(att)
    
    noExs = X.shape[0]
    
    trainDataX = []
    trainDataLabels = []
    trainDataAttrs = []
    testDataX = []
    testDataLabels = []
    testDataAttrs = []
    
    for ii in range(0,noExs):
        if(Y[ii] in trainClasses):
            trainDataX = trainDataX + [X[ii]]
            trainDataLabels = trainDataLabels + [Y[ii]]
            trainDataAttrs = trainDataAttrs + [att[Y[ii]]]
        elif(Y[ii] in testClasses):
            #print(str(Y[ii])  + "  is in   " + str(testClasses))
            testDataX = testDataX + [X[ii]]
            testDataLabels = testDataLabels + [Y[ii]]
            testDataAttrs = testDataAttrs + [att[Y[ii]]]
        else:
            print('Fatal Error... Please check code/data')
             
    trainDataX = np.array(trainDataX)
    trainDataLabels = np.array(trainDataLabels)
    trainDataAttrs = np.array(trainDataAttrs)
    testDataX = np.array(testDataX)
    testDataLabels = np.array(testDataLabels)
    testDataAttrs = np.array(testDataAttrs)
    
    X_tr=trainDataX
    X_te=testDataX
    if norm_data:
        X_tr=NormalizeFea(X_tr.T,2).T
    S_tr=trainDataAttrs
    if int_proj:
        X_tr =  np.concatenate([X_tr , S_tr], axis=1)
    
    
    S_te_pro=np.transpose(test_sig)
    Y_te=testDataLabels
    te_cl_id=np.unique(testDataLabels)
    if int_proj:
        X_te =  np.concatenate([X_te , testDataAttrs], axis=1)
    if norm_data:
        S_te_pro=NormalizeFea(S_te_pro.T,2).T

    
    return X_tr, X_te, S_tr, S_te_pro, Y_te, te_cl_id, trainDataLabels, testDataAttrs, att

#%% AWA2
def awa2(norm_data=True, int_proj=False):
    """
    #X_tr: train features [8821x2248]
    #X_te: test features [2967x2248]
    #S_te_pro: test semantics [50x20]
    #Y_te: test labels (indx) [2967x1]
    #te_cl_id: test labels (unique) [50x1]
    """
    
    res101 = io.loadmat(os.path.expanduser('~') + '/data/AWA2/res101.mat')
    att_splits=io.loadmat(os.path.expanduser('~') + '/data/AWA2/att_splits.mat')
     
    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'
         
    labels = res101['labels']
    labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
    labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])
     
    train_labels_seen = np.unique(labels_train)
    val_labels_unseen = np.unique(labels_val)
    test_labels_unseen = np.unique(labels_test)
    
    train_labels_seen = np.array([x-1 for x in train_labels_seen])
    val_labels_unseen= np.array([x-1 for x in val_labels_unseen])
    test_labels_unseen = np.array([x-1 for x in test_labels_unseen])
    i=0
    for labels in train_labels_seen:
      labels_train[labels_train == labels] = i    
      i+=1
    j=0
    for labels in val_labels_unseen:
      labels_val[labels_val == labels] = j
      j+=1
    k=0
    for labels in test_labels_unseen:
      labels_test[labels_test == labels] = k
      k+=1
     
    sig = att_splits['att']
    test_sig = sig[:, test_labels_unseen]
    testClasses = test_labels_unseen
    trainClasses =  np.concatenate([train_labels_seen , val_labels_unseen], axis=0)
    
    # Loads features
    X = res101['features'].transpose()
    Y_temp = res101['labels']
    Y_temp=Y_temp.transpose()
    Y_temp=Y_temp[0]
    Y = np.array([x-1 for x in Y_temp])
    Y =Y.astype(np.int32).transpose()
    att =  att_splits['att']
    att=np.transpose(att)
    noExs = X.shape[0]
    
    trainDataX = []
    trainDataLabels = []
    trainDataAttrs = []
    testDataX = []
    testDataLabels = []
    testDataAttrs = []
    
    for ii in range(0,noExs):
        if(Y[ii] in trainClasses):
            trainDataX = trainDataX + [X[ii]]
            trainDataLabels = trainDataLabels + [Y[ii]]
            trainDataAttrs = trainDataAttrs + [att[Y[ii]]]
        elif(Y[ii] in testClasses):
            #print(str(Y[ii])  + "  is in   " + str(testClasses))
            testDataX = testDataX + [X[ii]]
            testDataLabels = testDataLabels + [Y[ii]]
            testDataAttrs = testDataAttrs + [att[Y[ii]]]
        else:
            print('Fatal Error... Please check code/data')
                
    trainDataX = np.array(trainDataX)
    trainDataLabels = np.array(trainDataLabels)
    trainDataAttrs = np.array(trainDataAttrs)
    
    testDataX = np.array(testDataX)
    testDataLabels = np.array(testDataLabels)
    testDataAttrs = np.array(testDataAttrs)
        
    X_tr=trainDataX
    X_te=testDataX
    if norm_data:
        X_tr=NormalizeFea(X_tr.T,2).T
    S_tr=trainDataAttrs
    if int_proj:
        X_tr =  np.concatenate([X_tr , S_tr], axis=1)
    
    
    S_te_pro=np.transpose(test_sig)    
    Y_te=testDataLabels
    te_cl_id=np.unique(testDataLabels)
    if int_proj:
        X_te =  np.concatenate([X_te , testDataAttrs], axis=1)
    if norm_data:
        S_te_pro=NormalizeFea(S_te_pro.T,2).T

    return X_tr, X_te, S_tr, S_te_pro, Y_te, te_cl_id, trainDataLabels, testDataAttrs, att


#%% SUN
def sun(norm_data=True, int_proj=False):
    """
    #X_tr: train features [8821x2248]
    #X_te: test features [2967x2248]
    #S_te_pro: test semantics [50x20]
    #Y_te: test labels (indx) [2967x1]
    #te_cl_id: test labels (unique) [50x1]
    """
    res101 = io.loadmat(os.path.expanduser('~') + '/data/SUN/res101.mat')
    att_splits=io.loadmat(os.path.expanduser('~') + '/data/SUN/att_splits.mat')
     
    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'
         
    labels = res101['labels']
    labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
    labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])
     
    train_labels_seen = np.unique(labels_train)
    val_labels_unseen = np.unique(labels_val)
    test_labels_unseen = np.unique(labels_test)
    
    train_labels_seen = np.array([x-1 for x in train_labels_seen])
    val_labels_unseen= np.array([x-1 for x in val_labels_unseen])
    test_labels_unseen = np.array([x-1 for x in test_labels_unseen])
     
    i=0
    for labels in train_labels_seen:
      labels_train[labels_train == labels] = i    
      i+=1
    j=0
    for labels in val_labels_unseen:
      labels_val[labels_val == labels] = j
      j+=1
    k=0
    for labels in test_labels_unseen:
      labels_test[labels_test == labels] = k
      k+=1
     
    sig = att_splits['att']
    test_sig = sig[:, test_labels_unseen]
    testClasses = test_labels_unseen
    trainClasses =  np.concatenate([train_labels_seen , val_labels_unseen], axis=0)
    
    # Loads features
    X = res101['features'].transpose()
    Y_temp = res101['labels']
    Y_temp=Y_temp.transpose()
    Y_temp=Y_temp[0]
    Y = np.array([x-1 for x in Y_temp])
    Y =Y.astype(np.int32).transpose()
    att =  att_splits['att']
    att=np.transpose(att)  
    noExs = X.shape[0]
    
    trainDataX = []
    trainDataLabels = []
    trainDataAttrs = []
    testDataX = []
    testDataLabels = []
    testDataAttrs = []
    
    for ii in range(0,noExs):
        if(Y[ii] in trainClasses):
            trainDataX = trainDataX + [X[ii]]
            trainDataLabels = trainDataLabels + [Y[ii]]
            trainDataAttrs = trainDataAttrs + [att[Y[ii]]]
        elif(Y[ii] in testClasses):
            #print(str(Y[ii])  + "  is in   " + str(testClasses))
            testDataX = testDataX + [X[ii]]
            testDataLabels = testDataLabels + [Y[ii]]
            testDataAttrs = testDataAttrs + [att[Y[ii]]]
        else:
            print('Fatal Error... Please check code/data')
                
    trainDataX = np.array(trainDataX)
    trainDataLabels = np.array(trainDataLabels)
    trainDataAttrs = np.array(trainDataAttrs)
    
    testDataX = np.array(testDataX)
    testDataLabels = np.array(testDataLabels)
    testDataAttrs = np.array(testDataAttrs)
        
    X_tr=trainDataX
    X_te=testDataX
    if norm_data:
        X_tr=NormalizeFea(X_tr.T,2).T
    S_tr=trainDataAttrs
    if int_proj:
        X_tr =  np.concatenate([X_tr , S_tr], axis=1)
    
    S_te_pro=np.transpose(test_sig)    
    Y_te=testDataLabels
    te_cl_id=np.unique(testDataLabels)
    if int_proj:
        X_te =  np.concatenate([X_te , testDataAttrs], axis=1)
    if norm_data:
        S_te_pro=NormalizeFea(S_te_pro.T,2).T

    return X_tr, X_te, S_tr, S_te_pro, Y_te, te_cl_id, trainDataLabels, testDataAttrs, att



#%% AWA1
def awa1(norm_data=True, int_proj=False):
    
    res101 = io.loadmat(os.path.expanduser('~') + '/data/AWA1/res101.mat')
    att_splits=io.loadmat(os.path.expanduser('~') + '/data/AWA1/att_splits.mat')
     
    train_loc = 'train_loc'
    val_loc = 'val_loc'
    test_loc = 'test_unseen_loc'
         
    labels = res101['labels']
    labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
    labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
    labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])
     
    train_labels_seen = np.unique(labels_train)
    val_labels_unseen = np.unique(labels_val)
    test_labels_unseen = np.unique(labels_test)
    
    train_labels_seen = np.array([x-1 for x in train_labels_seen]) 
    val_labels_unseen= np.array([x-1 for x in val_labels_unseen])
    test_labels_unseen = np.array([x-1 for x in test_labels_unseen])
     
    i=0
    for labels in train_labels_seen:
      labels_train[labels_train == labels] = i    
      i+=1  
    j=0
    for labels in val_labels_unseen:
      labels_val[labels_val == labels] = j
      j+=1   
    k=0
    for labels in test_labels_unseen:
      labels_test[labels_test == labels] = k
      k+=1
     
    sig = att_splits['att']
    test_sig = sig[:, test_labels_unseen] 
    testClasses = test_labels_unseen
    trainClasses =  np.concatenate([train_labels_seen , val_labels_unseen], axis=0)
    
    # Loads features
    X = res101['features'].transpose()
    Y_temp = res101['labels']
    Y_temp=Y_temp.transpose()
    Y_temp=Y_temp[0]
    Y = np.array([x-1 for x in Y_temp])
    Y =Y.astype(np.int32).transpose()
    att =  att_splits['att']
    att=np.transpose(att)     
    noExs = X.shape[0]
    
    trainDataX = []
    trainDataLabels = []
    trainDataAttrs = []
    testDataX = []
    testDataLabels = []
    testDataAttrs = []
    
    for ii in range(0,noExs):
        if(Y[ii] in trainClasses):
            trainDataX = trainDataX + [X[ii]]
            trainDataLabels = trainDataLabels + [Y[ii]]
            trainDataAttrs = trainDataAttrs + [att[Y[ii]]]
        elif(Y[ii] in testClasses):
            #print(str(Y[ii])  + "  is in   " + str(testClasses))
            testDataX = testDataX + [X[ii]]
            testDataLabels = testDataLabels + [Y[ii]]
            testDataAttrs = testDataAttrs + [att[Y[ii]]]
        else:
            print('Fatal Error... Please check code/data')
                
    trainDataX = np.array(trainDataX)
    trainDataLabels = np.array(trainDataLabels)
    trainDataAttrs = np.array(trainDataAttrs)
    
    testDataX = np.array(testDataX)
    testDataLabels = np.array(testDataLabels)
    testDataAttrs = np.array(testDataAttrs)
        
    X_tr=trainDataX
    X_te=testDataX
    if norm_data:
        X_tr=NormalizeFea(X_tr.T,2).T
    S_tr=trainDataAttrs
    if int_proj:
        X_tr =  np.concatenate([X_tr , S_tr], axis=1)
    
    
    S_te_pro=np.transpose(test_sig)    
    Y_te=testDataLabels
    te_cl_id=np.unique(testDataLabels)
    if int_proj:
        X_te =  np.concatenate([X_te , testDataAttrs], axis=1)
    if norm_data:
        S_te_pro=NormalizeFea(S_te_pro.T,2).T

    return X_tr, X_te, S_tr, S_te_pro, Y_te, te_cl_id, trainDataLabels, testDataAttrs, att

#%% Accuracy
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
