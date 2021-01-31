
#ABRS_behavior_analysis

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
import os

from ABRS_modules import discrete_radon_transform
from ABRS_modules import etho2ethoAP
from ABRS_modules import smooth_1d
from ABRS_modules import create_LDA_training_dataset
from ABRS_modules import removeZeroLabelsFromTrainingData
from ABRS_modules import computeSpeedFromPosXY 

from ABRS_data_vis import create_colorMat
from ABRS_data_vis import cmapG



def remove_empty_ethograms (ethoMat,minBeh = 10000):

    shEthoMat = np.shape(ethoMat)

    ethoMatBin = np.zeros((shEthoMat[0],shEthoMat[1]))

    ethoMatBin[ethoMat != 7] = 1
    sumCol = np.sum(ethoMatBin,1)
    sumInd = np.where(sumCol<minBeh)

    ethoMatFull = np.zeros((shEthoMat[0]-np.shape(sumInd)[1],shEthoMat[1]))

    ind=0;
    for i in range(0,shEthoMat[0]):
        if sumCol[i]>minBeh:
            ethoMatFull[ind,:] = ethoMat[i,:]
            ind = ind+1

    return ethoMatFull

def get_behavior_probability(idx):

    shIdx = np.shape(idx)

    idxBin = np.zeros((1,shIdx[1]))

    probVect = np.zeros((int(np.max(idx)+1),1))

    for b in range(1,int(np.max(idx))+1):
            
        idxBin = np.zeros((1,shIdx[1]))
        idxBin[idx == b] = 1

        probVect[b-1,0] = np.sum(idxBin)

    return probVect    

def get_probability_progression(ethoMat):

    shEthoMat = np.shape(ethoMat)

    ethoMatBin = np.zeros((shEthoMat[0],shEthoMat[1]))

    windSize = 1000
    stepWin = 1

    probMat = np.zeros((int(np.max(ethoMat)+1),shEthoMat[1]-windSize))

    for b in range(1,int(np.max(ethoMat))+1):
        
        ethoMatBin = np.zeros((shEthoMat[0],shEthoMat[1]))
        ethoMatBin[ethoMat == b] = 1

        #print(b)

        for i in range(0,np.shape(ethoMatBin)[1]-windSize,stepWin):

            probMat[b-1,i] = np.sum(ethoMatBin[:,i:i+windSize])

    return probMat       

def get_durations (idx):

    shIdx = np.shape(idx)

    ind_d=0
    ind=1
      
    durCol=np.zeros((3,1))
    durRec=np.zeros((3,1))

    for i in range(1,shIdx[1]):
        
        if idx[0,i]!= idx[0,i-1] or i==0 or i==shIdx[1]:
                   
            ident=idx[0,i-1]
            #print(ident)
            
            dur=ind_d
            
            durCol[0,0]=ident
            durCol[1,0]=dur
            durCol[2,0]=i

            if i == 1:

                durRec = durCol

            if i > 1:

                durRec = np.hstack((durRec,durCol))
            
            ind_d=0
            ind=ind+1

        ind_d=ind_d+1

    return durRec

def get_syntax (idx):

    idx1 = idx

    idx2=np.zeros((1,np.shape(idx1)[1]))
    idx2[0,0:np.shape(idx1)[1]-1] = idx1[0,1:np.shape(idx1)[1]]

    TFmat = np.zeros((7,7))

    for i in range(0,np.shape(idx1)[1]):

        TFmat[int(idx1[0,i])-1,int(idx2[0,i])-1] = TFmat[int(idx1[0,i])-1,int(idx2[0,i])-1]+1
            
    TPmat=TFmat/np.shape(idx1)[1]

    TPmatNorm = np.zeros((7,7))

    for i in range(0,7):
            
        TPmatNorm[:,i]=TPmat[:,i]/np.sum(TPmat[:,i])

    TPnoSelf = TPmat

    for i in range(0,7):
            
        TPnoSelf[i,i] = 0
        sc = np.sum(TPnoSelf[i,:])
        
        if sc > 0:
            TPnoSelf[i,:] = TPnoSelf[i,:]/sc

    return TFmat, TPmat, TPmatNorm, TPnoSelf       


def etho2ethoAP (idx):

    sh = np.shape(idx);
    idxAP = np.zeros((1,sh[1]))

    idxAP[0,idx[0,:]==1]=1
    idxAP[0,idx[0,:]==2]=1
    idxAP[0,idx[0,:]==3]=2
    idxAP[0,idx[0,:]==4]=2
    idxAP[0,idx[0,:]==5]=2
    idxAP[0,idx[0,:]==6]=3   

    return idxAP

def remove_zeros_from_etho (idx):

    shIdx = np.shape(idx)

    idxNew = np.zeros((shIdx[0],shIdx[1]))

    for i in range(0,shIdx[1]):

        idxNew[0,i]=idx[0,i]

        if idx[0,i] == 0:
            
           idxNew[0,i]=idxNew[0,i-1]

    return idxNew       


def post_process_etho (idx):

    shIdx = np.shape(idx)

    idxAP = etho2ethoAP (idx)
    #durRecG = get_durations (idx);#np.mean(durRec[1,:])
    durRecAP = get_durations (idxAP);

    idxNew = np.zeros((1,shIdx[1]))
    idxNew[0,0:shIdx[1]] = idx[0,0:shIdx[1]]

    idxS = idx

    minDurWalk=5;
    minDurSilence=5;
    minDurAPW=10;
    minDurAPA=30;

    durRecAP = get_durations (idxAP)

    shDurRecAP = np.shape(durRecAP)

    for d in range(1,shDurRecAP[1]-1):
        
        if durRecAP[0,d] == 3 and durRecAP[1,d] < minDurWalk:

           #print(durRecAP[1,d])        
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-1)] = idxS[0,int(durRecAP[2,d-1]-1)]
        
        if durRecAP[0,d] == 1 and durRecAP[0,d-1] == 2 and durRecAP[0,d+1] == 2 and durRecAP[1,d] < minDurAPA and \
           durRecAP[1,d] < durRecAP[1,d-1] and durRecAP[0,d] < durRecAP[1,d+1]:
          
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-1)] = idxS[0,int(durRecAP[2,d-1]-1)]
        
        if durRecAP[0,d] == 2 and durRecAP[0,d-1] == 1 and durRecAP[0,d+1] == 1 and durRecAP[1,d] < minDurAPA and \
           durRecAP[1,d] < durRecAP[1,d-1] and durRecAP[1,d] < durRecAP[1,d+1]:
            
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-1)] = idxS[0,int(durRecAP[2,d-1]-1)];
        
        if durRecAP[1,d] < minDurAPW:
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-1)] = idxS[0,int(durRecAP[2,d-1]-1)]
           
    return idxNew

def post_process_etho3 (idx,minDurWalk,minDurSilence,minDurAPW,minDurAPA):

    shIdx = np.shape(idx)

    idxAP = etho2ethoAP (idx)
    durRecG = get_durations (idx);#np.mean(durRec[1,:])
    durRecAP = get_durations (idxAP);

    idxNew = np.zeros((1,shIdx[1]))
    idxNew[0,0:shIdx[1]] = idx[0,0:shIdx[1]]

    idxS = idx
    idxOriginal = idx
    idxOriginalAP = idxAP

    #minDurWalk=5;
    #minDurSilence=5;
    #minDurAPW=10;
    #minDurAPA=30;

    durRecAP = get_durations (idxAP)

    shDurRecAP = np.shape(durRecAP)

    for d in range(1,shDurRecAP[1]-1):
        
        if durRecAP[0,d] == 3 and durRecAP[1,d] < minDurWalk:
                
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-0)] = idxS[0,int(durRecAP[2,d-1]-1)]

    idxS = idxNew
    idxAP = etho2ethoAP (idxNew)       
    durRecAP = get_durations (idxAP)
    shDurRecAP = np.shape(durRecAP)

    for d in range(1,shDurRecAP[1]-1):       
        
        if durRecAP[0,d] == 1 and durRecAP[0,d-1] == 2 and durRecAP[0,d+1] == 2 and durRecAP[1,d] < minDurAPA and \
           durRecAP[1,d] < durRecAP[1,d-1] and durRecAP[0,d] < durRecAP[1,d+1]:       
          
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-0)] = idxS[0,int(durRecAP[2,d-1]-1)]

    idxS = idxNew
    idxAP = etho2ethoAP (idxNew)       
    durRecAP = get_durations (idxAP)
    shDurRecAP = np.shape(durRecAP)

    for d in range(1,shDurRecAP[1]-1):       
        
        if durRecAP[0,d] == 2 and durRecAP[0,d-1] == 1 and durRecAP[0,d+1] == 1 and durRecAP[1,d] < minDurAPA and \
           durRecAP[1,d] < durRecAP[1,d-1] and durRecAP[1,d] < durRecAP[1,d+1]:
           
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-0)] = idxS[0,int(durRecAP[2,d-1]-1)]

    idxS = idxNew
    idxAP = etho2ethoAP (idxNew)       
    durRecAP = get_durations (idxAP)
    shDurRecAP = np.shape(durRecAP)

    for d in range(1,shDurRecAP[1]-1):       
        
        if durRecAP[1,d] < minDurAPW:
           
           idxNew[0,int(durRecAP[2,d-1]): int(durRecAP[2,d]-0)] = idxS[0,int(durRecAP[2,d-1]-1)]
        
    #idxS = idxNew;
    idxNewAP = etho2ethoAP (idxNew)

    return idxNew


def get_behavior_freq(idx):

    behFreqCol = np.zeros((int(np.max(idx))+1,1))

    for b in range(0,int(np.max(idx))+1):

        idxBehBin = np.zeros((1,np.shape(idx)[1]))
        idxBehBin[idx==b] = 1
        behFreqCol[b] = np.sum(idxBehBin)

    return behFreqCol    

def get_comulative_freq(idx):
    
    shIdx = np.shape(idx)

    commFreq = np.zeros((int(np.max(idx)),shIdx[1]));
    sumMat = np.zeros((int(np.max(idx)),1));

    for i in range(1,shIdx[1]):
        
        sumMat[int(idx[0,i])-1,0] = sumMat[int(idx[0,i])-1,0] + 1
        commFreq[int(idx[0,i])-1,i] = sumMat[int(idx[0,i])-1,0]

    return commFreq, sumMat
    

def get_half_time (ethoMat):

    shEthoMat = np.shape(ethoMat)

    halfTimeRec = np.zeros((shEthoMat[0],1));
     
    for i in range(0,shEthoMat[0]):
         
        idx = ethoMat[[i]]
     
        idxAP = etho2ethoAP(idx)
        commFreq, sumMat = get_comulative_freq(idxAP)
        halfTime = np.where(commFreq[0,:]==np.round(sumMat[0]/2))[0][0]
        halfTimeRec[i,0]=halfTime

    return halfTimeRec

def count_cycles(idx):

    idxAP = etho2ethoAP (idx)
    
    TFmat, TPmat, TPmatNorm, TPnoSelf = get_syntax (idx)
    behFreqCol = get_behavior_freq(idxAP)

    fhTF = int(np.round((TFmat[1][0]+TFmat[0][1])/1))
    numCycFH = behFreqCol[1]/fhTF

    abwTF = int(np.round((TFmat[2][3]+TFmat[3][2]+TFmat[4][2]+TFmat[2][4])/1))
    numCycABW = behFreqCol[2]/fhTF
    
    return numCycFH, numCycABW


