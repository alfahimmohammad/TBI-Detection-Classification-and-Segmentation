#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:29:02 2020

@author: sindhura
"""

#%%
import numpy as np, os, pickle, cv2, glob

from sklearn import metrics
from imageio import imsave
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import random 
import nibabel as nib
from torch.utils.data import DataLoader
import torch.optim as optim

from datagenpt import *
from pt3dnestnet import *
random.seed(0)


def Sens(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SensI = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    return SensI  # TPR is also known as sensitivity

def Speci(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SpeciI = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return SpeciI  # FPR is one minus the specificity or true negative rate

def Jaccard_img(y_true, y_pred): #https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    y_true = np.asarray(y_true.detach().cpu().squeeze()).astype(np.bool)
    y_pred = np.asarray(y_pred.detach().cpu().squeeze()).astype(np.bool)
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    Jscore = np.sum(intersection)/np.sum(union)
    return Jscore


def dice_fun(im1, im2):
    im1 = np.asarray(im1.detach().cpu().squeeze()).astype(np.bool)
    im2 = np.asarray(im2.detach().cpu().squeeze()).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def dice_loss(y, pred):
    #smooth = 1.

    intersection = (y * pred).sum()
    
    return 1 - ((2. * intersection + 1.) /
              (y.sum() + pred.sum() + 1.))

def window_ct (ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0]=0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        ct_scan[:,:,s] = slice_s

    return ct_scan

def load_ct_mask(datasetDir, sub_n, window_specs):
    ct_dir_subj = Path(datasetDir, 'ct_scans', "{0:0=3d}.nii".format(sub_n))
    ct_scan_nifti = nib.load(str(ct_dir_subj))
    ct_scan = ct_scan_nifti.get_data()
    ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])  # Convert the CT scans using a brain window
    # Loading the masks
    masks_dir_subj = Path(datasetDir, 'masks', "{0:0=3d}.nii".format(sub_n))
    masks_nifti = nib.load(str(masks_dir_subj))
    mask = masks_nifti.get_data()
    return ct_scan, mask


if __name__=='__main__':

    NumEpochs=10
    batch_size = 1
    learning_rateI = 1e-5
    thresholdI= 0.5
    window_specs = [40, 120]  # Brain window
    #splitting data in 
    currentDir = Path(os.getcwd())
    dataset_zip_dir='computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1.zip'
    datasetDir=Path(currentDir, 'ich_data', dataset_zip_dir[:-4])
    hemorrhage_diagnosis_df = pd.read_csv(Path(datasetDir, 'hemorrhage_diagnosis_raw_ct.csv'))
    hda = hemorrhage_diagnosis_df[['PatientNumber','SliceNumber','Intraventricular','Intraparenchymal','Subarachnoid','Epidural', 'Subdural', 'No_Hemorrhage']].to_numpy()   
    hda[:, 0] = hda[:, 0] - 49
    s = np.unique(hda[:,0])
    n = [] #no hemorrhage
    y = [] #have hemorrhage
    for i in s:
        d = hda[hda[:,0]==i]
        if np.sum(d[:,-1]) == d.shape[0]:
            n.append(i)
        else:
            y.append(i)
    random.shuffle(n)
    random.shuffle(y)
    n, y = np.asarray(n), np.asarray(y)
    train = []
    valid = []
    test = []
    for i in range(int(0.6*len(n))):
        train.append(n[i])
    for i in range(int(0.6*len(y))):
        train.append(y[i])
    for i in range(int(0.6*len(n)),int(0.8*len(n))):
        valid.append(n[i])
    for i in range(int(0.6*len(y)),int(0.8*len(y))):
        valid.append(y[i])    
    for i in range(int(0.8*len(n)),len(n)):
        test.append(n[i])
    for i in range(int(0.8*len(y)),len(y)):
        test.append(y[i])
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    ct_scans = {}
    masks = {}

    for subn in train:
        ct_scans[str(subn)], masks[str(subn)] = load_ct_mask(datasetDir, subn+49, window_specs)

    for subn in valid:
        ct_scans[str(subn)], masks[str(subn)] = load_ct_mask(datasetDir, subn+49, window_specs)

    for subn in test:
        ct_scans[str(subn)], masks[str(subn)]= load_ct_mask(datasetDir, subn+49, window_specs)

    train_gen = DataGenerator(train, ct_scans, masks)
    trainer = DataLoader(train_gen, batch_size = 1, shuffle = False)
    
    valid_gen = DataGenerator(valid, ct_scans, masks)
    valider = DataLoader(valid_gen, batch_size = 1, shuffle = False)
    
    test_gen = DataGenerator(test, ct_scans, masks)
    tester = DataLoader(test_gen, batch_size = 1, shuffle= False)
#%%
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Nested_UNet()
    optimizer = optim.Adam(model.parameters(), learning_rateI)
    print('training starts')
    print('-'*10)
    for epoch in range(NumEpochs):
        model.train()
        print('\nEpoch: '+str(epoch+1))
        for idx, (x, y) in enumerate(trainer):
            print(str(idx),end=" ")
            #x = x.to('cuda:0')
            y = y.cuda()
            
            pred = model(x)
            optimizer.zero_grad()
            dice_loss(y, pred).backward()
            
            #optimizer.zero_grad()
            #loss.backward()
            optimizer.step()
            """
            x = x.detach().to('cpu').numpy()
            y = y.detach().to('cpu').numpy()
            pred = pred.detach().to('cpu').numpy()
            loss = loss.detach().to('cpu').numpy()
            """
            del x
            del y
            del pred
            torch.cuda.empty_cache()
        model.eval()
        L = 0
        valid_dice = np.zeros(len(valid))
        with torch.no_grad():
            print('\nValidating\n')
            for idx, (x, y) in enumerate(valider):
                print(str(idx),end=" ")
                x = x.to('cuda:0')
                y = y.to('cuda:0')
                
                pred = model(x)
                
                pred[pred>=thresholdI] = 1
                pred[pred<thresholdI] = 0
                valid_dice[idx] = dice_fun(pred, y)
                loss = dice_loss(y, pred)
                loss = loss.detach().to('cpu').numpy()
                L += float(loss)
                """
                x = x.detach().to('cpu').numpy()
                y = y.detach().to('cpu').numpy()
                pred = pred.detach().to('cpu').numpy()
                """
                del x, y, pred, loss
                torch.cuda.empty_cache()
            L = L.detach().to('cpu').numpy()
            AvgLoss = L/(idx+1)
            print('avergae validation loss is {:.3f}'.format(AvgLoss))
            print('average validation dice score is {:.3f}'.format(np.sum(valid_dice)/(idx+1)))
            print('max validation dice score is {:.3f}'.format(np.max(valid_dice)))
            print('min validation dice score is {:.3f}'.format(np.min(valid_dice)))
            checkpoint = {
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }
            torch.save(checkpoint, '3dnestnet-'+str(epoch+1)+'.pt')
    model.eval()
    test_dice = np.zeros(len(test))
    test_jaccard = np.zeros(len(test))
    test_loss = 0
    with torch.no_grad():
        print('\ntesting\n')
        for idx, (x, y) in enumerate(tester):
            print(str(idx),end=" ")
            x = x.to('cuda:0')
            y = y.to('cuda:0')
            
            pred = model(x)
            pred[pred>=thresholdI] = 1
            pred[pred<thresholdI] = 0
            test_dice[idx] = dice_fun(pred, y)
            test_jaccard[idx] = Jaccard_img(pred, y)
            loss = dice_loss(y, pred)
            loss = loss.detach().to('cpu').numpy()
            test_loss += float(loss)
            """
            x = x.detach().to('cpu').numpy()
            y = y.detach().to('cpu').numpy()
            pred = pred.detach().to('cpu').numpy()
            """
            del x, y, pred, loss
            torch.cuda.empty_cache()
        test_loss = test_loss.detach().to('cpu').numpy()
        AvgTestLoss = test_loss/(idx+1)
        print('Avg. test loss: {:.3f}'.format(AvgTestLoss))
        print('Dice -> Avg: {:.3f}, Max: {:.3f}, Min: {:.3f}'.format(np.sum(test_dice)/(idx+1), np.max(test_dice), np.min(test_dice)))
        print('Jaccard -> Avg: {:.3f"}, Max: {:.3f}, Min: {:.3f}'.format(np.sum(test_jaccard)/(idx+1), np.max(test_jaccard), np.min(test_jaccard)))
        
      
    