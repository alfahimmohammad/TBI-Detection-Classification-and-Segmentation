# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:46:03 2020

@author: Kather
"""

#%%
import zipfile

target = 'machine-learning-ex3.zip'
handle = zipfile.ZipFile(target)
handle.extractall('MLex3')
handle.close()
#%%
import pandas as pd

df = pd.read_csv('test_metadata_noidx.csv')
df1 = df.sort_values(by=["PatientID"], axis=0, ascending=True)
df1.to_csv('test_metadata.csv')
#%%
import pandas as pd

COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

train = pd.read_csv('stage_2_train.csv')
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
#%%
import pydicom as dicom
import matplotlib.pylab as plt

# specify your image path
image_path = 'ID_000039fa0.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)
#%%
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import pydicom
import pylab as pl
import sys
import matplotlib.path as mplPath

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('Slice Number: %s' % self.ind)
        self.im.axes.figure.canvas.draw()

fig, ax = plt.subplots(1,1)

os.system("tree C:/Users/Dicom_ROI")

plots = []

for f in glob.glob("C:/Users/Dicom_ROI/AXIAL_2/*.dcm"):
    pass
    filename = f.split("/")[-1]
    ds = pydicom.dcmread(filename)
    pix = ds.pixel_array
    pix = pix*1+(-1024)
    plots.append(pix)

y = np.dstack(plots)

tracker = IndexTracker(ax, y)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()