import h5py
import numpy as np
from tempfile import TemporaryFile
import os
import tensorflow as tf
#import pickle
import random
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
gh = []
l = []
labels = []
label = []
k = []
la = []
masks = []
images = []
f = []
pa = Path("D:/ICAD_Projekt/AI_Segmentation/Training_setting1/")
os.chdir(pa)
#labels = [name for name in os.listdir(".") if os.path.isdir(name)]
#print(labels)
#for name in labels:
f = h5py.File('IC_venous1_new.mat', 'r') #Location of matsrtuct 
g = h5py.File('IC_venous1_new.mat', 'r') #Location of matsrtuct
#k = h5py.File('test.mat', 'r+')

print(list(f.keys()))
print(list(g.keys()))
tv = f['IC']
la = g['IC']
print(list(tv.keys()))
flow = f[tv['data'][0,0]] #Name of struct field for the image data
print(flow)
labe = g[la['truth'][0,0]].value # Name of struct field for ground-truth
#mask = f[tv['label'][0,0]].value


print(flow.dtype)
print(labe.dtype)
#print(mask.dtype)

#print(la)
#te = k['test']
#print(list(la.keys()))
#flow = f[tv['data'][0,0]].value
#labe = g[la['data'][0,0]].value
#test = k[te['data'][0,0]].value
#print( flow.shape)

#print(tv.shape)
for i in tqdm(range(len(tv['data']))):
    flow = f[tv['data'][i,0]].value
    #labe = g[la['noise'][i,0]].value
    #mask = f[tv['label'][i,0]].value
    labe = g[la['truth'][i,0]].value
    X = flow.transpose()
    Y = labe.transpose()
    #Z = mask
    dim1 = X.shape[0]
    dim2 = X.shape[1]
    #h1 = round((dim1-160)/2)
    w2 = round((dim2-80)/2)
    #X = X[32:,:,:]
    #Y = Y[32:,:,:]
    Y[Y<1] = 0

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    #Z = Z.astype(np.float32)
   # Z = Z[:140,:,:]
    #for j in range(2):
    #j = 1
    #X[...,j] = (X[...,j] - np.amin(X[...,j]))/(np.amax(X[...,j]) - np.amin(X[...,j]))
    images.append(X)
    labels.append(Y)
    #masks.append(Z)
    
print(len(images))
print(len(labels))
print(images[0].shape)
print(labels[0].shape)
print(images[1].shape)
#print(mask[0])
#print(images[2].shape)
#print(images[3].shape)
#print(images[4].shape)

#print(masks[0].shape)

print(images[0].dtype)
print(labels[0].dtype)
print(np.unique(labels[0]))
#print(np.unique(masks[0]))




def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#plt.imshow(train_image[0])
#plt.show()
s = list(range(len(images)))
#random.shuffle(s)
test_filename = 'IC_venous1_new.tfrecords' #Name of saved tfrecord
#writer = tf.python_io.TFRecordWriter(test_filename)
writer = tf.io.TFRecordWriter(test_filename)

for i in s:
    image_raw =  images[i].tostring()
    label_raw = labels[i].tostring()
    #mask_raw = masks[i]

    height = images[i].shape[0]
    width = images[i].shape[1]
    depth = images[i].shape[2]
    features = {'test/image': _bytes_feature(image_raw),
               'test/label': _bytes_feature(label_raw),
               #'test/mask': _int64_feature(mask_raw),
               'test/height': _int64_feature(height),
               'test/depth':_int64_feature(depth),
               'test/width': _int64_feature(width)}
    examples = tf.train.Example(features = tf.train.Features(feature = features))
    writer.write(examples.SerializeToString())

writer.close()



#print(merge[0].shape)


