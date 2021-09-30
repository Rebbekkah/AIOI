import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from matplotlib import pyplot

# main_path = 
main_path = "/home/sdv/m2bi/rgoulancourt/AI/Kaggle/data/archive/chest_xray/chest_xray"

# path jusqu'aux dossiers contenant les data de train et test
train_path = os.path.join(main_path, "train")
test_path = os.path.join(main_path, "test")



# variables contenant les data de test et de train 
# d'apprentissage 
train_normal = glob.glob(train_path+"/NORMAL/*.jpeg")
train_pneumonia = glob.glob(train_path+"/PNEUMONIA/*.jpeg")
# échantillon de test
test_normal = glob.glob(test_path+"/NORMAL/*.jpeg")
test_pneumonia = glob.glob(test_path+"/PNEUMONIA/*.jpeg")


train_list = [x for x in train_normal]
train_list.extend([x for x in train_pneumonia])

df_train = pd.DataFrame(np.concatenate([['Normal']*len(train_normal) , ['Pneumonia']*len(train_pneumonia)]), columns = ['class'])
df_train['image'] = [x for x in train_list]

test_list = [x for x in test_normal]
test_list.extend([x for x in test_pneumonia])

df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*len(test_pneumonia)]), columns = ['class'])
df_test['image'] = [x for x in test_list]

print(df_train)
