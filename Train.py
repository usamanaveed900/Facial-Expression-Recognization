import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

df=pd.read_csv('DataSet/DATASET.csv')

X_train,train_y,X_test,test_y=[],[],[],[]

for index,row in df.iterrows():
	val=row['pixels'].split(" ")
	try:
		if 'Training' in row['Usage']:
			X_train.append(np.array(val,'float32'))
			train_y.append(row['emotion'])
		elif 'PublicTest' in row['Usage']:
			X_test.append(np.array(val,'float32'))
			test_y.append(row['emotion'])
	except:
		print(f"Error Occured At Index : {index} and row : {row}")


num_features=64
num_labels=7
batch_size=64
epochs=30
width,height=48,48


X_train=np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y,num_classes=num_labels)
test_y= np_utils.to_categorical(test_y,num_classes=num_labels)

# Normalizing Data Between 0 and 1

X_train-=np.mean(X_train,axis=0)
X_train/=np.std(X_train,axis=0)

X_test-=np.mean(X_test,axis=0)
X_test/=np.std(X_test,axis=0)

X_train=X_train.reshape(X_train.shape[0],48,48,1)

X_test=X_test.reshape(X_test.shape[0],48,48,1)

# CNN MODEL
# 1st Convolution Layer

model=Sequential()

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pooling_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# 2nd Convolution Layer
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pooling_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# 3rd Convolution Layer
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pooling_size=(2,2),strides=(2,2)))

model.add(Flatten())

# Fully Connected Neutral Networks
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))

model.summary()
# Compliling the model
model.compile(loss=categorical_crossentropy,
			optimizers=Adam(),
			metrics=['accuracy'])

# Training the model
model.fit(X_train,train_y,
		batch_size=batch_size,
		epochs=epochs,
		verbose=1,
		validation_data=(X_test,test_y),
		shuffle=True)

# Saving The Model To Use It Later on
trainData_json=model.to_json()
with open("Trained Model/trainedModel.json","w") as json_file:
	json_file.write(trainData_json)
model.save_weights("Trained Model/trainedModel.h5")