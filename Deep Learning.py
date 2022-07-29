#1.Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense


#2.data preporcessing
veriler = pd.read_csv('churn_model.csv',sep = ';')

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder: Kategorik -> Numeric

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

#le2 = preprocessing.LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])


ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]


#train test split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#scaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#3 Neural network designing

classifier = Sequential()


classifier.add(Dense(6, kernel_initializer="glorot_uniform", activation = 'relu' , input_dim = 11)) #input

classifier.add(Dense(6, kernel_initializer="glorot_uniform", activation = 'relu')) #hidden 1. layer

classifier.add(Dense(1, kernel_initializer="glorot_uniform", activation = 'sigmoid')) #hidden 2. layer

classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy' , metrics = ['sparse_categorical_accuracy'] ) #output

#only needed for training
'''
classifier.fit(X_train, y_train, epochs=50)
'''

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
 

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
X = np.asarray(X).astype(np.int_)
Y = np.array(Y).astype(np.int_)
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
