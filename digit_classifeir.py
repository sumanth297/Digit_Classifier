import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
plt.ion()
data=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train, validate=train_test_split(data, test_size=0.2, random_state=100)
train_x=train.drop('label',axis=1)
train_y=train['label']
validate_x=validate.drop('label', axis=1)
validate_y=validate['label']
#print(train_x.shape, train_y.shape, validate_x.shape, validate_y.shape)
data_x=data.drop('label',axis=1)
data_y=data['label']
train_x /=255 
validate_x /=255

model_dt=DecisionTreeClassifier(max_depth=19, random_state=100)
model_dt.fit(train_x, train_y)
validate_pred_dt=model_dt.predict(validate_x)
accuracy_dt=accuracy_score(validate_y, validate_pred_dt)

model_rf=RandomForestClassifier(random_state=100,n_estimators=500)
model_rf.fit(train_x,train_y)
validate_pred_rf=model_rf.predict(validate_x)
accuracy_rf=accuracy_score(validate_y, validate_pred_rf)

model_knn=KNeighborsClassifier(n_neighbors=3)
model_knn.fit(train_x, train_y)
validate_pred_knn=model_knn.predict(validate_x)
accuracy_knn=accuracy_score(validate_y, validate_pred_knn)

model_knn=KNeighborsClassifier(n_neighbors=3)
model_knn.fit(data_x,data_y)
test_pred=model_knn.predict(test)

for i in range(31,41):
    actual_class = np.where(validate_y[i] == 1)[0][0]
    predcited_class =test_pred[i]
    plt.imshow(validate_x[i].reshape(28,28))
    plt.show()
    print('actual : {0} predicted : {1}'.format(actual_class, predcited_class))
