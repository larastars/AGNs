
import csv 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scipy import stats
from keras.utils import to_categorical
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import warnings
import math
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')


#find the transpose of a matrix 
def matrixTranspose(matrix):
    if not matrix: 
        return []
    else:
        try:
            return [ [ row[ i ] for row in matrix ] for i in range( len( matrix[ 0 ] ) ) ]
        except:
            result = []
            for i in matrix:
                result.append([i])
            return result
        
    
def neural_network():
    #print('run')
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    #model.add(Dense(12, activation='relu', input_shape=(55,)))
    model.add(Dense(50,activation='relu', kernel_initializer='normal', input_shape=(55,)))


    # Add one hidden layer 
    #model.add(Dense(5, activation='relu'))

    # Add an output layer 
    model.add(Dense(1, kernel_initializer='normal'))
    
    #Model Summary 
    #Model output shape
    #model.___________
    
    #Model summary
    #model.__________
    
    #Model config
    #model.get_config()
    
    #List all weight tensors 
    #model.get_weights()
    
    #compile and fit model
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    
    #target = list(dict.fromkeys(target))
    #print(target)
    #print(np.array(target))
    #target = to_categorical(target)
   # k_model = KerasClassifier(build_fn=model, epochs=3, batch_size=5, verbose=0)
    
   # model.fit(np.array(training), np.array(target), epochs=3, batch_size=5, verbose=0)    
    print('done creating the model')
    return model 

def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    seed = 7
    np.random.seed(seed)
    
    #target = to_categorical(target)
   # model = neural_network(training,target)
    k_model = KerasRegressor(build_fn=neural_network, epochs=2000, batch_size=50, verbose=0)
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=seed)
    
    #kfold = KFold(n_splits=10, random_state=seed)
    #results = cross_val_score(k_model, np.array(x_train), np.array(y_train), cv=kfold)
   # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    k_model.fit(np.array(x_train), np.array(y_train))
    print('done fitting the model')
    
    y_pred = k_model.predict(np.array(x_test))
    
    for i in range(len(y_pred)):
        print(y_pred[i], y_test[i])
        
        
    print(mean_squared_error(y_test, y_pred))
    
   # y_score_round = nearestHalf(y_score)
    #print(y_score_round)
    #print(y_test)
    
    
   # print ('accuracy: ', round (accuracy_score(y_test, y_score_round),3)*100, '%')
   # print ('precision: ', round (precision_score(y_test, y_score, average='weighted'),3)*100)
   # print ('recall: ', round (recall_score(y_test, y_score, average='weighted'),3)*100)
   # print ('f1 score: ', round (f1_score(y_test, y_score, average='weighted'),3)*100)
    #print(' ')
    
   
    
   # k_model.fit(np.array(training),np.array(target))
    
    #model = neural_network(training, target)
    #logistic regression 
    #model = LogisticRegression()
    #rfe = RFE(estimator=k_model,n_features_to_select=k, step=1)
    
   # rfe.fit(np.array(x_train),np.array(y_train))
    
    #test
 #   y_score = k_model.predict(np.array(x_test))
   # y_score = rfe.predict(x_test)
   # y_score = ['%.2f' % score for score in y_score]
   # y_test = ['%.2f' % test for test in y_test]
#
   # print(y_test)
   # print(y_score)
  #  print('scores:')
   # print ('accuracy: ', round (accuracy_score(y_test, y_score),3)*100, '%')
   
def nearestHalf(list):
    list2 = []
    for i in list:
        num = round(i * 2) / 2
        if (num > 8.5):
            list2.append(8.5)
        else:
            list2.append(num)
    return list2

def logFunction(list):
    list2 = toFloat(list)
    result = []
    for i in list2:
        try:
            ans = round(math.log(i,10),1)
        except:
            ans = 0.0
        result.append(ans)
        #time.sleep(0.1)
    return result

#convert a list to float 
def toFloat(list):
    list2 = []
    for i in list:
        try:
            list2.append(float(i))
        except:
            #print('error',i)
            [ans] = i
            list2.append(float(ans))
    return list2
############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
utarget= []
file_reader = open('RatiosGrid_test3.csv', "r")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    if(row[:1] != '' and row[3:][0] != ''):
        utarget.append(row[1:2])
        target.append(row[:1])
        training.append(row[3:])
#remove the labelling row 
[names] = training[:1]
training = training[1:]

target = target[1:] 
utarget = utarget[1:]    


############################# PREPROCESS DATA #############################
#data is stored as string rather than float so we have to conver them
#there are some missing data so we handle that by placing 
#999 in that spot 

for i in range(len(training)):
    for j in range(len(training[1])):
        try:
            training[i][j] = float(training[i][j]) 
        except:
            training[i][j] = float(1)
   
 
for i in range(len(utarget)):
    utarget[i] = utarget[i][0]
    
for i in range(len(target)):
    target[i] = target[i][0]
    

for i in range(len(target)):
    target[i] = float(target[i])

targetlog = logFunction(target)  


#normalize data using zscores
#round the number to 3 decimal place
trans = matrixTranspose(training)
newTraining = []

for i in trans:
    zscoreList = stats.zscore(i)
    zscoreList2 = []
    for j in zscoreList:
        num = str(round(j,3))
       # print(num)
        if (num == 'nan'):
            num2 =
            zscoreList2.append(num2)
        else:
            num2 = float(num) 
            zscoreList2.append(num2)
        
    newTraining.append(zscoreList2)

training2 = matrixTranspose(newTraining)



#cross_validation(5,training,targetlog)

#neural_network(training,target)