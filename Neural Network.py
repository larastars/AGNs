
import csv 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scipy import stats
from keras.utils import to_categorical
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
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
    print('run')
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    #model.add(Dense(12, activation='relu', input_shape=(55,)))
    model.add(Dense(50,activation='relu', input_shape=(55,)))


    # Add one hidden layer 
    model.add(Dense(25, activation='relu'))

    # Add an output layer 
    model.add(Dense(14, activation='sigmoid'))
    
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
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    #target = list(dict.fromkeys(target))
    #print(target)
    #print(np.array(target))
    #target = to_categorical(target)
   # k_model = KerasClassifier(build_fn=model, epochs=3, batch_size=5, verbose=0)
    
   # model.fit(np.array(training), np.array(target), epochs=3, batch_size=5, verbose=0)    
    print('done')
    return model 

def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    seed = 7
    np.random.seed(seed)
    
   # model = neural_network(training,target)
    k_model = KerasClassifier(build_fn=neural_network, epochs=500, batch_size=5, verbose=0)
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=seed)
    k_model.fit(np.array(training),np.array(target))
    
    #model = neural_network(training, target)
    #logistic regression 
    #model = LogisticRegression()
    #rfe = RFE(estimator=k_model,n_features_to_select=k, step=1)
    
   # rfe.fit(np.array(x_train),np.array(y_train))
    
    #test
    y_score = k_model.predict(np.array(x_test))
   # y_score = rfe.predict(x_test)
    y_score = ['%.2f' % score for score in y_score]
    y_test = ['%.2f' % test for test in y_test]

   # print(y_test)
   # print(y_score)
    print('scores:')
    print ('accuracy: ', round (accuracy_score(y_test, y_score),3)*100, '%')
   # print ('precision: ', round (precision_score(y_test, y_score, average='weighted'),3)*100)
   # print ('recall: ', round (recall_score(y_test, y_score, average='weighted'),3)*100)
   # print ('f1 score: ', round (f1_score(y_test, y_score, average='weighted'),3)*100)
   # print(' ')
   

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
     
"""

#normalize data using zscores
#round the number to 3 decimal place
trans = matrixTranspose(training)
newTraining = []

for i in trans:
    zscoreList = stats.zscore(i)
    zscoreList2 = []
    for j in zscoreList:
        num = str(round(j,3))
        num2 = float(num)
        zscoreList2.append(num2)
        
    newTraining.append(zscoreList2)

training2 = matrixTranspose(newTraining)

"""
cross_validation(5,training,target)

#neural_network(training,target)