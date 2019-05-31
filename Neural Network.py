
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_regression
from scipy import spatial
from scipy import stats
from matplotlib import pyplot as plt
import warnings
import math
warnings.filterwarnings('always') 
warnings.filterwarnings('ignore')

global n 
#cosine similarity as a feature selection method 
def cosineSimilarity(training, names, n):
    vectorSet = matrixTranspose(training)
    simSet = []
    #find pairwise cosine similarity

    for i in vectorSet:
        sum = 0
        for j in vectorSet:
            sim = 1 - spatial.distance.cosine(i, j)
            sum += sim
        simSet.append(sum)
       
    sortedSim, sortedNames = zip(*sorted(zip(simSet, names), reverse = True))
    print(sortedSim)
    sortedNames = sortedNames[:n]
   
    return sortedNames
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
        
    
def neural_network_regressor():

    print('Building the model')
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(40,activation='relu', kernel_initializer='normal', input_shape=(n,)))


    # Add one hidden layer 
    model.add(Dense(15, activation='tanh'))
    
    # Add an output layer 
    model.add(Dense(1, kernel_initializer='normal'))
    
    
    #List all weight tensors 
    #model.get_weights()
    
    #compile and fit model
    model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
    print('Done building the model')
    
    return model 

def cross_validation_regressor(k,training,target):
    #folds 
    fold = 100/k
    fold = fold/100
    
    seed = 7
    np.random.seed(seed)
    
    print('building the regressor')
    #build a regressor
    k_model = KerasRegressor(build_fn=neural_network_regressor, epochs=15000, batch_size=30, verbose=0)
    mse = 0
    accuracy = 0
    #for i in range(k):
        #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=seed)
        
    #plot
    #learning_curve(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), neural_network())
    
    print('fitting the regressor')
    #fit the model 
    k_model.fit(np.array(x_train), np.array(y_train))

    #make a prediction 
    y_pred = k_model.predict(np.array(x_test))
        
    
    #print comparision
    for i in range(len(y_pred)):
        print(round(y_pred[i],1), y_test[i])
          
    #print mse
    #print('mse: ', mean_squared_error(y_test, y_pred))
    mse += mean_squared_error(toFloat(y_test), toFloat(y_pred))
        
    #prepare for accuracy 
    y_pred_round = nearestHalf(y_pred)
        
        
    #change data to string values 
    y_pred_round = ['%.2f' % score for score in y_pred_round]
    y_test = ['%.2f' % test for test in y_test]
        
    accuracy += accuracy_score(y_test, y_pred_round)
        #accuracy 
        #print ('accuracy: ', round (accuracy_score(y_test, y_pred_round),3)*100, '%')
    #print(i)
        
   # print('mse: ', (mse/k))
   # print ('accuracy: ', round (accuracy/k,3)*100, '%')
    print('mse: ', mse)
   # print ('accuracy: ', round (accuracy,3)*100, '%')


def neural_network_classifier():
    #print('run')
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    #model.add(Dense(12, activation='relu', input_shape=(55,)))
    model.add(Dense(50,activation='relu', kernel_initializer='normal', input_shape=(n,)))


    # Add one hidden layer 
    model.add(Dense(5, activation='relu'))

    # Add an output layer 
    model.add(Dense(14, kernel_initializer='normal'))
    
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
    #k_model = KerasClassifier(build_fn=model, epochs=3, batch_size=5, verbose=0)
    
    #model.fit(np.array(training), np.array(target), epochs=3, batch_size=5, verbose=0)    
    print('done creating the model')
    return model 


def cross_validation_classifier(k,training,target):
    #folds 
    fold = 100/k
    fold = fold/100
    
    seed = 7
    np.random.seed(seed)
    
    #build a regressor
    k_model = KerasClassifier(build_fn=neural_network_classifier, epochs=20000, batch_size=30, verbose=0)
    mse = 0
    accuracy = 0
  #  for i in range(k):
        #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=seed)
        
        #fit the model 
    k_model.fit(np.array(x_train), np.array(y_train))
        
        #make a prediction 
    y_pred = k_model.predict(np.array(x_test))
        
        #print comparision
        #for i in range(len(y_pred)):
         #   print(y_pred[i], y_test[i])
          
        #print mse
        #print('mse: ', mean_squared_error(y_test, y_pred))
    mse += mean_squared_error(y_test, y_pred)
        
        #prepare for accuracy 
    y_pred_round = nearestHalf(y_pred)
        
        
        #change data to string values 
    y_pred_round = ['%.2f' % score for score in y_pred_round]
    y_test = ['%.2f' % test for test in y_test]
        
    accuracy += accuracy_score(y_test, y_pred_round)
        #accuracy 
        #print ('accuracy: ', round (accuracy_score(y_test, y_pred_round),3)*100, '%')
        #print(i)
        
    #print('mse: ', (mse/k))
    #print ('accuracy: ', round (accuracy/k,3)*100, '%')
    print('mse: ', mse)
    print ('accuracy: ', round (accuracy,3)*100, '%')

def nearestHalf(list):
    list2 = []
    for i in list:
        num = abs(round(i * 2) / 2)
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


def removeData(training, names):
    #if the average = first value 
    training2 = matrixTranspose(training)
    names2 = matrixTranspose(names)
    
    training3 = []
    names3 = []
    
    for i in range(len(training2)):
        avg = sum(training2[i])/ float(len(training2[i]))
        if (avg != training2[i][0]):
            training3.append(training2[i])
            names3.append(names2[i])
          
    training4 = matrixTranspose(training3)
    names4 = matrixTranspose(names3)
    [names5] = names4
   

    return training4, names5


def indpData(training, names, indpSet):
    trainingTrans = matrixTranspose(training)
    newTrainTrans = []
    for i in range(len(names)):
        if (names[i] in indpSet):
            newTrainTrans.append(trainingTrans[i])
            
    newTrain = matrixTranspose(newTrainTrans)
    return newTrain

def select_k_best(trainig, target, names):
    #We will select the features using chi square
    test = SelectKBest(mutual_info_regression,k='all')
    
    #Fit the function for ranking the features by score
    fit = test.fit(training, target)
    
    #Summarize scores numpy.set_printoptions(precision=3) print(fit.scores_)
    #Apply the transformation on to dataset
    features = fit.transform(training)
    
    #Summarize selected features print(features[0:5,:])
    np.set_printoptions(precision=3) 
    scores_list = fit.scores_

    #create a list of pairs [feature, score]
    scores_list, names2 = zip(*sorted(zip(scores_list, names), reverse= True))
    
    training2 = indpData(training, names, names2[:n])
    return training2, names2[:n]
    

def boxcox(training):
    trainingTrans = matrixTranspose(training)
    newTrainTrans = []
    for i in trainingTrans:
        new = stats.boxcox(i)
        newTrainTrans.append(new)
    
    return matrixTranspose(newTrainTrans)

def learning_curve(train_features, train_labels, test_features, test_labels, model):
    test_accuracy = [];
    test_loss = [];
    train_accuracy = [];
    train_loss = [];
      
    increment = 64
    #split data and train some of it
    chunks_train_data = [train_features[x:x+increment] for x in range(0, len(train_features), increment)]
    #split labels and train some of it 
    chunks_train_labels = [train_labels[x:x+increment] for x in range(0, len(train_features), increment)]
    
    #from 0-5
    for epoch in range(0, 5):
        #from 0 to the length of chunks_train_data
        for i, el in enumerate(chunks_train_data):
            #print(i)
            train_loss_and_metrics = model.train_on_batch(el, chunks_train_labels[i])
            #print(train_loss_and_metrics)
            train_loss.append(train_loss_and_metrics[0])
            train_accuracy.append(train_loss_and_metrics[1])
            test_loss_and_metrics = model.evaluate(test_features, test_labels, batch_size=128)
            #print(test_loss_and_metrics)
            test_loss.append(test_loss_and_metrics[0])
            test_accuracy.append(test_loss_and_metrics[1])
    
    
    fig = plt.figure()
    ##visualize the learning curve  
    ax1 = fig.add_subplot(211)
    ax1.plot(train_loss)
    ax1.plot(test_loss)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Iteration')
    ax1.legend(['training', 'testing'], loc='upper left')

    ax2 = fig.add_subplot(212)
    ax2.plot(train_accuracy)
    ax2.plot(test_accuracy)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Iteration')
    ax2.legend(['training', 'testing'], loc='upper left')
    
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
            training[i][j] = 0
   
 
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
            num2 =0
            zscoreList2.append(num2)
        else:
            num2 = float(num) 
            zscoreList2.append(num2)
        
    newTraining.append(zscoreList2)

training2 = matrixTranspose(newTraining)

#remove unnessecary data 
training3, names2 = removeData(training2, names)

#find n most indepent columns
n = 51
#indpSet = cosineSimilarity(training3, names2, n)

#training4 = indpData(training3, names2, indpSet)

cross_validation_regressor(5,training3,targetlog)

#neural_network(training,target)