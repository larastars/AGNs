
import csv 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scipy import stats


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
        
    
def neural_network(training, target):
    print('run')
    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(55,)))

    # Add one hidden layer 
    model.add(Dense(8, activation='relu'))

    # Add an output layer 
    model.add(Dense(1, activation='sigmoid'))
    
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
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.fit(training, target, epochs=20, batch_size=1, verbose=1)
    print('done')

def cross_validation(k,training,target):
    fold = 100/k
    fold = fold/100
    
    #split
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size= fold, random_state=0)
    
    #logistic regression 
    model = LogisticRegression()
    rfe = RFE(model,k)
    
    fit = rfe.fit(x_train, y_train)
    
    #test
    y_score = rfe.predict(x_test)
    print('scores:')
    print ('accuracy: ', round (accuracy_score(y_test, y_score),3)*100, '%')
    print ('precision: ', round (precision_score(y_test, y_score, average='weighted'),3)*100)
    print ('recall: ', round (recall_score(y_test, y_score, average='weighted'),3)*100)
    print ('f1 score: ', round (f1_score(y_test, y_score, average='weighted'),3)*100)
    print(' ')
   

#features to use
k = 10
############################# READ TRAINING DATA #############################
training = []
#read target of training data 
target = []
file_reader = open('RatiosGrid_test3.csv', "r", encoding= "ascii")
read = csv.reader(file_reader)
for row in read:
    #separate training and target
    target.append(row[:1])
    training.append(row[3:])

#remove the labelling row 
[names] = training[:1]
training = training[1:]
target = target[1:]         


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
   
 
for i in range(len(target)):
    target[i] = target[i][0]


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


############################# TRAIN THE DATA #############################
#Feature Extraction 
#model = LogisticRegression()
#rfe = RFE(model,k)

#fit = rfe.fit(training, target)


#print result 
"""
pair = []
for i in range(len(names)):
    pair.append([names[i], fit.ranking_[i], fit.support_[i]])

pair = sorted(pair, key=lambda x: x[1])


############################# PRINTING DATA #############################
print("Num Features: %d"% fit.n_features_) 
for i in range(len(pair)):
    if pair[i][2] == True:
        print(pair[i][0])

"""
#cross_validation(5,training,target)
neural_network(training, target)

