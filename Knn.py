import collections
import pandas as pd
import numpy as np
data_train = pd.read_csv('TrainData.txt', sep=",",header=None)
data_test = pd.read_csv('TestData.txt', sep=",",header=None)

def euclideanDistance(data1,data2,length):
    distance=0
    for x in range(length):
        distance+=np.square(data1[x]-data2[x])
    return np.sqrt(distance)

# Defining our KNN model
def knn(train,test,k):
    
    
    length=test.shape[1]-1
    result=[]
    # calculating euclideanDistance between each row of training data and test data
    for y in range(len(test)):
        distances=[]
        
        for x in range(len(train)):
            dist=euclideanDistance(test.iloc[y],train.iloc[x],length)
            distances.append(dist)
           
        neighbors=[]
        
        ''' map distances to index of rows in dataframe (train) '''
        
        mapped = dict(zip(distances,train.index.values))
        
        od = collections.OrderedDict(sorted(mapped.items()))
                
        neighbors = list(od.values())[:k]                
        
        classvotes=[0,0,0,0,0,0,0,0,0,0]
        
        specifies = list(train[8].unique())
        
        specifies_map = [0,1,2,3,4,5,6,7,8,9] 
        dict_specifies_map = dict(zip(specifies,specifies_map ))
        for x in range(len(neighbors)):
            response = train.iloc[neighbors[x],-1]
            response = dict_specifies_map[response]
            if response in classvotes:
                classvotes[response]+=1
            else:
                classvotes[response]=1
        
        max_vote =  max(classvotes)
        counter_for_max_vote = 0
        for i in range(0,len(classvotes)):
            if max_vote == classvotes[i]:
                counter_for_max_vote+=1
        
        if (counter_for_max_vote > 2):
                result.append(train.iloc[0,-1])
        else :
            single_output = classvotes.index(max(classvotes))
            
            #result.append(classvotes.index(max(classvotes)))
            for key,value in dict_specifies_map.items():
                if single_output == value :
                    result.append(key)
                
    return (result)

print("         When K is 1 ")
Predicted_class_1=knn(data_train,data_test,1)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_1[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 2 ")
Predicted_class_2=knn(data_train,data_test,2)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_2[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 3 ")
Predicted_class_3=knn(data_train,data_test,3)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_3[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 4 ")
Predicted_class_4=knn(data_train,data_test,4)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_4[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 5 ")
Predicted_class_5=knn(data_train,data_test,5)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_5[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 6 ")
Predicted_class_6=knn(data_train,data_test,6)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_6[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 7 ")
Predicted_class_7=knn(data_train,data_test,7)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_7[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################

print("         When K is 8 ")
Predicted_class_8=knn(data_train,data_test,8)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_8[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################
print("         When K is 9 ")
Predicted_class_9=knn(data_train,data_test,9)
Actual_class = list(data_test[8])
counter = 0
for i in range(len(Actual_class)):
    if Actual_class[i] == Predicted_class_9[i]:
        counter+=1
        
print('Number of correctly classified instances : ', counter)
print('Total number of instances :',len(Actual_class))
print('Accuracy :',counter /len(Actual_class))
###############################################################################