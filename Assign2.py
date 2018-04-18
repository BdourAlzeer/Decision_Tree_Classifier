#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
import collections

#Code fragments provided by Raj Patel

#Load your dataset here
input_file = csv.DictReader(open('data.csv'))
all_data = list(input_file)

#DataPreprocessing
#Hybraid Data Preprocessing
def DicreteVariablesToBoolean(filename):
    data = pd.read_csv(filename)
    country_code = {'CAN':[],'EURO':[],'USA':[]}
    position = {'R': [], 'C': [], 'L': [],'D':[]}
    for e in data['country_group']:
        if e == 'CAN':
            country_code['CAN'].append(1)
        else:
            country_code['CAN'].append(0)
        if e == 'USA':
            country_code['USA'].append(1)
        else:
            country_code['USA'].append(0)
        if e == 'EURO':
            country_code['EURO'].append(1)
        else:
            country_code['EURO'].append(0)
    for e in data['Position']:
        if e == 'C':
            position['C'].append(1)
        else:
            position['C'].append(0)
        if e == 'R':
            position['R'].append(1)
        else:
            position['R'].append(0)
        if e == 'L':
            position['L'].append(1)
        else:
            position['L'].append(0)
        if e == 'D':
            position['D'].append(1)
        else:
            position['D'].append(0)
    data['CAN']=country_code['CAN']
    data['EURO'] = country_code['EURO']
    data['USA'] = country_code['USA']
    data['R'] = position['R']
    data['L'] = position['L']
    data['C'] = position['C']
    data['D'] = position['D']
    del data['country_group']
    del data['Position']
    return data


#Create your Features "x" and Target Values "targets" here from dataset
#preprocessing steps, Add quadratic
def Add_quadraticInteractionTerms(data):

    columnsNames = list(data)
    columnsCount = len(columnsNames)

    for i in range(columnsCount-1):
        for j in range(i+1, columnsCount):
            data[columnsNames[i]+"*"+ columnsNames[j]] = data[columnsNames[i]] *  data[columnsNames[j]]
    return data

means=[]
stds=[]

#preprocessing steps, standardize features
def SubtractMeanSTD(data):
    columnsNames = list(data)
    for i in range(len(columnsNames)-21):
        if i not in range(15,22):
            data[columnsNames[i]] -= data[columnsNames[i]].mean()
            data[columnsNames[i]] /= data[columnsNames[i]].std()
    return data

# def SubtractMeanSTDTesting(Testdata):
#     columnsNames = list(Testdata)
#     for i in range(len(columnsNames)-21):
#         Testdata[columnsNames[i]] -= means[i]
#         Testdata[columnsNames[i]] /= stds[i]
#     return Testdata

#Here write a function that takes as input a weight vector and outputs the squared-error loss (Validation error)
#Calculating the sqaured error
def ErrorSquared(data, weightVector, output_column):
    error = 0
    X_matrix = data.as_matrix()
    y_matrix = output_column.as_matrix()

    for i in range(len(y_matrix)):
        error += (y_matrix[i]-(np.dot(weightVector,X_matrix[i])))**2
    return error/len(y_matrix)
    #return error



#Here write a function that takes as input a lambda value and outputs the optimal weight vector
#def calc_weights()...
def WightVector(data, lambdas, output_column):

    X_matrix= data.as_matrix()
    y_matrix = output_column.as_matrix()
    X_transpose = X_matrix.transpose()

    X_transpose_Mul_X = np.matmul(X_transpose,X_matrix)
    Identity_ = np.identity(len(X_transpose))
    Lamda_Identity = np.dot(Identity_,lambdas)
    X_transpose_Mul_X_Plus_LamdaIdentity = Lamda_Identity+X_transpose_Mul_X
    Inverse_Sum= np.linalg.inv(X_transpose_Mul_X_Plus_LamdaIdentity)
    XTransposeMul_Y = np.matmul(X_transpose,y_matrix)
    Inverse_Sum_Mul_XTransposeMul_Y = np.matmul(Inverse_Sum,XTransposeMul_Y)
    weight_vector = Inverse_Sum_Mul_XTransposeMul_Y

    #weight_vector = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_transpose,X_matrix) + np.dot(np.identity(len(X_transpose)),lambdas)), X_transpose),y_matrix)
    #ErrorSquared(data,weight_vector,output_column)

    return weight_vector


if __name__ =="__main__":

    lambdas = [0.01, 0.1, 1, 10, 100, 1000]
    data = DicreteVariablesToBoolean("data.csv")
    test_data_2007 = DicreteVariablesToBoolean("testing_data_2007.csv")
    data= Add_quadraticInteractionTerms(data)
    data = SubtractMeanSTD(data)
    y_vector_testing = test_data_2007['sum_7yr_GP']
    del test_data_2007['sum_7yr_GP']
    test_data_2007= Add_quadraticInteractionTerms(test_data_2007)
    test_data_2007 = SubtractMeanSTD(test_data_2007)
    data_df = pd.read_csv('Data_7YearsSum.csv')
    y_vector = data_df['sum_7yr_GP']
    validationErrorSet={}
    k_folds = 10
    test_data_size= len(data)/k_folds
    train_data_size= len(data)-test_data_size
    test_data_Errors_Dict ={i:0.0 for i in lambdas}
    train_data_Errors_Dict ={i:0.0 for i in lambdas}
    weight_vectors_list=[]
    errors_2007_testing=[]

# K-folds implementation
    for l in lambdas:
        for i in range(10):
           test_data= data.iloc[i*test_data_size:(i+1)*test_data_size]
           train_data = data.drop(data.index[i*test_data_size:(i+1)*test_data_size])

           output_col_test= y_vector.iloc[i*test_data_size:(i+1)*test_data_size]
           output_col_train= y_vector.drop(y_vector.index[i*test_data_size:(i+1)*test_data_size])

           weight_vector = WightVector(train_data,l,output_col_train)
           weight_vectors_list.append(weight_vector)

           train_data_Errors_Dict[l]+= (ErrorSquared(train_data, weight_vector, output_col_train))
           test_data_Errors_Dict[l]+= (ErrorSquared(test_data, weight_vector, output_col_test))

        train_data_Errors_Dict[l]/=10
        test_data_Errors_Dict[l]/=10

    for l in lambdas:
        weight_vector_2007 = WightVector(data, l, y_vector)
        errors_2007_testing.append(ErrorSquared(test_data_2007,weight_vector_2007,y_vector_testing))

#To store Validation errors (Square error) for different values of lambda
#returns the lamda with least error
BestLamdaKey = test_data_Errors_Dict.keys()[test_data_Errors_Dict.values().index(min(test_data_Errors_Dict.values()))]
BestLamda= min(test_data_Errors_Dict.values())
#Find the best value of Error from Validation error set
print "Lamda: ", BestLamdaKey," With Error Value: ",BestLamda

weight_vector_2007 = WightVector(data, BestLamdaKey, y_vector)
best_error_2007=(ErrorSquared(test_data_2007, weight_vector_2007, y_vector_testing))

#Print the affect of the quadratic interaction terms
BestVector_BestLamda= weight_vector_2007
i=0
ColumnWeightVectorList=[]
for col in list(data):
    ColumnWeightVectorList.append((col, abs(BestVector_BestLamda[i])))
    i+=1
ColumnWeightVectorList=sorted(ColumnWeightVectorList,key=lambda tup:tup[1])

for element in ColumnWeightVectorList:
    print element

#Produce a plot of results.
#Plot of the two curves, the training set and the testing set results
lmb = "Best Lambda: "+ str(BestLamdaKey)
error = "Error at Best Lambda: %.4f"%BestLamda
test_data_Errors_Dict = collections.OrderedDict(sorted(test_data_Errors_Dict.items()))
plt.semilogx(lambdas, test_data_Errors_Dict.values(), label='Training Errors on 2004, 2005, 2006 (K-Folds)')
plt.semilogx(lambdas, errors_2007_testing, label='Testing Errors on 2007')
plt.semilogx(BestLamdaKey,BestLamda,marker='o',color='r',label="Best Error 2004, 2005, 2006")
plt.semilogx(BestLamdaKey,best_error_2007,marker='o',color='b',label="Best Error 2007")
plt.ylabel('Sum Squared Error')
plt.text(5, 116, lmb, fontsize=15)
plt.text(5, 109, error, fontsize=15)
plt.legend()
plt.xlabel('Lambda')
plt.show()

#Plot of the training set results
lmb = "Best Lambda: "+ str(BestLamdaKey)
error = "Error at Best Lambda: %.4f"%BestLamda
test_data_Errors_Dict = collections.OrderedDict(sorted(test_data_Errors_Dict.items()))
plt.semilogx(lambdas, test_data_Errors_Dict.values(), label='Training Errors on 2004, 2005, 2006 (K-Folds)')
plt.semilogx(BestLamdaKey,BestLamda,marker='o',color='r',label="Best Error 2004, 2005, 2006")
plt.ylabel('Sum Squared Error')
plt.text(5, 116, lmb, fontsize=15)
plt.text(5, 109, error, fontsize=15)
plt.legend()
plt.xlabel('Lambda')
plt.show()

#Plot of the testing set results
lmb = "Best Lambda: "+ str(BestLamdaKey)
error = "Error at Best Lambda: %.4f"%BestLamda
plt.semilogx(lambdas, errors_2007_testing, label='Testing Errors on 2007')
plt.semilogx(BestLamdaKey,best_error_2007,marker='o',color='b',label="Best Error 2007")
plt.ylabel('Sum Squared Error')
plt.text(5, 116, lmb, fontsize=15)
plt.text(5, 109, error, fontsize=15)
plt.legend()
plt.xlabel('Lambda')
plt.show()

