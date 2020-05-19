# importing requisites
import math 
import random
import numpy as np
from urllib.request import urlretrieve
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


#Decision tree Algorithm
def doDTC(XY,X,Y,class_label_list,num_folds):
    total_class = len(class_label_list)
    #performance measure
    accuracy = 0
    specificity = 0
    sensitivity=0
    precision = 0
    fmeasure = 0
    count_of_a = 0
    count_of_b = 0
    actual_count_of_a=0
    actual_count_of_b=0
    #performance measure by 10 fold operation
    k_fold = KFold(num_folds,True,None)
    for train,test in k_fold.split(XY):
        X_train = [] # train data
        X1_test = [] # test data
        Y_train = [] # train class
        Y1_test = [] # test class
        for index in train:
            X_train.append(X[index])
            Y_train.append(Y[index])
        for index in test:
            X1_test.append(X[index])
            Y1_test.append(Y[index])
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X1_test = np.array(X1_test)
        Y1_test = np.array(Y1_test)
        #classification
        classification = tree.DecisionTreeClassifier()
        classification = classification.fit(X_train,Y_train)        
        predicted_classes = classification.predict(X1_test)
        #prediction result of 10 folds to count 
        predict_count_of_a = count_of_a
        predict_count_of_b = count_of_b
        count_of_a = 0
        count_of_b = 0
        for ele in predicted_classes:
            if ele == class_label_list[0]:
                count_of_a = count_of_a + 1
            else:
                count_of_b = count_of_b + 1
        count_of_a = count_of_a + predict_count_of_a
        count_of_b = count_of_b + predict_count_of_b
        # counting actual class labels for each fold
        result_count_of_a = actual_count_of_a
        result_count_of_b = actual_count_of_b
        actual_count_of_a = 0
        actual_count_of_b = 0
        for ele in Y1_test:
            if ele == class_label_list[0]:
                actual_count_of_a += 1
            else:
                actual_count_of_b += 1
        actual_count_of_a += result_count_of_a
        actual_count_of_b += result_count_of_b
        # Accuracy calculation
        total_test_samples=len(Y1_test)
        accurate = 0
        for i in range(total_test_samples):
            if predicted_classes[i]==Y1_test[i]:
                accurate += 1
        accura = accurate / total_test_samples
        accuracy += accura
        print(" Accuracy:---> ",accuracy)
        print()
        #2*2 matrix for binary classification
        confusion=[]
        # we will be creating a confusion matrix here; classes will be labelled from 0
        for class_label in class_label_list:
            temp = []
            for i in range(total_class):
                #appending
                temp.append(0)
            # counting predicted class
            for i in range(total_test_samples):
                # Atlast found a sample with this class label
                if Y1_test[i] == class_label:
                    # find the predicted class label
                    predicted_label_here = predicted_classes[i]
                    # find the index
                    col_index = class_label_list.index(predicted_label_here)
                    temp[col_index]+=1
            confusion.append(temp)
        #Now check for specificity
        num = confusion[0][0]
        den = confusion[0][0]+confusion[0][1]
        specificity = 0
        if den!=0:
            specify = num/den
        specificity += specify
        print(" Specificity: ",specificity)
        print()
        #Now sensitivity
        num = confusion[1][1]
        den = confusion[1][0]+confusion[1][1]
        sensiy = 0
        if den!=0:
            sensiy = num/den
        sensitivity += sensiy
        print(" Sensitivity: ",sensitivity)
        print()
        #Now precision
        num = confusion[1][1]
        den = confusion[0][1]+confusion[1][1]
        precision = 0
        if den!=0:
            precise = num/den
        precision += precise
        print(" Precision: ",precision)
        print()
        #Now f-measure
        den = precision+sensitivity
        fmeasur = 0
        if den!=0:
            fmeasur = 2*precision*sensitivity/(precision+sensitivity)
        fmeasure += fmeasur
        print(" F-measure: ",fmeasure)
        print()

    #Overall measures for k folds
    accuracy /= num_folds
    specificity /= num_folds
    sensitivity /= num_folds
    precision /= num_folds
    fmeasure /= num_folds
    print("After peforming all operation Results:----> ")
    print()
    print("Ultimate Accuracy: ",accuracy)
    print("Ultimate Specificity: ",specificity)
    print("Ultimate Sensitivity: ",sensitivity)
    print("Ultimate Precision: ",precision)
    print("Ultimate F-measure: ",fmeasure)
    print("Ultimate Prediction: ",count_of_a," ",count_of_b)
    print("Ultimate Actual Class label: ",actual_count_of_a," ",actual_count_of_b)

#Now the thord case i.e. draw ROC curve
def drawROC(XY,X,Y,class_label_list):
    X_new,X1_new,Y_new,Y1_new = train_test_split(X,Y,test_size=0.25,random_state=42)
    # classification
    classification = tree.DecisionTreeClassifier()
    classification = classification.fit(X_new,Y_new)
    # find out required parameters
    predicted_classes = classification.predict_proba(X1_new)
    fpr,tpr,threshold = metrics.roc_curve(Y1_new,predicted_classes[:,1],pos_label=class_label_list[len(class_label_list)-1])
    roc_auc = metrics.auc(fpr,tpr)

    #Plottting in graph
    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Actual Positive Rate')
    plt.xlabel('Not Actual Positive Rate')
    plt.show()

#classification
df=pd.read_csv("haberman.csv")
# overall dataset
XY=np.array(df)
# splitting vertically into class label and attrs
scaler=StandardScaler() 
X=np.array(scaler.fit_transform(df.drop(["class"],1).astype(float)))
Y=np.array(df[['class']].astype(float))
Y_mod=[]
for val in Y:
    Y_mod.append(val[0])
Y=np.array(Y_mod)
# call DTC
doDTC(XY,X,Y,[1,2],10)
drawROC(XY,X,Y,[1,2])


#classification
df=pd.read_csv("wine.csv")
# overall dataset
XY=np.array(df)
# splitting vertically into class label and attrs
scaler=StandardScaler() 
X=np.array(scaler.fit_transform(df.drop(["class"],1).astype(float)))
Y=np.array(df[['class']].astype(float))
Y_mod=[]
for val in Y:
    Y_mod.append(val[0])
Y=np.array(Y_mod)
# call DTC
doDTC(XY,X,Y,[1,2,3],10)
drawROC(XY,X,Y,[1,2,3])


#classification
df=pd.read_csv("bcw.csv")
# overall dataset
XY=np.array(df)
# splitting vertically into class label and attrs
scaler=StandardScaler() 
X=np.array(scaler.fit_transform(df.drop(["class"],1).astype(float)))
Y=np.array(df[['class']].astype(float))
Y_mod=[]
for val in Y:
    Y_mod.append(val[0])
Y=np.array(Y_mod)
# call DTC
doDTC(XY,X,Y,[2,4],10)
drawROC(XY,X,Y,[2,4])
