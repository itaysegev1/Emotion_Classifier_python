import math
import numpy as np
import pandas as pd
from tabulate import tabulate



# the function distance created for the knn
# the function return the length between 2 sentences.
def distance(test, train):
    x = set(test.lower().split())
    y = set(train.lower().split())
    x1 = x.union(y)
    y1 = x.intersection(y)
    return len(x1 - y1) / len(x1)


# the function add 2 columns to the data frame, one of knn and one of nb(naive base)
def addcolumn(testing_data):
    testing_data["knn"] = 0
    testing_data["nb"] = 0


# the function calculate the probability of a sentence from the testing data
# to be every emotion in the pk
# the function return the emotion that the sentence was max odds it will be in.
def probality(document, pk, pki, ptot, word_dict):
    words = set(document.lower().split())
    class_log_p = {}
    for class_label in pk:
        class_log_p[class_label] = math.log((pk[class_label] + 1) / (ptot + 2))
        # add log probabillity for each word:
        for word in words:
            if word in word_dict:
                word_prob = (pki[word][class_label] + 1) / (pk[class_label] + 2)
                class_log_p[class_label] += math.log(word_prob)
    return max(class_log_p, key=class_log_p.get)


# nb function run on all the testing data row by row and add to the nb column in the row which emotion
# return from the probability function
def naivebase(testing_data, pk, pki, ptot, word_dict):
    for index, row in testing_data.iterrows():
        testing_data.loc[index, "nb"] = probality(row['Comment'], pk, pki, ptot, word_dict)


# knn function run all the testing data row by row and add to the knn column in the row which emotion
# return from the pknn function
def knn(testing_data, training_data):
    for index, row in testing_data.iterrows():
        doc = row['Comment']
        testing_data.loc[index, "knn"] = pknn(doc, training_data)


# pknn function is helping function for the knn
# it goes over all the training data and check the distance between the actual sentence to the sentece from the training
# data. the function take the 15 nearest neighbors and return the emotion of the max of them.
def pknn(doc, training_data):
    distances = []
    for i, row in training_data.iterrows():
        distances.append(distance(doc, row["Comment"]))
    topk = np.argsort(distances)[:15]
    knearest = {}
    for x in topk:
        a = training_data.iloc[x]
        if a["Emotion"] not in knearest:
            knearest[a["Emotion"]] = 1
        else:
            knearest[a["Emotion"]] += 1
    return max(knearest, key=knearest.get)


# clean_data function remove all the row that the data have missing columns.
def clean_data(df):
    df.dropna(axis=0)


# dictionary function creates set of words that appears in all the training data.
# the function return the word_dict
def dictionary(training_data):
    word_dict = set()
    for _, row in training_data.iterrows():
        words = row['Comment'].lower().split()
        for word in words:
            word_dict.add(word)
    return word_dict


# pk function goes over all the training data and count how many rows there are of every emotion
# the function return a dictionary that the key is the emotion and the value is how many there is from that emotion
def pk(training_data):
    pk = {}
    for _, row in training_data.iterrows():
        if row['Emotion'] not in pk:
            pk[row['Emotion']] = 1
        else:
            pk[row['Emotion']] += 1
    return pk


# pki function creates and return a pki:
# explanation of pki:
# the pki is dictionary of dictionaries that the word from the word_dict is the key and the value is dictionary
# that dictionary have 3 keys which they are the emotion and the value is how many that word appears in a sentence
# of that emotion
# pki function have some steps:
# first step: creates the pki: with empty values( each inner value will be 0)
# second step: fill the pki: go over each row and add values to the pki.
def pki(training_data, pk, word_dict):
    pki = {}
    for word in word_dict:
        pki[word] = {}
        for class_label in pk:
            pki[word][class_label] = 0
    for _, row in training_data.iterrows():
        words = set(row['Comment'].lower().split())
        for word in words:
            pki[word][row['Emotion']] += 1
    return pki


# The recall function measure by calculating number of true guesses of a specific emotion
# divided by how many times the emotion exist
def recall(testing_data, pk):
    results = []
    for class_label in pk:
        nb = 0
        knn1 = 0
        real = 0
        for index, row in testing_data.iterrows():
            if row["Emotion"] == class_label:
                real += 1
                if row["Emotion"] == row["nb"]:
                    nb += 1
                if row["Emotion"] == row["knn"]:
                    knn1 += 1
        results.append(round((nb / real) * 100, 3))
        results.append(round((knn1 / real) * 100, 3))
    return results


# The precision function measure by calculate number of true guesses of a specific emotion
# divided by how many time it classified as this emotion
def precision(testing_data, pk):
    results = []
    for class_label in pk:
        truenb = 0
        trueknn = 0
        clasinb = 0
        clasiknn = 0
        for index, row in testing_data.iterrows():
            if row["nb"] == class_label:
                clasinb += 1
            if row["knn"] == class_label:
                clasiknn += 1
            if row["Emotion"] == class_label:
                if row["Emotion"] == row["nb"]:
                    truenb += 1
                if row["Emotion"] == row["knn"]:
                    trueknn += 1
        results.append(round((truenb / clasinb) * 100, 3))
        results.append(round((trueknn / clasiknn) * 100, 3))
    return results


# the F_measure function measure by calculating by this formula:
# 2 * (precision * recall) divided by (precision + recall)
def F_measure(reresults, prresults):
    results = []
    for i in range(6):
        x = 2 * (reresults[i] * prresults[i])
        y = reresults[i] + prresults[i]
        results.append(round((x / y), 3))
    return results


df = pd.read_csv('Emotion_classify_Data.csv')
# clean the data
if df.isnull().sum().sum() == 0:
    clean_data(df)
# step 1: split the data set to train and test
size = len(df)
trsize = int(0.7 * size)  # the size of training data
training_data = df.loc[:trsize].copy()  # split the data to training and testing
testing_data = df.loc[trsize + 1:].copy()  # copy for add ind rows of knn and nb(naive base)
addcolumn(testing_data)
# step 2:calculate the total number of options in the training data
ptot = len(training_data)
# step 3: count how many options every class has
pk = pk(training_data)
# step 4: create a dictionary with all unique word from all documents
word_dict = dictionary(training_data)
# step 5: create pki
pki = pki(training_data, pk, word_dict)
# step6: do the naive base and the knn
naivebase(testing_data, pk, pki, ptot, word_dict)
knn(testing_data, training_data)
# step 7: count the result of the accuracy on the knn and naive bays, by counting how many times it
# from all the data
nb = 0
knn1 = 0
for index, row in testing_data.iterrows():
    if row["Emotion"] == row["nb"]:
        nb += 1
    if row["Emotion"] == row["knn"]:
        knn1 += 1
print("The data frame after adding columns and fill them with the data")
print(testing_data)
print("The results on the testing data:")
print("the accuracy of naive base to sucssed is : ", round((nb / len(testing_data)) * 100, 3), "%")
print("the accuracy of knn to sucssed is : ", round((knn1 / len(testing_data) * 100), 3), "%")
# step 8: calculate all the results on the testing by using the recall, precision, and F_measure functions
rec = recall(testing_data, pk)
pre = precision(testing_data, pk)
fm = F_measure(rec, pre)
# step 9: adding each result which kind of result is it
rec.insert(0, "recall  ")
pre.insert(0, "precision  ")
fm.insert(0, "F-measure  ")
# step 10: put all the results in one list
datax = [rec, pre, fm]
labels_values = ["", "fear:'naive bays'", "fear:'knn'", "anger: 'naive bays'", "anger: 'knn'", "joy:'naive bays'",
                 "joy:'knn'"]
# step 11: make and print a table of the rusult by usint the tabulate
print(tabulate(datax, tablefmt="fancy_grid", headers=labels_values))
################################################
# repeating steps 6-11 on the training data
addcolumn(training_data)
naivebase(training_data, pk, pki, ptot, word_dict)
knn(training_data, training_data)

nb = 0
knn1 = 0
for index, row in training_data.iterrows():
    if row["Emotion"] == row["nb"]:
        nb += 1
    if row["Emotion"] == row["knn"]:
        knn1 += 1
print("The data frame after adding columns and fill them with the data")
print(training_data)
print("the result on the training data")
print("the accuracy of naive base to sucssed is : ", round((nb / len(training_data) * 100), 3), "%")
print("the accuracy of knn to sucssed is : ", round((knn1 / len(training_data) * 100), 3), "%")
rec = recall(training_data, pk)
pre = precision(training_data, pk)
fm = F_measure(rec, pre)
rec.insert(0, "recall  ")
pre.insert(0, "precision  ")
fm.insert(0, "F-measure  ")
datax = [rec, pre, fm]
print(tabulate(datax, tablefmt="fancy_grid", headers=labels_values))
