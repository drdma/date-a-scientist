import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction.text import CountVectorizer	#bayse
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier	#KNN regression
from sklearn.neighbors import KNeighborsRegressor	#KNN regression

from sklearn.cluster import KMeans 		#K-Means clustering

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score 
from sklearn.metrics import confusion_matrix

import time 		#to time segments of code

#Create your df here:
df = pd.read_csv('profiles.csv')

print(df.columns)

# # print(df.job.head())
# # print(df.columns)					
# # print(df.describe())				#describes stats on numerical columns
# # print(df.job.value_counts())		#gives stats on all possible answers

#-----------------------------------------------------------
#Column headings
# - age - continuous
# - body_type - categorical
# - diet - categorical
# - drinks - categorical
# - drugs - categorical
# - education - categorical
# - ethnicity - categorical
# - height - continuous
# - income - discrete
# - job - categorical
# - offspring - categorical/words
# - orientation - categorical (straight, gay, bisexual)
# - pets - categorical
# - religion - categorical/words
# - sex - discrete
# - sign - categorical
# - smokes - categorical
# - speaks - categorical/words
# - status - categorical
#-----------------------------------------------------------

col = df.status

print(col.value_counts())
print(col.describe()) 


#-----------------------------------------------------------
#Processing on short answer essays
#-----------------------------------------------------------

#combine all essay columns to 1 string
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

#Removing the NaNs (regular expressions eg. *.txt = ^.*\.txt$)
all_essays = df[essay_cols].replace(np.nan, '', regex=True)

#Join the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#add column to data frame of essay length (no. of characters?)
df["essay_len"] = all_essays.apply(lambda x: len(x))

#--------------
#add columns for total word count and average word length by:

# 1. Remove non-alphanumeric characters
# 2. Split text into list
# 3. Create new list based on length of each element within the list using list comprehension
# 4. Get mean of new list

df['essays'] = all_essays.str.replace('<br />', '')	#removes <br />
# df.essays = df.essays.apply(lambda x: re.sub(r'\W+', ' ', x))		#removes non-alphanumeric chars
df['essays'] = df.essays.str.split()	#split into list
# df.nwords = len(df.essays)
df['nwords'] = [len(word) for word in df.essays]		#number of words in essays list
df['avg_word_length'] = np.where(df['nwords']==0, 0, df.essay_len/df.nwords)#[0 if x==0 else df.essay_len/df.nwords for x in df.nwords]


#--------------
#count number of occurences of "I", "me", "myself" and save as new column df.me

me = [0]*len(df.essays)
for w in range(len(df.essays)):
	me[w] = df.essays.iloc[w].count('me') + df.essays.iloc[w].count('i') + df.essays.iloc[w].count('myself')

df['me']=me

#-----------------------------------------------------------
#Mapping of categorical data
#-----------------------------------------------------------

#---------------------
#sex mapping answer to numbers
sex_mapping = {'m':0,'f':1}
df['sex_code']=df.sex.map(sex_mapping)

#---------------------
#status mapping answer to numbers
df = df[df.status != 'unknown']

status_mapping = {'single':0,'available':1,'seeing someone':2,'married':3}
df['status_code']=df.status.map(status_mapping)


#---------------------
#keep only data with "graduated"
df = df.loc[df['education'].str.contains('graduated', na=False)]
graduated_mapping = {'graduated from space camp':0,'graduated from high school':1,'graduated from two-year college':2,'graduated from college/university':3,'graduated from masters program':4,'graduated from law school':4,'graduated from med school':5,'graduated from ph.d program':5}
df['graduated_code']=df.education.map(graduated_mapping)

#Drop graduated from space camp category
df = df[df.graduated_code != 0]

#---------------------
#drink mapping answer to numbers
drink_mapping = {'not at all':0,'rarely':1,'socially':2,'often':3,'very often':4,'desperately':5}
df['drinks_code']=df.drinks.map(drink_mapping)

#---------------------
#smoke mapping answer to numbers
smokes_mapping = {'no':0,'trying to quit':1,'when drinking' :2,'sometimes':3,'yes':4}
df['smokes_code']=df.smokes.map(smokes_mapping)

#---------------------
#drugs mapping answer to numbers
drugs_mapping = {'never':0,'sometimes':1,'often' :2}
df['drugs_code']=df.drugs.map(drugs_mapping)

#-----------------------------------------------------------
#Data cleaning
#-----------------------------------------------------------

#Drop outliers in age - entries older than 100
# df = df[df.age < 100]

df = df[df.age < 100]

#Drop outliers in height - entries likely to be in inches, so invalid for less than 50 and more than 90? (5 foot=60in, 7foot = 84in)
df = df[df.height >= 50]
df = df[df.height <= 90]


#Drop outliers in income
df = df[df.income != -1]
df = df[df.income < 1000000]

#---------------------
#drop all NaN rows

df = df.dropna()

#-----------------------------------------------------------
#Find clusters by visualisation
#-----------------------------------------------------------



# x=df.income
# y=df.avg_word_length
# h = df.education

# sns.scatterplot(x, y, hue=h)

# fields = ['age', 'income', 'me', 'education']#,'avg_word_length', 'nwords', 'essay_len', 'me', 'smokes_code', 'drinks_code', 'drugs_code']
# sns.pairplot(df[fields], hue='education', plot_kws={'alpha': 0.7})
# plt.show()


#-----------------------------------------------------------
#CLASSIFICATION
#-----------------------------------------------------------

#-----------------------

fields = ['age', 'income', 'sex_code'] # 'me', 

dataset = df[fields]
labels = df.graduated_code

print(labels.unique())

# split data to training and validation sets
(training_set, validation_set, training_labels, validation_labels) = train_test_split(dataset, labels, train_size=0.8, test_size=0.2, random_state =42)




# #-----------------------
# t0 = time.time()


# #Bayes theroem
# Bayes_classifier = MultinomialNB()
# Bayes_classifier.fit(training_set, training_labels)
# Bayes_guesses = Bayes_classifier.predict(validation_set)

# t1 = time.time()
# time_bayes = t1-t0

# #find accuracies for Bayes theroem
# # Bayes_accuracy = Bayes_classifier.score(validation_set, validation_labels)

# guesses = Bayes_guesses
# Bayes_accuracy = accuracy_score(validation_labels, guesses)
# Bayes_recall = recall_score(validation_labels, guesses, average=None)
# Bayes_precision = precision_score(validation_labels, guesses, average=None)
# Bayes_f1 = f1_score(validation_labels, guesses, average=None)

# # print(confusion_matrix(validation_labels, guesses))

#-----------------------

#Find optimum k for K-nearest neighbour
accuracies=[]
k_list = range(1,150)
for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors=k)

  classifier.fit(training_set, training_labels)
  accuracies.append(classifier.score(validation_set, validation_labels))

KNN_accuracy = max(accuracies)
k_optimum = accuracies.index(KNN_accuracy) +1


t0 = time.time()

classifier = KNeighborsClassifier(n_neighbors=k_optimum)
classifier.fit(training_set, training_labels)
KNN_guesses = classifier.predict(validation_set)

t1 = time.time()
time_KNN = t1-t0

sns.lineplot(k_list, accuracies)
plt.savefig("KNN_kaccuracies.png")

guesses = KNN_guesses
KNN_recall = recall_score(validation_labels, guesses, average=None)
KNN_precision = precision_score(validation_labels, guesses, average=None)
KNN_f1 = f1_score(validation_labels, guesses, average=None)


# print(ct.to_string())	#to print entire output as string

# #Investigate inertia
# num_clusters = list(range(1,15))
# inertias = []

# for i in num_clusters:
#   model = KMeans(n_clusters = i)
#   model.fit(samples)
#   inertias.append(model.inertia_)

# plt.plot(num_clusters, inertias, '-o')
# plt.show()

print('DONE')



# # #-----------------------
# # #K-Means - unsupervised classification

# # k=6

# # KM_model = KMeans(n_clusters=k)

# # KM_model.fit(samples)

# # labels = KM_model.predict(samples)

# # df['labels'] = labels
# # ct = pd.crosstab(df.labels, df.education)


# # print(df.labels)
# # sns.scatterplot(df.age, df.me, hue=df.labels)
# # plt.show()

# # # print(ct.to_string())	#to print entire output as string

# # # #Investigate inertia
# # # num_clusters = list(range(1,15))
# # # inertias = []

# # # for i in num_clusters:
# # #   model = KMeans(n_clusters = i)
# # #   model.fit(samples)
# # #   inertias.append(model.inertia_)

# # # plt.plot(num_clusters, inertias, '-o')
# # # plt.show()

# # print('DONE')

# #-----------------------------------------------------------
# #Find correlations by visualisation
# #-----------------------------------------------------------

# # fields = ['age', 'height', 'avg_word_length', 'nwords', 'essay_len', 'me', 'graduated_code', 'status_code', 'sex', 'status']

# # fields = ['age', 'smokes_code', 'drinks_code', 'drugs_code', 'height', 'sex_code', 'income', 'graduated_code', 'status_code', 'avg_word_length', 'me']

# # sns.set(style="ticks", color_codes=True)

# # g = sns.pairplot(df[fields], kind="reg", height=1.8, aspect=1.2)


# # # pp = sns.pairplot(df[fields], hue='sex_code', 
# # #                   height=1.8, aspect=1.2,
# # #                   plot_kws=dict(edgecolor="k", linewidth=0.5),
# # #                   diag_kws=dict(shade=True), # "diag" adjusts/tunes the diagonal plots
# # #                   diag_kind="kde") # use "kde" for diagonal plots

# # # fig = pp.fig 
# # # fig.subplots_adjust(top=0.93, wspace=0.3)
# # # fig.suptitle('Wine Attributes Pairwise Plots', 
# # #               fontsize=14, fontweight='bold')


# # plt.show()


# #-----------------------------------------------------------
# #LINEAR REGRESSION
# #-----------------------------------------------------------

# #Q1 - can we predict height based on sex and height and age?

# cols = ['sex_code', 'income']#, 'avg_word_length']
# y = df.height


# x = df[cols] 
# (x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state =2)


# lm = LinearRegression()
# model = lm.fit(x_train, y_train)

# y_predict = lm.predict(x_test)

# lm_accuracy = lm.score(x_test, y_test)

# # print(lm.score(x_train, y_train))
# # print(lm.score(x_test, y_test))
# # print(lm.coef_)

# residuals = y_test - y_predict

# sns.scatterplot(y_test, y_predict)
# plt.ylabel('predicted height')
# # sns.scatterplot(y_predict, residuals)
# # sns.line(range())
# plt.savefig('linearregression.png')
# # plt.show()


# print(lm_accuracy)

#-----------------------------------------------------------
#Resuts
#-----------------------------------------------------------

# print(time_bayes)
# print('Bayes_accuracy', Bayes_accuracy)
# print('Bayes_recall', Bayes_recall) 
# print('Bayes_precision', Bayes_precision)
# print('Bayes_f1', Bayes_f1)

print(time_KNN)
print('KNN_accuracy', KNN_accuracy)
print('KNN_recall', KNN_recall) 
print('KNN_precision', KNN_precision)
print('KNN_f1', KNN_f1)

# f = open('Results.txt', 'wb')
# f.write(str(Bayes_accuracy))
# 	# 'Bayes_recall', Bayes_recall, 
# 	# 'Bayes_precision', Bayes_precision,
# 	# 'Bayes_f1', Bayes_f1,
# 	# 'KNN_accuracy', KNN_accuracy,
# 	# 'KNN_recall', KNN_recall, 
# 	# 'KNN_precision', KNN_precision,
# 	# 'KNN_f1', KNN_f1,
	
# f.close()  	
