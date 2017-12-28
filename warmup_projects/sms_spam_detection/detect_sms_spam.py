# In this file, a Multinomial Naive Bayes classifier was trained, and then used to  
# predict whether an SMS message is a spam or not. It was tested with Anaconda 3.6.3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

# Read in the data set
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])
print(df.head(), '\n')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.sms_message, df.label, random_state=1)

print('Number of rows in the total set:', df.shape[0])
print('Number of rows in the training set:', X_train.shape[0])
print('Number of rows in the test set:', X_test.shape[0])

# Compute the word frequency matrix for training data
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

# print('Feature Names:', count_vector.get_feature_names())
# print('Training Data:'); print(training_data)
# print('Testing Data:'); print(testing_data)

# Use MultinomialNB classifier and train it with training data set
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

print('\nModel:', naive_bayes, '\n')

# Test and evaluate the model with testing data set
y_predicted = naive_bayes.predict(testing_data)

#print('y_test:', y_test)
#print('y_predicted:', y_predicted)

print('Accuracy score: %.3f' % (accuracy_score(y_test, y_predicted)))
print('Precision score: %.3f' % (precision_score(y_test, y_predicted, pos_label='spam')))
print('Recall score: %.3f' % (recall_score(y_test, y_predicted, pos_label='spam')))
print('F1 score: %.3f' % (f1_score(y_test, y_predicted, pos_label='spam')))
print('F-beta score: %.3f' % (fbeta_score(y_test, y_predicted, 0.5, pos_label='spam')))
