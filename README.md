# Spam Filtering based on Naive Bayes Classication

We will implement a simple machine learning algorithm to classify emails (Spam or Not Spam aka Ham) using Scikit-learn. We are going to implement a Spam filtering classifier based on Multinomial Naive Bayes Algorithm


## Prerequisites

To follow this doc you need to have the basic ideas of the following
* Machine Learning
* numpy (http://www.numpy.org/), a powerful N-dimensional array object
* Multinomial Naive Bayes Algorithm (https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/)

We are going to solve this spam filtering problem in three different sections
* Prepare the data
* Build the training data
* Train the classifier
* Test the classifier for its accuracy
* Classify a new email with our trained classifer

## Prepare the data

Download the publicly available Lingspam dataset (http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz).
We are going to use the `bare > part1` from the downloaded dataset as the trainig data and `bare > part2` as the test data. Create two new directories, rename them - train_data and test_data. Now copy all the files from `bare > part1` to train_data and copy some random files from `base > part2` to test_data.

```
import os
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB

# Creates dictionary from all the emails in the directory
def build_dictionary(dir):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # Array to hold all the words in the emails
  dictionary = []

  # Collecting all words from those emails
  for email in emails:
    m = open(os.path.join(dir, email))
    for i, line in enumerate(m):
      if i == 2: # Body of email is only 3rd line of text file
        words = line.split()
        dictionary += words

  # We now have the array of words, whoch may have duplicate entries
  dictionary = list(set(dictionary)) # Removes duplicates

  # Removes puctuations and non alphabets
  for index, word in enumerate(dictionary):
    if (word.isalpha() == False) or (len(word) == 1):
      del dictionary[index]

  return dictionary

def build_features(dir, dictionary):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # ndarray to have the features
  features_matrix = np.zeros((len(emails), len(dictionary)))

  # collecting the number of occurances of each of the words in the emails
  for email_index, email in enumerate(emails):
    m = open(os.path.join(dir, email))
    for line_index, line in enumerate(m):
      if line_index == 2:
        words = line.split()
        for word_index, word in enumerate(dictionary):
          features_matrix[email_index, word_index] = words.count(word)

  return features_matrix

def build_labels(dir):
  # Read the file names
  emails = os.listdir(dir)
  emails.sort()
  # ndarray of labels
  labels_matrix = np.zeros(len(emails))

  for index, email in enumerate(emails):
    labels_matrix[index] = 1 if re.search('spms*', email) else 0

  return labels_matrix

train_dir = '/home/alameen/Desktop/sklearn/spam_ham_classifier/data/train-mails'
dictionary = build_dictionary(train_dir)
features_train = build_features(train_dir, dictionary)
labels_train = build_labels(train_dir)

classifier = MultinomialNB()
classifier.fit(features_train, labels_train)

test_dir = '/home/alameen/Desktop/sklearn/spam_ham_classifier/data/test-mails'
features_test = build_features(test_dir, dictionary)
labels_test = build_labels(test_dir)

accuracy = classifier.score(features_test, labels_test)
print(accuracy)

```

### References
* https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/
* https://appliedmachinelearning.wordpress.com/2017/01/23/email-spam-filter-python-scikit-learn/
* http://www.tutorialspoint.com/numpy/
