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

train_dir = './train_data'
print('1. Building dictionary')
dictionary = build_dictionary(train_dir)

print('2. Building training features and labels')
features_train = build_features(train_dir, dictionary)
labels_train = build_labels(train_dir)

classifier = MultinomialNB()
print('3. Training the classifier')
classifier.fit(features_train, labels_train)

test_dir = './test_data'
print('4. Building the test features and labels')
features_test = build_features(test_dir, dictionary)
labels_test = build_labels(test_dir)

print('5. Calculating accuracy of the trained classifier')
accuracy = classifier.score(features_test, labels_test)
print 'Accuracy : ', accuracy
