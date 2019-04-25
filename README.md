# Sentiment-Analysis-IMDB
Sentiment Analysis using TF-IDF features from the IMDB Movie reviews Dataset.

# Code Structure
The folder has two python files, preprocessing.py and driver.py for execution from scratchwith
the raw aclImdb dataset. preprocessing.py takes in the raw data, both training and test, and
converts it into a .csv file for ease during classification. The .csv file is structured in the
following manner [‘S.No’, ‘Text’, ‘Label’] . The label is 1 is given for positive reviews and label
0 is given for negative reviews. The text is formatted by removal of punctuation, stop words and
numbers. All alphabet is then changed to lowercase. The datasets are then written to create
TrainingSet.csv and TestSet.csv.

If these files are already present we can directly run the script driver.py which gives us the
accuracy metrics for our prediction after performing the classification task on the test dataset.
First it reads both the training and test data , and shuffles it so that reviews of a similar type are
not clubbed together. Then we create a bigram vectorizer using the training data and fit it to both
the training and test data . To get more features in our model, we create term-frequency inverse
document-frequency ( tf-idf ) vectors of the bigram counts we obtained. These are then fit on a
Linear SVM classifier . Finally it prints the overall-accuracy of our model and the confusion
matrix of the results.

# Tools used
1. The pandas library is used to perform data manipulations to clean the data into a usable format
2. The nltk library is used for text cleaning and providing a corpus to remove common stop words
3. The string library is used to remove punctuations
4. The os library is used for accessing the aclImdb dataset
5. The sklearn or sci-kit learn library is used to perform the following functions
  1. Create bigram vectors and then the tf-idf vectors
  2. Providing the metric report from using the confusion matrix
  3. Perform classification using the linear SVM

# Accuracy Metrics
The metrics were reported as follows -
1. Overall Accuracy - 89.34%
2. True Positives - 11191/12500
3. True Negatives - 11145/12500
4. False Positives - 1355/12500
5. False Negatives - 1309/12500
