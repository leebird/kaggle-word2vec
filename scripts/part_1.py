import sys
import re
from bs4 import BeautifulSoup
import pandas
from sklearn.feature_extraction.text import CountVectorizer

import nltk

nltk.data.path.append('/home/leebird/Projects/kaggle-word2vec-nlp-tutorial/data')
from nltk.corpus import stopwords


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


if __name__ == '__main__':
    train_file = 'data/labeledTrainData.tsv'
    test_file = 'data/testData.tsv'
    train = pandas.read_csv(train_file, header=0, delimiter="\t", quoting=3)
    num_reviews = train['review'].size
    clean_reviews = []
    for i in range(0, num_reviews):
        if (i + 1) % 1000 == 0:
            print("Review %d of %d" % ( i + 1, num_reviews ), end='\r')
        clean_reviews.append(review_to_words(train["review"][i]))

    print()

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_reviews)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    print("Training the random forest...")
    from sklearn.ensemble import RandomForestClassifier

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit(train_data_features, train["sentiment"])

    # Read the test data
    test = pandas.read_csv(test_file, header=0, delimiter="\t", quoting=3)

    # Verify that there are 25,000 rows and 2 columns
    print(test.shape)

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = []

    print("Cleaning and parsing the test set movie reviews...")
    for i in range(0, num_reviews):
        if (i + 1) % 1000 == 0:
            print("Review %d of %d" % (i + 1, num_reviews), end='\r')
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pandas.DataFrame(data={"id": test["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)