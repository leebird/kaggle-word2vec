from sklearn.cluster import KMeans
import time
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import pandas as pd
from gensim.models import Word2Vec
import numpy as np  # Make sure that numpy is imported

# Load the punkt tokenizer
nltk.data.path.append('/home/leebird/Projects/kaggle-word2vec-nlp-tutorial/data')
tokenizer = nltk.data.load('corpora/tokenizers/punkt/english.pickle')

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return reviewFeatureVecs


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters and non-numbers
    review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)

# Read data from files
train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# load model
model = Word2Vec.load("300features_40minwords_10context")

num_features = 300  # Word vector dimensionality

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# load model
model = Word2Vec.load("300features_40minwords_10context")

start = time.time()  # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = int(word_vectors.shape[0] / 5)

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(model.index2word, idx))
word_centroid_map_length = len(list(word_centroid_map.values()))
word_centroid_map_values = list(word_centroid_map.values())
word_centroid_map_keys = list(word_centroid_map.keys())

# For the first 10 clusters
for cluster in range(0, 10):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    
    for i in range(0, word_centroid_map_length):
        if word_centroid_map_values[i] == cluster:
            words.append(word_centroid_map_keys[i])
    print(words)


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros(( test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier

# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators=100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)

# Your submission scored 0.84588, which is not an improvement of your best score. Keep trying!