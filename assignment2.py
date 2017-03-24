#!/usr/bin/env python
# -*- coding: utf-8 -*-
__mtime__ = '2017/3/11 0011'
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR, SVR
import numpy as np
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
def pm(p):
    if p in product_ratings_mean:
        return product_ratings_mean[p]
    else:
        return product_ratings_global_mean


n_topics = 100

X = []
y = []
product_ratings = defaultdict(list)
product_ratings_mean = {}
user_ratings = defaultdict(list)
file = open("movies.txt", encoding='ISO-8859-2')
wordSet_r = set()
wordSet_s = set()
product_review_maxlen = defaultdict(int)
product_summary_maxlen = defaultdict(int)
user_review_maxlen = defaultdict(int)
user_summary_maxlen = defaultdict(int)
WordCount = defaultdict(int)
WordCount_summary = defaultdict(int)
ratinghist = [0]*5
for i in range(0, 50000):
    l = [file.readline() for _ in range(0, 9)]
    user, product = l[1][l[1].find(':')+1:], l[0][l[0].find(':')+1:]
    review = re.findall('\w+', l[7][l[7].find(':')+1:])
    for word in review:
        WordCount[word] += 1
    summary = re.findall('\w+', l[6][l[6].find(':')+1:])
    for word in summary:
        WordCount_summary[word] += 1
    rating = int(l[4][l[4].find(':')+1:-3])
    ratinghist[rating-1] += 1
    product_review_maxlen[product] = max(len(review), product_review_maxlen[product])
    product_summary_maxlen[product] = max(len(summary), product_summary_maxlen[product])
    user_review_maxlen[user] = max(len(review), user_review_maxlen[user])
    user_summary_maxlen[user] = max(len(summary), user_summary_maxlen[user])
    wordSet_r.update(review)
    wordSet_s.update(summary)
    y.append(float(l[4][l[4].find(':')+1:]))
    X.append((l[6][l[6].find(':')+1:], l[7][l[7].find(':')+1:], l[0][l[0].find(':')+1:], l[1][l[1].find(':')+1:]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

for i in range(0, len(X_train)):
    product_ratings[X_train[i][2]].append(y_train[i])
reviewTxt = []
maxlen = 0
minlen = 0
for i in range(0, len(X)):
    r = re.findall('\w+', X[i][1] )
    if len(r) > maxlen:
        maxlen = len(r)
        maxlenuser = X[i][3]
        maxlenitem = X[i][2]
    if len(r) < minlen:
        minlen = len(r)
        minlenuser = X[i][3]
        minlenitem = X[i][2]
    reviewTxt.append(r)

    user_ratings[X[i][3]].append(y[i])
print(len(user_ratings))
maxreview = 0
minreview = 10
for a, b in user_ratings.items():
    if len(b) > maxreview:
        maxreview = len(b)
        maxuser = a
    if len(b) < minreview:
        minreview = len(b)
        minuser = a

product_ratings_global_mean = 0.0
for a, b in product_ratings.items():
    product_ratings_mean[a] = np.mean(b)
    product_ratings_global_mean += np.sum(b)


product_ratings_global_mean /= np.sum(len(b) for b in product_ratings.values())
plt.hist([len(b) for a,b in user_ratings.items()], np.arange(0,maxreview))
plt.xlabel('number of reviews of a user')
plt.ylabel('number of user')
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                   max_features=10000,
                                   stop_words='english')
X_train_review_tf = tf_vectorizer.fit_transform(r[1] for r in X_train)
X_test_review_tf = tf_vectorizer.transform(r[1] for r in X_test)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=200,
                                   stop_words='english')

X_train_summary_tf = tfidf_vectorizer.fit_transform(r[0] for r in X_train).todense()
X_test_summary_tf = tfidf_vectorizer.transform(r[0] for r in X_test).todense()

X_train_pm = [pm(r[2]) for r in X_train]
X_test_pm = [pm(r[2]) for r in X_test]

lda = LatentDirichletAllocation(n_topics=n_topics,
                                learning_method='online',
                                random_state=0, n_jobs=1)

X_train_lda = lda.fit_transform(X_train_review_tf)
X_test_lda = lda.transform(X_test_review_tf)

X_train = np.column_stack((X_train_lda, X_train_summary_tf, X_train_pm))
X_test = np.column_stack((X_test_lda, X_test_summary_tf, X_test_pm))

svm = LinearSVR()
svm.fit(X_train, y_train)
mse = mean_squared_error(y_test, svm.predict(X_test))
print(mse)



svm = LinearSVR()
svm.fit(X_train_summary_tf, y_train)
mse = mean_squared_error(y_test, svm.predict(X_test_summary_tf))
print(mse)

product_maxreviewlen = [b for a, b in product_review_maxlen.items()]
product_maxsummarylen = [b for a, b in product_summary_maxlen.items()]
user_maxreviewlen = [b for b in user_review_maxlen.values()]
user_maxsummarylen = [b for b in user_summary_maxlen.values()]
plt.bar(range(len(product_review_maxlen)), product_maxreviewlen, color = 'r')
plt.xlabel('product')
plt.ylabel('maximum reivew length of each product')

plt.bar(range(1,6), ratinghist, color = 'red')
plt.xlabel('ratings/star')
plt.ylabel('number of user')
plt.title('rating distribution histogram')
plt.xticks(np.arange(1,6)+0.4, ('1.0', '2.0', '3.0', '4.0', '5.0'))
wordFreq = list(reversed(sorted( WordCount.values() )))[:2000]
wordFreq_summary = list(reversed(sorted(WordCount_summary.values() )))[:2000]
plt.bar(range(2000), wordFreq)
plt.xlabel('words')
plt.ylabel('word frequency')
