
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import numpy
from scipy import stats


#%%

YT = "testsataset.csv"
twitter = "Twitter dataset filepath"
YTtwitter = "combined dataset filepath"

YoutubeTrendingDatasetCommentsFilepath = 'UScomments.csv'
YoutubeTrendingDatasetVideosFilepath = 'USvideos.csv'


#%%

def main():

  #base prediction youtube training
  sv = predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YT,'Youtube','base','Base prediction')
  #last parameter can be used for printouts and graph titles
  sg = predictionFunction(SGDClassifier(),'SGDC',YT,'Youtube', 'base','Base prediction')
  co = predictionFunction(ComplementNB(),'Comp. NB',YT,'Youtube' ,'base','Base prediction')
  mu = predictionFunction(MultinomialNB(),'Mult. NB',YT,'Youtube' ,'base','Base prediction')
  lo = predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YT,'Youtube','base','Base prediction')

  #can be used to create histograms
  values = [sv, sg, co, mu, lo]
  colors = [(0.72,0.87,0.8),(0.35,0.73,0.82),(0.22,0.52,0.71),
  (0.15,0.29,0.57),(0.09,0.12,0.39)]
  labels = ['SVM', 'SGDC', 'Comp. NB', 'Mult. NB', 'Log. regr.']
  plt.figure(1, figsize=(9,5))
  n, bins, patches = plt.hist(values, color=colors, label=labels)
  plt.xticks(bins)
  plt.grid(color='darkgrey', lw = 0.5, axis='y')
  plt.ylabel('Number of videos')
  plt.xlabel('Absolute error')
  plt.legend()
  plt.show()
#base prediction twitter training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',twitter,'Twitter', 'base','Base prediction')
#sg=predictionFunction(SGDClassifier(),'SGDC',twitter,'Twitter','base','Base prediction')
#co=predictionFunction(ComplementNB(),'Comp. NB',twitter,'Twitter','base','Base prediction')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',twitter,'Twitter','base','Base prediction')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',twitter,'Twitter','base','Base prediction')

#base prediction combined training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YTtwitter,'Youtube + Twitter','base','Base prediction')
#sg=predictionFunction(SGDClassifier(),'SGDC',YTtwitter,'Youtube + Twitter','base','Base prediction')
#co=predictionFunction(ComplementNB(),'Comp. NB',YTtwitter,'Youtube + Twitter','base','Base prediction')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YTtwitter,'Youtube + Twitter','base','Base prediction')
 #lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YTtwitter,'Youtube + Twitter','base','Base prediction')

#prediction 2 Youtube training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YT,'Youtube','p2','Counting neutral comments as negative')
#sg=predictionFunction(SGDClassifier(),'SGDC',YT,'Youtube','p2','Counting neutral comments as negative')
#co=predictionFunction(ComplementNB(),'Comp. NB',YT,'Youtube','p2','Counting neutral comments as negative')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YT,'Youtube','p2','Counting neutral comments as negative')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YT,'Youtube','p2','Counting neutral comments
# as negative')

#prediction 2 Twitter training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',twitter,'Twitter','p2','Counting neutral comments as negative')
#sg=predictionFunction(SGDClassifier(),'SGDC',twitter,'Twitter','p2','Counting neutral comments as negative')
#co=predictionFunction(ComplementNB(),'Comp. NB',twitter,'Twitter','p2', 'Counting neutral comments as negative')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',twitter,'Twitter','p2', 'Counting neutral comments as negative')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',twitter, 'Twitter','p2','Counting neutral comments
# as negative')

#prediction 2 combined training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YTtwitter, 'Youtube + Twitter','p2','Counting neutral comments
# as negative')
#sg=predictionFunction(SGDClassifier(),'SGDC',YTtwitter,'Youtube + Twitter','p2', 'Counting neutral comments as negative')
#co=predictionFunction(ComplementNB(),'Comp. NB',YTtwitter,'Youtube + Twitter','p2','Counting neutral comments as negative')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YTtwitter,'Youtube + Twitter','p2','Counting neutral comments as negative')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YTtwitter, 'Youtube + Twitter','p2',
# 'Counting neutral comments as negative')

#prediction 3 Youtube training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YT,'Youtube','p3',
# 'Counting half of neutral comments as positive')
#sg=predictionFunction(SGDClassifier(),'SGDC',YT,'Youtube','p3',
# 'Counting half of neutral comments as positive')
#co=predictionFunction(ComplementNB(),'Comp. NB',YT,'Youtube','p3',
# 'Counting half of neutral comments as positive')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YT,'Youtube','p3',
# 'Counting half of neutral comments as positive')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YT,'Youtube','p3',
# 'Counting half of neutral comments as positive')

#prediction 3 Twitter training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',twitter,'Twitter','p3',
# 'Counting half of neutral comments as positive')
#sg=predictionFunction(SGDClassifier(),'SGDC',twitter,'Twitter','p3',
# 'Counting half of neutral comments as positive')
#co=predictionFunction(ComplementNB(),'Comp. NB',twitter,'Twitter','p3',
# 'Counting half of neutral comments as positive')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',twitter,'Twitter','p3',
# 'Counting half of neutral comments as positive')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',twitter,'Twitter','p3',
# 'Counting half of neutral comments as positive')

#prediction 3 combined training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YTtwitter,'Youtube + Twitter','p3','Counting half of
# neutral comments as positive')
#sg=predictionFunction(SGDClassifier(),'SGDC',YTtwitter,'Youtube + Twitter','p3','Counting half of neutral
# comments as positive')
#co=predictionFunction(ComplementNB(),'Comp. NB',YTtwitter,'Youtube + Twitter','p3',
# 'Counting half of neutral comments as positive')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YTtwitter,'Youtube + Twitter','p3',
# 'Counting half of neutral comments as positive')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YTtwitter,'Youtube + Twitter','p3',
# 'Counting half of neutral comments as positive')

#prediction 4 Youtube training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YT,'Youtube','p4',
# 'Counting all of neutral comments as positive')
#sg=predictionFunction(SGDClassifier(),'SGDC',YT,'Youtube','p4',
# 'Counting all of neutral comments as positive')
#co=predictionFunction(ComplementNB(),'Comp. NB',YT,'Youtube','p4',
# 'Counting all of neutral comments as positive')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YT,'Youtube','p4',
# 'Counting all of neutral comments as positive')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.', YT,'Youtube','p4','Counting all of
# neutral comments as positive')

#prediction 4 Twitter training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',twitter,'Twitter','p4', 'Counting all of neutral
# comments as positive')
#sg=predictionFunction(SGDClassifier(),'SGDC',twitter,'Twitter','p4', 'Counting all of neutral comments as positive')
#co=predictionFunction(ComplementNB(),'Comp. NB',twitter,'Twitter','p4', 'Counting all of neutral comments as positive')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',twitter,'Twitter','p4', 'Counting all of neutral comments as positive')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',twitter,'Twitter','p4','Counting all of
# neutral comments as positive')

#prediction 4 combined training
#sv=predictionFunction(svm.LinearSVC(max_iter=1000),'SVM',YTtwitter,'Youtube + Twitter','p4','Counting all of
# neutral comments as positive')
#sg=predictionFunction(SGDClassifier(),'SGDC',YTtwitter,'Youtube + Twitter','p4','Counting all of
# neutral comments as positive')
#co=predictionFunction(ComplementNB(),'Comp. NB',YTtwitter,'Youtube + Twitter','p4','Counting all of
# neutral comments as positive')
#mu=predictionFunction(MultinomialNB(),'Mult. NB',YTtwitter,'Youtube + Twitter', 'p4','Counting all of
# neutral comments as positive')
#lo=predictionFunction(LogisticRegression(max_iter=1000),'Log. regr.',YTtwitter, 'Youtube + Twitter','p4',
# 'Counting all of neutral comments as positive')

#%%

#predictionFunction runs the training, prediction and presents the results
def predictionFunction(model, modelname, dataAdrs, dataSetName, prediction, predictionName):
    data = []
    data_sentiment = []

    with open(dataAdrs, encoding="utf8") as yt_f:
        reader = csv.reader(yt_f, delimiter=',')

        for row in reader:
            data.append(row[0]) # text
            data_sentiment.append(row[1]) # sentiment

    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )

    features = vectorizer.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        data_sentiment,
        train_size=0.80,
        random_state=1234)

    model = model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    #predict training dataset
    yhat = model.predict(X_test)
    # evaluate accuracy
    trainingDataAccuracy = accuracy_score(y_test, yhat)

    videoSentiments = {}
    commentLists = {}

    with open(YoutubeTrendingDatasetCommentsFilepath,
        encoding="utf8") as GBUSCommentsFile:
        reader = csv.reader(GBUSCommentsFile, delimiter=',')
        for row in reader:
            if len(row[0]) > 11:
                continue
            if row[0] not in videoSentiments:
                videoSentiments[row[0]] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if row[0] not in commentLists:
                commentLists[row[0]] = [row[1]]
            cl = commentLists[row[0]]
            cl.append(row[1])
            commentLists[row[0]] = cl

    noComments = 0
    totPcount = 0
    totNeutCount = 0
    totNegCount = 0
    for vid in commentLists:
        youtubeFeatures = vectorizer.transform(commentLists[vid])
        predictions = model.predict(youtubeFeatures)
        sentimentScores = videoSentiments[vid]

        for comment in range(0, len(commentLists[vid])):
            if predictions[comment] == '4':
                sentimentScores[0] += 1.0
                totPcount += 1
            if predictions[comment] == '2':
                sentimentScores[1] += 1.0
                totNeutCount += 1
            elif predictions[comment] == '0':
                sentimentScores[2] += 1.0
                totNegCount += 1
            noComments +=1
        videoSentiments[vid] = sentimentScores

    with open(YoutubeTrendingDatasetVideosFilepath,
                encoding="utf8") as GBUSVideosFile:
        reader = csv.reader(GBUSVideosFile, delimiter=',')
        for row in reader:
            if row[0] in videoSentiments:
                setVideo = videoSentiments[row[0]]
                setVideo[3] = float(row[2]) # likes
                setVideo[4] = float(row[3]) # dislikes

                try:
                    # actual like ratio
                    setVideo[6] = setVideo[3] / (setVideo[3] + setVideo[4])
                    # predicted like ratio
                    if prediction == 'base':
                        setVideo[5] = setVideo[0]/(setVideo[0]+setVideo[2])
                    if prediction == 'p2':
                        setVideo[5]=setVideo[0]/(setVideo[0]+setVideo[1]+setVideo[2])
                    if prediction == 'p3':
                        setVideo[5]=(setVideo[0]+0.5*setVideo[1])/(setVideo[0]+setVideo[1]+setVideo[2])
                    if prediction == 'p4':
                        setVideo[5]=(setVideo[0]+1*setVideo[1])/(setVideo[0]+setVideo[1]+setVideo[2])
                    videoSentiments[row[0]] = setVideo
                except: # incase only neutral comments for base prediction
                    del videoSentiments[row[0]]

    actualRatioSequential = []
    predictSequential = []

    for key in videoSentiments:
        actualRatioSequential.append(videoSentiments[key][6])
        predictSequential.append(videoSentiments[key][5])

    mae = 0
    differences = []
    absdifferences = []

    for i in range(0, len(actualRatioSequential)):
        mae += abs(predictSequential[i] - actualRatioSequential[i])
        differences.append(predictSequential[i] - actualRatioSequential[i])
        absdifferences.append(abs(predictSequential[i] - actualRatioSequential[i]))

    mae = mae / len(actualRatioSequential)
    print(modelname, dataSetName, predictionName)
    print("training dataset accuracy: ", trainingDataAccuracy)
    print("number of samples in testing dataset: ", len(differences))
    pearsonR, pValue = stats.pearsonr(predictSequential, actualRatioSequential)
    print("Pearson correlation: ", pearsonR, "Two tailed p-value", pValue)
    print("Mean absolute error: ", mae)
    print("SD of likeratio differences:", numpy.std(differences, ddof=1))
    print("average like ratio testing dataset: ", numpy.average(actualRatioSequential))
    print("SD like ratio testing dataset",numpy.std(actualRatioSequential, ddof=1))
    print("number of comments testing dataset: ", noComments)
    print("Positive comments training dataset: ", totPcount)
    print("Neutral comments testing dataset:", totNeutCount)
    print("Negative comments testing dataset: ", totNegCount)
    print("----------------------------------------------------------------------------------------------------------")
    return absdifferences # list of differences between predicted & actual like proportions


def predictionFunction2(model, modelname, dataAdrs, dataSetName, prediction, predictionName):
    data = []
    data_sentiment = []

    with open(dataAdrs, encoding="utf8") as yt_f:
        reader = csv.reader(yt_f, delimiter=',')

        for row in reader:
            data.append(row[0]) # text
            data_sentiment.append(row[1]) # sentiment

    vectorizer = CountVectorizer(
        analyzer='word',
        lowercase=False,
    )

    features = vectorizer.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        data_sentiment,
        train_size=0.80,
        random_state=1234)

    model = model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X_test)

    #predict training dataset
    yhat = model.predict(X_test)
    # evaluate accuracy
    trainingDataAccuracy = accuracy_score(y_test, yhat)

    return trainingDataAccuracy



sv = predictionFunction2(svm.LinearSVC(max_iter=1000),'SVM',YT,'Youtube','base','Base prediction')
print(sv)