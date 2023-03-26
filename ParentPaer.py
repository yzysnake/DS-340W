import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_data(filename):
    return pd.read_csv(filename)


def textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


def main():
    # Replace 'dataset.csv' with the filename of your dataset
    data = load_data('dataset.csv')

    # Separate the comments (features) and labels (targets)
    X = data['Comments']
    y = data['Label']

    # Initialize TfidfVectorizer to convert text data into numerical features
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Transform the comments into a matrix of TF-IDF features
    X = vectorizer.fit_transform(X)

    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the classifiers to be used in the analysis
    classifiers = {
        'TextBlob': textblob_sentiment,
        'SVM': SVC(),
        'SGDClassifier': SGDClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier()
    }

    # Train and evaluate each classifier
    for classifier_name, classifier in classifiers.items():
        if classifier_name == 'TextBlob':
            y_pred = [textblob_sentiment(str(text)) for text in X_test.toarray()]
        else:
            # Train the classifier on the training set
            classifier.fit(X_train, y_train)

            # Predict the sentiment labels for the test set
            y_pred = classifier.predict(X_test)

        # Calculate the accuracy by comparing the predicted labels to the true labels
        accuracy = accuracy_score(y_test, y_pred)

        # Print the accuracy for the current classifier
        print(f"{classifier_name} accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
