import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from googleapiclient.discovery import build
import googleapiclient.discovery
import googleapiclient.errors
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from collections import Counter

# YouTube API key
API_KEY = 'AIzaSyC8Y7glTka8H9WqymYDKTN190EoJUFlqWU'

# YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)


def collect_comments(video_id, max_comments=None, api_key=API_KEY):
    comments = []

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    request = youtube.commentThreads().list(
        part="snippet,replies",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and (max_comments is None or len(comments) < max_comments):
        response = request.execute()
        for item in response["items"]:
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            if "replies" in item and item["replies"]["comments"]:
                for reply in item["replies"]["comments"]:
                    comments.append(reply["snippet"]["textDisplay"])

        # Check if there is a nextPageToken and update the request accordingly
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText",
                pageToken=response["nextPageToken"]
            )
        else:
            request = None

    # If max_comments is specified, return only the requested number of comments
    if max_comments is not None:
        comments = comments[:max_comments]

    return comments


# Function to remove non-BMP characters
def remove_nonbmp_chars(text):
    return ''.join(c for c in text if c <= '\uFFFF')


def preprocess_text(text):
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    return text


def create_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Load sentiment dataset from CSV and preprocess
sentiment_data = pd.read_csv('output.csv')
sentiment_data['comment'] = sentiment_data['comment'].apply(preprocess_text)

# Tokenize and pad sequences
max_length = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentiment_data['comment'])
sequences = tokenizer.texts_to_sequences(sentiment_data['comment'])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiment_data['label'], test_size=0.2,
                                                    random_state=42)

# Create and train the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
model = create_model(vocab_size, embedding_dim, max_length)
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Print training and validation accuracy during training
for epoch in range(len(history.history['accuracy'])):
    train_acc = history.history['accuracy'][epoch]
    val_acc = history.history['val_accuracy'][epoch]
    print(f"Epoch {epoch + 1}: Train accuracy = {train_acc:.4f}, Validation accuracy = {val_acc:.4f}")


# Function to handle the Analyze button click event
def analyze_comments():
    video_id = video_id_entry.get()
    if not video_id:
        messagebox.showerror("Error", "Please enter a valid video ID.")
        return

    try:
        max_comments = max_comments_entry.get()
        max_comments = int(max_comments) if max_comments.isdigit() else None

        comments = collect_comments(video_id, max_comments=max_comments)
        preprocessed_comments = [preprocess_text(comment) for comment in comments]
        comment_sequences = tokenizer.texts_to_sequences(preprocessed_comments)
        padded_comment_sequences = pad_sequences(comment_sequences, maxlen=max_length)
        predictions = np.argmax(model.predict(padded_comment_sequences), axis=-1)
        sentiments = ['negative' if pred == 0 else ('neutral' if pred == 1 else 'positive') for pred in predictions]

        sentiment_count = Counter(sentiments)

        result_text.delete('1.0', tk.END)
        summary_text.delete('1.0', tk.END)

        for comment, sentiment in zip(comments, sentiments):
            sanitized_comment = remove_nonbmp_chars(comment)
            result_text.insert(tk.END, f"{sanitized_comment}: {sentiment}\n\n")

        summary_text.insert(tk.END, f"Negative comments: {sentiment_count['negative']}\n")
        summary_text.insert(tk.END, f"Neutral comments: {sentiment_count['neutral']}\n")
        summary_text.insert(tk.END, f"Positive comments: {sentiment_count['positive']}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create the main application window
app = tk.Tk()
app.title("YouTube Video Comment Sentiment Analysis")

# Add input fields, labels, and buttons
video_id_label = ttk.Label(app, text="Video ID:")
video_id_label.grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)

video_id_entry = ttk.Entry(app)
video_id_entry.grid(column=1, row=0, padx=10, pady=10, sticky=tk.W)

max_comments_label = ttk.Label(app, text="Max Comments:")
max_comments_label.grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)

max_comments_entry = ttk.Entry(app)
max_comments_entry.grid(column=1, row=1, padx=10, pady=10, sticky=tk.W)

analyze_button = ttk.Button(app, text="Analyze", command=analyze_comments)
analyze_button.grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)

result_label = ttk.Label(app, text="Sentiment Analysis Results:")
result_label.grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)

result_text = ScrolledText(app, wrap=tk.WORD, width=80, height=20)
result_text.grid(column=0, row=4, padx=10, pady=10, columnspan=2)

summary_label = ttk.Label(app, text="Sentiment Summary:")
summary_label.grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)

summary_text = ScrolledText(app, wrap=tk.WORD, width=50, height=6)
summary_text.grid(column=0, row=6, padx=10, pady=10, columnspan=2)

# Run the application
app.mainloop()
