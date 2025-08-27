
import time
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import streamlit as st


def load_data():
    data = pd.read_csv('./data/sentiment_data.csv')
    return data


# load data
sentiment_df = load_data()

# load data for tokenize
df_train = pd.read_csv('./data/train/X_train.csv')
train_set = df_train['sentiment'].tolist()


def stream_text(sentence):
    for word in sentence.split(' '):
        yield word + " "
        time.sleep(0.05)


def prediction_summary(prediction, encoded_labels):
    pred_idx = np.argmax(prediction)
    pred_label = encoded_labels[pred_idx]
    return {
        'prob': {
            'Positive': prediction[0],
            'Neutral': prediction[1],
            'Negative': prediction[2]
        },
        'label': pred_label.title()
    }


def load_random_data():
    random_idx = np.random.choice(range(len(sentiment_df)), size=1)[0]
    random_data = sentiment_df.iloc[random_idx]['sentiment']
    return random_data


def labels_encoder():
    df_labels = pd.read_csv('./data/train/y_train.csv')
    # convert dataframe to list for tokenize
    labels = df_labels['label'].tolist()
    # tokenize the label
    label_tokenizer = Tokenizer()
    # fit tokenizer to labels
    label_tokenizer.fit_on_texts(labels)
    # create label-index dictionary
    label_index_word = label_tokenizer.index_word
    # start encoder label from 0
    encoded_labels = {key - 1: value for key,
                      value in label_index_word.items()}
    print(f'labels encoded: {encoded_labels}')
    return encoded_labels


def predict_model(sentence):
    # ==== SET HYPERPARAMETERS ====
    # Vocabulary size of the tokenizer
    vocab_size = 10000
    # Maximum length of the padded sequences
    max_length = 50

    # tokenize
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_set)

    # load model
    loaded_model = load_model('./model/checkpoint-06-0.873-0.823.keras')

    # input for prediction
    input_sequences = tokenizer.texts_to_sequences([sentence])
    padded_input = pad_sequences(
        input_sequences, maxlen=max_length, padding='post', truncating='post')
    prediction = loaded_model.predict(
        padded_input)  # show softmax probabiliites, shape(1,m)

    encoded_labels = labels_encoder()  # label encoder for classification
    summary = prediction_summary(prediction[0], encoded_labels)

    return summary
