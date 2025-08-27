import streamlit as st
from utils import *


def show_result(summary):
    probs, label = summary.values()

    # assign the prob of result
    prob = probs[label] * 1
    print(prob)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label='Sentiment Prediction',
                  value=label)
    with col2:
        st.metric(label='Prediciton Prob:', value=f'{round(prob,4)}')

    # style_metric_cards(
    #     border_color=None,
    #     border_left_color="#FFA500",
    # )

    st.bar_chart(probs, color="#FFA500")


def sidebar():
    with st.sidebar:
        st.header(
            'Get the experience sentiment analysis firsthand here!')
        st.image('./images/qr-code.png', caption='Scan me!')
        st.link_button("Word embedding", "https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/sayid-alt/sentiment-analysis-nlp/main/model/embedding_word/projector_config.json")


def main():
    sidebar()

    # logo
    st.image('images/banner.png')
    st.title('What\'s on your mind?')
    st.caption(
        "🚀 This project will classify the sentiment of your input into predefined categories: positive, neutral, or negative.")

    # user input
    input_sentence = st.text_area('Your sentence')

    # condition if user input the sentence
    if input_sentence:
        # UI run the inputed text streamly

        st.write_stream(stream_text(input_sentence))
        # run prediction
        with st.spinner('Predict...'):
            summary = predict_model(input_sentence)
        show_result(summary)

    # condition if user choose to generate random sentence
    if st.button("Random Sentence"):
        # generate random sentence
        random_sentence = load_random_data()

        # write the random sentence streamly
        st.write_stream(stream_text(random_sentence))

        # Run prediction
        with st.spinner('Predict...'):
            summary = predict_model(input_sentence)

        # function to show the result of predicition
        show_result(summary)


if __name__ == "__main__":
    main()
