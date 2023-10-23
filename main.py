
import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide")
# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 

with st.sidebar.expander("About"):
    st.write("Welcome to the Sentiment Analysis NLP App!")
    st.write("This app allows you to analyze the sentiment of a text input. It provides information about the polarity and subjectivity of the text.")
    st.write("Additionally, you can see the sentiment of individual tokens within the text.")
    st.write("Feel free to enter text and click 'Analyze' to see the results.")
    st.write("You can also explore the 'Token Sentiment' section to see the sentiment of individual words.")



def main():
    st.title("Sentiment Analysis NLP App")
    with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
    col1, col2, col3 = st.columns(3)
    if submit_button:

            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

            with col2:
                st.info("Token Sentiment")

                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

            with col3:
                st.info("Visualization")
                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric'
                )
                st.altair_chart(c, use_container_width=True)
                
                
                st.info("Sentiment Distribution")
                st.subheader("Pie Chart of Sentiment Distribution")
                sentiment_counts = [len(token_sentiments['positives']), len(token_sentiments['negatives']), len(token_sentiments['neutral'])]
                labels = ['Positive', 'Negative', 'Neutral']
                fig, ax = plt.subplots()  # Create a figure and axis
                ax.pie(sentiment_counts, labels=labels, autopct='%1.1f%%')
                st.pyplot(fig)
                



if __name__ == '__main__':
	main()