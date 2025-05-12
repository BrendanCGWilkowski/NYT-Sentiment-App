import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Get data out of the pickle
nyt = pd.read_pickle("nyt_filtered.pkl")
nyt['year_month'] = nyt['pub_date'].dt.to_period('M')
nyt_filtered = nyt[nyt['year_month'].between('2024-01', '2025-03')]

st.set_page_config(layout="wide")
st.title("NYT 2024 Presidential Candidate Headline Sentiment Analysis: Trump, Biden, Harris")

#Select the month, year combination
months = sorted(nyt_filtered['year_month'].unique())
selected_month = st.selectbox("Select a Month of 2024/25", months)

#Basic details
candidates = ['Trump', 'Biden', 'Harris']
sentiments = ['positive', 'neutral', 'negative']
month_df = nyt_filtered[nyt_filtered['year_month'] == selected_month]

#This will show the info
st.subheader(f"Sentiment Proportion – {selected_month.strftime('%B %Y')}")
sentiment_data = {}

#Do this stuff again to collect data from the pickle file
for candidate in candidates:
    df_cand = month_df[month_df['main_headline'].str.contains(candidate, case=False, na=False)]
    counts = df_cand['headline_sentiment_vader'].value_counts(normalize=True)
    sentiment_data[candidate] = {s: counts.get(s, 0) for s in sentiments}

sentiment_df = pd.DataFrame(sentiment_data).T[sentiments]
#We need to show the colors properly
colors = {'positive': 'green', 'neutral': 'grey', 'negative': 'red'}
fig, ax = plt.subplots(figsize=(8, 4))
sentiment_df.plot(kind='bar', color=[colors[s] for s in sentiments], ax=ax)
plt.ylabel("Proportion")
plt.title("Headline Sentiment by Candidate")
plt.xticks(rotation=0)
st.pyplot(fig)

#World clouds
st.subheader("Word Clouds by Candidate")
cols = st.columns(3)

#Get some key wordclouds for each sentiment
for idx, candidate in enumerate(candidates):
    with cols[idx]:
        st.markdown(f"**{candidate}**")
        df_cand = month_df[month_df['main_headline'].str.contains(candidate, case=False, na=False)]
        text = " ".join(df_cand['main_headline'].dropna().tolist())
        if text.strip():
            wc = WordCloud(width=300, height=200, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No headlines available.")

#Show top headlines for each sentiment
st.subheader(f"Top Headlines by Sentiment – {selected_month.strftime('%B %Y')}")

for candidate in candidates:
    st.markdown(f"### {candidate}")
    df_cand = month_df[month_df['main_headline'].str.contains(candidate, case=False, na=False)]

    for sentiment in sentiments:
        st.markdown(f"**{sentiment.capitalize()}**")
        subset = df_cand[df_cand['headline_sentiment_vader'] == sentiment]
        headlines = subset['main_headline'].head(3).tolist()
        if headlines:
            for hl in headlines:
                st.write(f"- {hl}")
        else:
            st.write("No headlines available.")
