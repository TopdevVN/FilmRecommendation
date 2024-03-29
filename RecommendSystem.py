import streamlit as st
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data1 = pd.read_csv("film_data.csv")
data = data1.loc[0:5000]

m = data['Tomatometer count'].quantile(0.90)  # 120
C = data['Tomatometer count'].mean()

t = data['Tomatometer score'].quantile(0.50)  # 72


def weighted_rating(x, m=m, C=C):
    v = x['Tomatometer count']
    R = x['Tomatometer score']
    # Calculation based on the WR formula
    return round((v / (v + m) * R) + (m / (m + v) * C), 2)


def recommendations(title):
    film_data = data
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(film_data['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    film_data = film_data.reset_index()
    indices = pd.Series(film_data.index, index=film_data['Title'])

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:16]
    movie_indices = [i[0] for i in sim_scores]

    movies = film_data.iloc[movie_indices][['Title', 'Genre', 'Tomatometer count', 'Tomatometer score']]
    qualified = movies[(movies['Tomatometer count'] >= m) & (movies['Tomatometer score'] >= t)]
    qualified['WR_score'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('WR_score', ascending=False).head(15)
    return qualified


st.set_page_config(
    page_title="RECOMMEND SYSTEM",
    page_icon="🔥",
    layout="wide")
st.title('**:blue[FILM RECOMMENDATION SYSTEM]**')
with st.form("Thông tin"):
    options = data["Title"]
    name = st.selectbox('**:red[Typing the film title]**', options=options)
    submit = st.form_submit_button("**Get films**")

# Nút tìm kiếm
if submit:
    with st.spinner("Loading..."):
        time.sleep(0.25)
    try:
        a = recommendations(name)
        st.success('**Success**', icon="✅")
        st.write('**:orange[Here are movies similar to]**', name)
        for i in range(len(a)):
            with st.form('' + str(i) + ''):
                st.markdown(f':green[**👀Title**:] {a.iloc[i, 0]}')
                st.markdown(f':green[**👀Genre**:] {a.iloc[i, 1]}')
                st.markdown(f':green[**👀Tomatometer count**:] {a.iloc[i, 2]}')
                st.markdown(f':green[**👀Tomatometer score**:] 🍅 {a.iloc[i, 3]} 🍅')
                st.markdown(f':green[**👀	WR_score**:] {a.iloc[i, 4]}')
                submit = st.form_submit_button(str(i + 1), disabled=True)
    except:
        st.error('There are no movies that are similar to ' + name, icon="❌")

##########################
# import base64
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()
#
#
# img = get_img_as_base64("D:/pythonProject/image4.jpg")


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://previews.123rf.com/images/anatolir/anatolir1905/anatolir190503987/122504351-cinema-popcorn-box-pattern-seamless.jpg");
background-size: 100%;
background-position: center;
background-repeat: repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
