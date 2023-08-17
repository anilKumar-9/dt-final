import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "About page", "Contact"],  # required
        icons=["house", "book", "envelope"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )



if selected == "Home":
    st.header('Book Recommender System Using Machine Learning')
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
    book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))


    def fetch_poster(suggestion):
        book_name = []
        ids_index = []
        poster_url = []

        for book_id in suggestion:
            book_name.append(book_pivot.index[book_id])

        for name in book_name[0]:
            ids = np.where(final_rating['title'] == name)[0][0]
            ids_index.append(ids)

        for idx in ids_index:
            url = final_rating.iloc[idx]['image_url']
            poster_url.append(url)

        return poster_url


    def recommend_book(book_name):
        books_list = []
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        poster_url = fetch_poster(suggestion)

        for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                books_list.append(j)
        return books_list, poster_url


    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names
    )

    if st.button('Show Recommendation'):
        recommended_books, poster_url = recommend_book(selected_books)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_books[1])
            st.image(poster_url[1])
        with col2:
            st.text(recommended_books[2])
            st.image(poster_url[2])

        with col3:
            st.text(recommended_books[3])
            st.image(poster_url[3])
        with col4:
            st.text(recommended_books[4])
            st.image(poster_url[4])
        with col5:
            st.text(recommended_books[5])
            st.image(poster_url[5])

if selected == "About page":
    st.write("# Welcome to book Recommendation page!")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        A recommendation system filters information by predicting ratings or preferences of customers for items that the customers would like to use. It tries to recommend items to the customers according to their needs and taste.? Select a demo from the sidebar** to see some examples
        of what book Recommendation  can do!
        ### Want to learn more?
        - Check out [book recommendation system ](https://medium.com/@amitdlmlai/book-recommendation-system-61bf9284f659)
        - Jump into our [documentation](https://cse.anits.edu.in/projects/projects2021B10.pdf)
        ### See more complex demos
        - DATA SET [Books Data set](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
        - Project code [github](https://github.com/anilKumar-9/DT-Project)
    """
    )
if selected == "Contact":
        st.write("# Project BY Batch NO:- 5")
        st.write("## NAME            || ROLL NUMBER")
        st.write("### N.ANIL KUMAR   || 21R11A1243")
        st.write("### K.MANASA REDDY || 21R11A1223")
        st.write("### UVS.ABHI RAM   || 21R11A1259")
        st.write("### G.JOEL ANAND   || 21R11A1215")












