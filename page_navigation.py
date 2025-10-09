import streamlit as st

def initialise_and_begin_navigation():
    main_screen_page = st.Page("pages/1_main_screen.py", title="main")
    pages = [main_screen_page]
    page = st.navigation(pages)
    page.run()