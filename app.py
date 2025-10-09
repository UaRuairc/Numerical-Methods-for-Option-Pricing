import streamlit as st
from src.config.config_init import set_defaults
from page_navigation import initialise_and_begin_navigation

set_defaults()
st.set_page_config(initial_sidebar_state="expanded")
initialise_and_begin_navigation()