import streamlit as st
from src.config.config_management import ConfigManager as cm
import warnings
warnings.filterwarnings("ignore", message=".*was created with a default value.*")

def setup_page_ui(version=2):
    st.title("Numerical Option Pricing")

def stats_screen_ui():
    st.markdown("Nothing to show here")
