import streamlit as st
from PIL import Image
from pages import realtime, games

im = Image.open("favicon.ico")
st.set_page_config(page_title="ASL Sign Language Recognition",
                   layout="centered",
                   page_icon=im)
page_names_to_funcs = {
    "Fingerspelling Recognition": realtime,
    "Games": games,
}

st.sidebar.title("Settings")
demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
