import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import base64

st.set_page_config(
    page_title="NoteFinder",
    page_icon=":notebook:",
)

col1, col2, col3 = st.columns([0.5,1,1])
with col1:
    st.write(' ')
with col2:
    st.image("/Users/sidsomani/Desktop/streamlit_projects/pages/logovs2.png", width = 420)
with col3:
    st.write(' ')



st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)


if "APIkey" not in st.session_state:
    st.session_state["APIkey"] = ""

#Add our icon next to this
# st.write("")

st.markdown("<h1 style='text-align: center; color: white;'>NoteFinder Login</h1>", unsafe_allow_html=True)
#st.markdown("<p style='text-align: center; color: white;'>Enter your OpenAI API key:</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.write(' ')

with col2:
    openai_api_key = st.text_input('', type='password')
    st.session_state.openai_api_key = openai_api_key
    if not st.session_state.openai_api_key:
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    elif not st.session_state.openai_api_key.startswith('sk-'):
        st.error("Invalid OpenAI API key!", icon='⚠')
    else:
        st.success("Valid OpenAI API key")

with col3:
    st.write(' ')


#Add a second conditional which checks if the APIkey entered is a real key

col1, col2, col3 = st.columns([1.6,1.1,1])
with col1:
    st.write(' ')
with col2:
    if st.button("Proceed"):
        switch_page("notefinder")
with col3:
    st.write(' ')

