import streamlit as st

st.set_page_config(page_title="My First AI App", page_icon="🤖")
st.title("🤖 Welcome to my AI App")
st.write("This is my very first Streamlit application.")

# A simple interactive button
if st.button("Click me!"):
   st.balloons()
   st.success("Streamlit is working perfectly!")
