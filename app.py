import streamlit as st
from subprocess import Popen  
import time
from features_extracted import extract_features,generate_image
import os

def main():
    """The main function for the Streamlit app."""

    st.title("AI Image Generator")

    theme = st.selectbox("Choose a Theme:", ["abstract", "nature", "fantasy", "other"])

    if theme == "other":
        custom_theme = st.text_input("Enter your custom theme:")

    if st.button("Draw Your Image"):
        with st.spinner("Opening drawing interface..."):  
            process = Popen(["python", "drawing.py"])  

        drawing_finished = False
        while not drawing_finished:
            if os.path.exists("drawing_finished.txt"): 
                drawing_finished = True
                os.remove("drawing_finished.txt")

        time.sleep(5)  
    
        features = extract_features("drawing.png", theme=custom_theme if theme == "other" else theme)
        generated_image = generate_image(features, theme=custom_theme if theme == "other" else theme)

        st.subheader("Generated Image")
        st.image(generated_image)

if __name__ == "__main__":
    main()