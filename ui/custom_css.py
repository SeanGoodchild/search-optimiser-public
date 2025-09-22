import streamlit as st

def inject_custom_styles():
    st.markdown("""
        <style>
            .stApp {
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
            .stButton button {
                background-color: #0c69ea;
                color: white;
                border-radius: 8px;
                font-weight: 600;
                border: none;
                padding: 0.5rem 1rem;
            }
            .stButton button:hover {
                background-color: #0959c4;  
            }
            a, .stMarkdown a {
                color: #42485c;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)
