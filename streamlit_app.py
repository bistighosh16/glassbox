import streamlit as st
import subprocess
import time
import socket
import os

# 1. Set page to full width and dark theme
st.set_page_config(page_title="GlassBox AI", layout="wide", initial_sidebar_state="collapsed")

# 2. Hide Streamlit's default menu/header to keep our Dark UI clean
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 0; margin: 0;}
    iframe {width: 100%; height: 100vh; border: none;}
    </style>
""", unsafe_allow_html=True)

# 3. Start your FastAPI server in the background
@st.cache_resource
def launch_glassbox():
    # Run the model weights download if they are missing
    if not os.path.exists("models/SmolLM2-135M/model.safetensors"):
        subprocess.run(["python", "download_model.py"])
    
    # Start the engine
    proc = subprocess.Popen(["python", "run_server.py"])
    
    # Wait for the server to wake up on port 8000
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1):
                break
        except:
            time.sleep(1)
    return proc

launch_glassbox()

# 4. Show your beautiful instrument panel inside an IFrame
st.components.v1.iframe("http://localhost:8000", height=900)