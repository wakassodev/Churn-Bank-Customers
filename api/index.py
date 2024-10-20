from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import streamlit as st
from main import st as streamlit_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    # Run the Streamlit app and capture its output
    streamlit_script = streamlit_app._get_script_run_ctx().script_path
    streamlit_output = st.script_runner.get_script_output(streamlit_script)
    
    # Return the Streamlit app's HTML output
    return HTMLResponse(content=streamlit_output, status_code=200)
