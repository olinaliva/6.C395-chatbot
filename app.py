"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from src.chat import Chatbot
import pandas as pd

CUSTOM_CSS = """
/* Full-page background image with light overlay */
.gradio-container {
    background:
        linear-gradient(rgba(255,255,255,0.68), rgba(255,255,255,0.68)),
        url('/gradio_api/file=data/boston.jpg') center center / cover no-repeat fixed !important;
    min-height: 100vh !important;
}

/* Make the inner app panel readable */
.main, .wrap, .gap {
    background: transparent !important;
}

.contain, .block-container {
    max-width: 960px !important;
    width: 98vw !important;
    margin: 0 auto !important;
    border-radius: 14px !important;
    padding: 32px !important;
    background: rgba(255, 255, 255, 0.78) !important;
    backdrop-filter: blur(4px);
}

/* Title font */
h1 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    color: #00234b !important;
    margin-bottom: 0 !important;
}

/* Input box */
textarea {
    min-height: 80px !important;
    border: 2.5px solid #00234b !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}

textarea:focus {
    border-color: #1565c0 !important;
    box-shadow: 0 0 0 3px rgba(0,35,75,0.18) !important;
    outline: none !important;
}

/* Buttons — exclude icon/action buttons inside the chat */
button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.15s !important;
}
button:not([aria-label]):not(.icon-button):not([class*="icon"]) {
    border: 2px solid #00234b !important;
}
/* Explicitly clear borders on chat icon buttons */
button[aria-label], .icon-button, button[class*="icon"], button[class*="action"] {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

button.primary, button[class*="primary"] {
    background-color: #00234b !important;
    color: white !important;
    border-color: #00234b !important;
}

button.secondary, button[class*="secondary"] {
    background-color: #ffffff !important;
    color: #00234b !important;
    border-color: #00234b !important;
}

.examples button, button[class*="example"] {
    background-color: #e8f0fe !important;
    color: #00234b !important;
    border: 1.5px solid #1565c0 !important;
    border-radius: 20px !important;
}

footer {
    display: none !important;
}
"""

def create_chatbot():
    schools_df = pd.read_csv("data/schools.csv")
    languages_df = pd.read_csv("data/languages.csv")
    alternative_df = pd.read_csv("data/alternative.csv")
    chatbot = Chatbot(schools_df, languages_df, alternative_df)

    def chat(message, history):
        return chatbot.get_response(message, history)

    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.HTML('<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700;800&display=swap" rel="stylesheet">')
        gr.HTML("""
            <h1>
                Boston Public Schools
                <span style="font-weight: 300; font-size: 0.75em; color: #2a4a7f; display: block; letter-spacing: 0.04em;">
                    Enrollment Assistant
                </span>
            </h1>
        """)
        gr.HTML("<div style='margin-top: 20px;'></div>")
        gr.Markdown(
            "Ask about BPS schools, language programs, or alternative education options. "
            "Since this is a free-tier chatbot, you may see a 503 error when it's busy — "
            "just wait a few seconds and try again."
        )

        gr.HTML("""
            <button
                onclick="
                    var p = document.getElementById('info-panel');
                    var arrow = document.getElementById('info-arrow');
                    if (p.style.display === 'none') {
                        p.style.display = 'block';
                        arrow.textContent = '▴';
                    } else {
                        p.style.display = 'none';
                        arrow.textContent = '▾';
                    }
                "
                style="
                    background: white;
                    color: #00234b;
                    border: 2px solid #00234b;
                    border-radius: 8px;
                    padding: 4px 12px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    cursor: pointer;
                    margin-bottom: 8px;
                "
            >What I can help with <span id="info-arrow">▾</span></button>
            <div id="info-panel" style="display:none; color:#1f4e79; background:#eef6ff; padding:12px; border-radius:8px; margin-bottom:8px;">
                <ul>
                    <li>Finding schools by grade level, location, or type</li>
                    <li>Dual language programs (Spanish, Vietnamese, Chinese, Haitian Kreyòl, ASL)</li>
                    <li>Alternative &amp; re-engagement programs for flexible learners</li>
                    <li>General BPS enrollment questions</li>
                </ul>
                <p><strong>Data includes</strong> ~120 BPS schools, current as of the 2024–25 school year.</p>
                <blockquote style="color:#444;">
                    For official enrollment decisions, always confirm details directly with
                    <a href="https://www.bostonpublicschools.org/" target="_blank">Boston Public Schools</a>.
                </blockquote>
            </div>
        """)

        gr.ChatInterface(
            chat,
            examples=[
                "What schools offer Spanish dual language programs?",
                "What are my high school options in East Boston?",
                "Are there alternative programs for students who need flexible schedules?",
                "How do I enroll my kindergartener in BPS?",
            ],
        )

        gr.HTML("<div style='margin-top: 16px;'></div>")
        gr.Markdown(
            "_This chatbot is for informational purposes only. "
            "Contact BPS directly to confirm enrollment details._"
        )

    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue"),
        allowed_paths=["data"]
    )