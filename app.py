import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN)

SYSTEM_PROMPT = """
You are an AI coding tutor for beginner programming students.
Do NOT give the full code immediately.
Guide the student step by step with questions and hints.
Keep answers short, simple, and supportive.
"""

chat_history_text = ""

def tutor_agent(message):
    global chat_history_text

    if not message.strip():
        return chat_history_text, ""

    prompt = f"""
{SYSTEM_PROMPT}

Conversation so far:
{chat_history_text}

Student: {message}
Tutor:
"""

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=250,
            temperature=0.7,
            return_full_text=False
        )

        chat_history_text += f"\n\nStudent: {message}\n\nTutor: {response.strip()}"
        return chat_history_text, ""

    except Exception as e:
        error_text = f"\n\nStudent: {message}\n\nTutor: ERROR: {str(e)}"
        chat_history_text += error_text
        return chat_history_text, ""

def clear_chat():
    global chat_history_text
    chat_history_text = ""
    return ""

with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("Enter a programming problem. The tutor will guide you step by step instead of giving direct code.")

    chatbox = gr.Textbox(
        label="Tutor Chat",
        lines=18,
        interactive=False
    )

    msg = gr.Textbox(
        label="Your programming question",
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(tutor_agent, inputs=msg, outputs=[chatbox, msg])
    msg.submit(tutor_agent, inputs=msg, outputs=[chatbox, msg])
    clear.click(clear_chat, outputs=chatbox)

if __name__ == "__main__":
    demo.launch()