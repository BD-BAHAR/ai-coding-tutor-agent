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

def build_prompt(user_message, history):
    prompt = SYSTEM_PROMPT + "\n\n"

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"Student: {content}\n"
        elif role == "assistant":
            prompt += f"Tutor: {content}\n"

    prompt += f"Student: {user_message}\nTutor:"
    return prompt

def tutor_agent(message, history):
    if history is None:
        history = []

    if not message.strip():
        return "", history

    prompt = build_prompt(message, history)

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.7,
            return_full_text=False
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.strip()})
        return "", history

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": f"Error from Hugging Face model:\n\n{str(e)}"
        })
        return "", history

with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("Enter a programming problem. The tutor will guide you step by step instead of giving direct code.")

    chatbot = gr.Chatbot(label="Tutor Chat", type="messages")

    msg = gr.Textbox(
        label="Your programming question",
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(tutor_agent, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(tutor_agent, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: [], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()