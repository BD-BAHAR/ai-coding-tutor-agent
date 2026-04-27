import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

client = InferenceClient(
    model=MODEL_NAME,
    token=HF_TOKEN
)

SYSTEM_PROMPT = """
You are an AI coding tutor for beginner programming students.

A student will write a programming question.
Do NOT give the full code immediately.

Guide the student step by step:
1. Ask what the problem means.
2. Ask for input and output.
3. Ask about the logic.
4. Help with pseudocode.
5. Give small hints.
6. Help debug code.

Keep answers short, simple, and supportive.
"""

def build_prompt(user_message, history):
    prompt = SYSTEM_PROMPT + "\n\n"

    if history:
        for user, bot in history:
            prompt += f"Student: {user}\nTutor: {bot}\n"

    prompt += f"Student: {user_message}\nTutor:"
    return prompt

def tutor_agent(message, history):
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

        history.append((message, response.strip()))
        return "", history

    except Exception as e:
        error_message = f"""
Error from Hugging Face model:

{str(e)}

Check:
1. HF_TOKEN is added in Settings → Variables and secrets
2. Secret name is exactly HF_TOKEN
3. Restart the Space
4. Check Logs
"""
        history.append((message, error_message))
        return "", history

with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("Enter a programming problem. The tutor will guide you step by step instead of giving direct code.")

    chatbot = gr.Chatbot(label="Tutor Chat")
    msg = gr.Textbox(
        label="Your programming question",
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(
        tutor_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        tutor_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()