import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3-8b-Instruct"

client = InferenceClient(token=HF_TOKEN)

SYSTEM_PROMPT = """
You are an AI coding tutor for beginner programming students.

Do NOT give full code immediately.

Guide step-by-step:
1. Ask what the problem means
2. Ask input/output
3. Ask logic
4. Help with pseudocode
5. Give small hints
6. Help debug code

Be simple, short, and supportive.
"""

def tutor_agent(message, history):
    if history is None:
        history = []

    if not message.strip():
        return history, ""

    # Build messages for model
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )

        reply = response.choices[0].message.content

    except Exception as e:
        reply = f"ERROR:\n{str(e)}"

    history.append((message, reply))
    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("Enter a programming problem. The tutor will guide you step by step.")

    chatbot = gr.Chatbot()  # ✅ NO type argument

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
        outputs=[chatbot, msg]   # ⚠️ important order
    )

    msg.submit(
        tutor_agent,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )

    clear.click(
        lambda: [],
        inputs=None,
        outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch()