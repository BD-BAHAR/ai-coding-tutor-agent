import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

MODEL_NAME = "meta-llama/Llama-3-8b-Instruct"   # ✅ very stable

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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})

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
    return "", history


with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("Enter a programming problem. The tutor will guide you step by step.")

    chatbot = gr.Chatbot()

    msg = gr.Textbox(
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(tutor_agent, [msg, chatbot], [msg, chatbot])
    msg.submit(tutor_agent, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()