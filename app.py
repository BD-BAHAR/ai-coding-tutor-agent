import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(token=HF_TOKEN)

# 🔥 UPDATED PROMPT FOR INTERACTIVE BEHAVIOR
SYSTEM_PROMPT = """
You are an AI coding tutor.

Your goal is to guide the student step by step — NOT to explain everything at once.

STRICT RULES:
- Ask ONLY ONE question at a time
- Do NOT give full solution unless student asks explicitly
- Do NOT give full code early
- Do NOT give long explanations
- WAIT for student response before next step

Teaching flow:
1. Ask what the problem means
2. Ask input/output
3. Ask logic idea
4. Ask for pseudocode
5. Then give small code hints (only if needed)
6. Help debug if student provides code

Style:
- Very short
- Interactive
- Conversational
- Encourage thinking

BAD:
❌ long explanation
❌ full steps at once
❌ full code immediately

GOOD:
✔ ask one question
✔ wait
✔ guide slowly

Example:
Student: Write program to add two numbers

You:
"Good question 😊  
What are the inputs in this problem?"
"""

def tutor_agent(message, history):
    if history is None:
        history = []

    if not message.strip():
        return history, ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,  # 🔥 limit response size
            temperature=0.6,
        )

        reply = response.choices[0].message.content

        # 🔥 FORCE SHORT RESPONSE (very important)
        reply = reply.split("\n\n")[0]

    except Exception as e:
        reply = f"ERROR:\n{str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("Ask a programming question. The tutor will guide you step-by-step (no direct answers).")

    chatbot = gr.Chatbot()

    msg = gr.Textbox(
        label="Your programming question",
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    submit = gr.Button("Submit")
    clear = gr.Button("Clear")

    submit.click(tutor_agent, [msg, chatbot], [chatbot, msg])
    msg.submit(tutor_agent, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()