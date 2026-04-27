import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient()

SYSTEM_PROMPT = """
You are an AI coding tutor for beginner students.

A student will write a programming question, such as:
"Write a Java program to swap two numbers."

Your job is NOT to give the full answer immediately.
Guide the student step by step.

Follow this process:
1. Ask the student to explain the problem in their own words.
2. Help them identify input and output.
3. Help them think about the logic.
4. Help them write pseudocode.
5. Give small code hints only when needed.
6. If they submit code, help debug it.

Rules:
- Do not give full code first.
- Ask questions.
- Give hints.
- Encourage thinking.
- Keep the language simple.
"""

def tutor_agent(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=tutor_agent,
    title="AI Coding Tutor Agent",
    description="Enter a programming problem. The tutor will guide you step by step instead of giving direct code.",
    textbox=gr.Textbox(
        placeholder="Example: Write a Java program to swap two numbers",
        lines=2
    )
)

demo.launch()