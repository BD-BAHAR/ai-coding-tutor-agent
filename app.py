import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(token=HF_TOKEN)

SYSTEM_PROMPT = """
You are an AI coding tutor for beginner programming students.

A student will write a programming question, such as:
"Write a Java program to add two numbers."

Your job is to guide the student step by step.
Do NOT give the full code immediately.

Follow this teaching flow:
1. Ask the student to explain the problem in their own words.
2. Help the student identify input and output.
3. Guide the student to think about the logic.
4. Help the student write pseudocode.
5. Give small code hints only when needed.
6. If the student submits code, help debug it.

Rules:
- Do not give full solution first.
- Ask guiding questions.
- Give hints instead of direct answers.
- Keep the language simple.
- Encourage the student.
"""

def tutor_agent(message, history):
    if not message.strip():
        return ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role and content:
                    messages.append({"role": role, "content": content})
            else:
                user_msg, bot_msg = item
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"""
Error connecting to model.

Check:
- HF_TOKEN is set correctly
- Space restarted

Error:
{str(e)}
"""

demo = gr.ChatInterface(
    fn=tutor_agent,
    title="AI Coding Tutor Agent",
    description="Enter a programming problem. The tutor will guide you step by step instead of giving direct code.",
    textbox=gr.Textbox(
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    ),
)

if __name__ == "__main__":
    demo.launch()