import os
import gradio as gr
from huggingface_hub import InferenceClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(token=HF_TOKEN)

STAGES = [
    "Task Understanding",
    "Input–Output Mapping",
    "Logic Planning",
    "Guided Coding",
    "Testing & Improvement",
    "Learning Reflection"
]

SYSTEM_PROMPT = """
You are an interactive AI programming tutor.

Your job is to guide the student through one stage at a time.

Stages:
1. Task Understanding
2. Input–Output Mapping
3. Logic Planning
4. Guided Coding
5. Testing & Improvement
6. Learning Reflection

STRICT RULES:
- Ask only ONE question at a time.
- Do NOT give full code immediately.
- Do NOT explain all stages at once.
- Keep responses short and interactive.
- Give feedback on the student's answer.
- Move to the next stage only when the student gives a reasonable response.
- If the student asks for direct code, first encourage them to try.
- If they are stuck, give a small hint.
"""

student_data = {
    "student_name": "",
    "problem": "",
    "stage": 0,
    "responses": {stage: "" for stage in STAGES},
    "feedback": {stage: "" for stage in STAGES}
}

def detect_feedback(stage, student_message):
    prompt = f"""
You are evaluating a beginner programming student's response.

Current stage: {stage}
Student response: {student_message}

Give very short feedback in 1-2 sentences.
Do not give full solution.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a concise programming tutor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=120,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Feedback unavailable: {str(e)}"

def tutor_reply(message, history):
    if history is None:
        history = []

    if not message.strip():
        return history, ""

    current_stage = STAGES[student_data["stage"]]

    if student_data["problem"] == "":
        student_data["problem"] = message
        reply = (
            f"Great. We will solve this step by step.\n\n"
            f"Stage 1: {STAGES[0]}\n"
            f"Can you explain the problem in your own words?"
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return history, ""

    feedback = detect_feedback(current_stage, message)
    student_data["responses"][current_stage] += message + "\n"
    student_data["feedback"][current_stage] = feedback

    if student_data["stage"] < len(STAGES) - 1:
        student_data["stage"] += 1
        next_stage = STAGES[student_data["stage"]]

        next_question_map = {
            "Input–Output Mapping": "What are the inputs and expected output for this problem?",
            "Logic Planning": "What steps or logic will solve this problem before writing code?",
            "Guided Coding": "Now try writing a small part of the code. What would you write first?",
            "Testing & Improvement": "How will you test your program? Give one valid and one invalid/test case.",
            "Learning Reflection": "What did you learn from solving this problem?"
        }

        reply = (
            f"{feedback}\n\n"
            f"Now move to Stage {student_data['stage'] + 1}: {next_stage}\n"
            f"{next_question_map[next_stage]}"
        )
    else:
        reply = (
            f"{feedback}\n\n"
            "Excellent. You completed all stages. Now you can generate your PDF report."
        )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""

def generate_pdf(student_name):
    if not student_name.strip():
        student_name = "Student"

    filename = f"{student_name.replace(' ', '_')}_CS_Learning_Companion_Report.pdf"
    filepath = f"/tmp/{filename}"

    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    y = height - 50

    def write_line(text, size=10, bold=False):
        nonlocal y
        if y < 70:
            c.showPage()
            y = height - 50

        font = "Helvetica-Bold" if bold else "Helvetica"
        c.setFont(font, size)

        max_chars = 90
        lines = []
        while len(text) > max_chars:
            split_at = text.rfind(" ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            lines.append(text[:split_at])
            text = text[split_at:].strip()
        lines.append(text)

        for line in lines:
            c.drawString(50, y, line)
            y -= 15

    write_line("CS Learning Companion Report", 16, True)
    write_line(f"Student Name: {student_name}", 11)
    write_line(f"Date: {datetime.now().strftime('%Y-%m-%d')}", 11)
    write_line(f"Programming Problem: {student_data['problem']}", 11)
    y -= 10

    for i, stage in enumerate(STAGES, start=1):
        write_line(f"Stage {i}: {stage}", 13, True)
        write_line("Student Response:", 11, True)
        write_line(student_data["responses"][stage] or "Not completed")
        write_line("Assistant Feedback:", 11, True)
        write_line(student_data["feedback"][stage] or "No feedback recorded")
        y -= 8

    write_line("Final Summary", 14, True)
    completed = sum(1 for stage in STAGES if student_data["responses"][stage].strip())

    write_line(f"Completed Stages: {completed} out of {len(STAGES)}")
    write_line("Performance Assessment:", 12, True)

    if completed == len(STAGES):
        write_line("Problem-Solving Progress: 5 - Excellent structured progress.")
        write_line("Coding Development: 5 - Completed guided coding and testing stages.")
        write_line("Learning Reflection: 5 - Completed reflection stage.")
    elif completed >= 4:
        write_line("Problem-Solving Progress: 4 - Good progress with minor missing parts.")
        write_line("Coding Development: 4 - Mostly complete.")
        write_line("Learning Reflection: 4 - Reflection or testing may need more detail.")
    else:
        write_line("Problem-Solving Progress: 3 - Some stages need more work.")
        write_line("Coding Development: 3 - Continue working through the guided steps.")
        write_line("Learning Reflection: 3 - More explanation is needed.")

    write_line("Suggestions for Growth:", 12, True)
    write_line("- Continue explaining the problem before writing code.")
    write_line("- Practice identifying inputs, outputs, and logic first.")
    write_line("- Test your program with different cases.")
    write_line("- Reflect on what errors you faced and how you fixed them.")

    c.save()
    return filepath

def reset_all():
    global student_data
    student_data = {
        "student_name": "",
        "problem": "",
        "stage": 0,
        "responses": {stage: "" for stage in STAGES},
        "feedback": {stage: "" for stage in STAGES}
    }
    return [], ""

with gr.Blocks() as demo:
    gr.Markdown("# AI Coding Tutor Agent")
    gr.Markdown("This tutor guides students step by step and generates a final learning report.")

    student_name = gr.Textbox(
        label="Student Name",
        placeholder="Enter student name"
    )

    chatbot = gr.Chatbot()

    msg = gr.Textbox(
        label="Programming Problem / Response",
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    submit = gr.Button("Submit")
    clear = gr.Button("Clear")
    pdf_button = gr.Button("Generate PDF Report")
    pdf_output = gr.File(label="Download Report")

    submit.click(tutor_reply, [msg, chatbot], [chatbot, msg])
    msg.submit(tutor_reply, [msg, chatbot], [chatbot, msg])
    clear.click(reset_all, outputs=[chatbot, msg])
    pdf_button.click(generate_pdf, inputs=student_name, outputs=pdf_output)

if __name__ == "__main__":
    demo.launch()