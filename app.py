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
    "Problem Interpretation",
    "Input and Output Analysis",
    "Solution Strategy",
    "Code Construction",
    "Testing and Debugging",
    "Learning Reflection"
]

student_data = {
    "problem": "",
    "stage": 0,
    "chat": "",
    "responses": {s: "" for s in STAGES},
    "feedback": {s: "" for s in STAGES},
    "scores": {s: 0 for s in STAGES}
}

def add_chat(user_text, assistant_text):
    student_data["chat"] += f"\n\nStudent: {user_text}\nTutor: {assistant_text}"
    return student_data["chat"]

def get_feedback_and_score(stage, answer):
    prompt = f"""
You are an AI programming tutor.

Evaluate the student's response for this stage.

Stage: {stage}
Student Answer: {answer}

Return short feedback and a score from 1 to 5.

Format exactly:
Feedback: ...
Score: X
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.4,
        )

        text = response.choices[0].message.content.strip()

        score = 3
        for line in text.splitlines():
            if "Score" in line:
                for n in ["5", "4", "3", "2", "1"]:
                    if n in line:
                        score = int(n)
                        break

        return text, score

    except Exception as e:
        return f"Feedback unavailable: {str(e)}", 0

def get_hint():
    stage = STAGES[student_data["stage"]]

    prompt = f"""
You are helping a beginner programming student.

Current stage: {stage}

Give ONE short hint only.
Do not give full code.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Hint unavailable: {str(e)}"

def start_problem(problem):
    if not problem.strip():
        return student_data["chat"], "Please enter a programming problem first."

    student_data["problem"] = problem
    student_data["stage"] = 0
    student_data["chat"] = ""

    for s in STAGES:
        student_data["responses"][s] = ""
        student_data["feedback"][s] = ""
        student_data["scores"][s] = 0

    reply = (
        f"Great. We will solve this step by step.\n\n"
        f"Current Stage: 1/{len(STAGES)} - {STAGES[0]}\n\n"
        f"First, explain the problem in your own words."
    )

    chat = add_chat(problem, reply)
    status = f"Current Stage: 1/{len(STAGES)} - {STAGES[0]}"
    return chat, status

def check_answer(answer):
    if not student_data["problem"]:
        return student_data["chat"], "Please start with a programming problem first."

    if not answer.strip():
        return student_data["chat"], "Please write your answer first."

    stage = STAGES[student_data["stage"]]

    feedback, score = get_feedback_and_score(stage, answer)

    student_data["responses"][stage] = answer
    student_data["feedback"][stage] = feedback
    student_data["scores"][stage] = score

    reply = f"{feedback}\n\nScore: {score}/5"

    chat = add_chat(answer, reply)
    status = f"Current Stage: {student_data['stage'] + 1}/{len(STAGES)} - {stage}"
    return chat, status

def next_step():
    if not student_data["problem"]:
        return student_data["chat"], "Please start with a programming problem first."

    if student_data["stage"] < len(STAGES) - 1:
        student_data["stage"] += 1
        stage = STAGES[student_data["stage"]]

        questions = {
            "Input and Output Analysis": "What are the inputs and expected output for this problem?",
            "Solution Strategy": "What logic or steps will solve the problem before writing code?",
            "Code Construction": "Now write the first small part of the code.",
            "Testing and Debugging": "How will you test the program? Give at least one test case.",
            "Learning Reflection": "What did you learn from solving this problem?"
        }

        reply = (
            f"Now moving to Stage {student_data['stage'] + 1}/{len(STAGES)}: {stage}\n\n"
            f"{questions.get(stage, 'Continue with this stage.')}"
        )

        chat = add_chat("Next step", reply)
        status = f"Current Stage: {student_data['stage'] + 1}/{len(STAGES)} - {stage}"
        return chat, status

    reply = "You completed all stages. You can now generate the report."
    chat = add_chat("Next step", reply)
    return chat, "All stages completed"

def give_hint():
    if not student_data["problem"]:
        return student_data["chat"], "Please start with a programming problem first."

    hint = get_hint()
    chat = add_chat("Give me a hint", hint)
    status = f"Current Stage: {student_data['stage'] + 1}/{len(STAGES)} - {STAGES[student_data['stage']]}"
    return chat, status

def generate_report(student_name):
    if not student_name.strip():
        student_name = "Student"

    safe_name = student_name.replace(" ", "_")
    filepath = f"/tmp/{safe_name}_AI_Programming_Learning_Report.pdf"

    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    y = height - 50

    def write_line(text, size=10, bold=False):
        nonlocal y

        if y < 70:
            c.showPage()
            y = height - 50

        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)

        text = str(text).replace("\n", " ")
        max_chars = 90

        while len(text) > max_chars:
            split_at = text.rfind(" ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            c.drawString(50, y, text[:split_at])
            text = text[split_at:].strip()
            y -= 15

            if y < 70:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica-Bold" if bold else "Helvetica", size)

        c.drawString(50, y, text)
        y -= 15

    write_line("AI Programming Learning Companion Report", 16, True)
    write_line(f"Student Name: {student_name}", 11)
    write_line(f"Date: {datetime.now().strftime('%Y-%m-%d')}", 11)
    write_line(f"Problem: {student_data['problem']}", 11)
    y -= 10

    completed = 0

    for i, stage in enumerate(STAGES, start=1):
        write_line(f"Stage {i}: {stage}", 13, True)

        response = student_data["responses"][stage]
        feedback = student_data["feedback"][stage]
        score = student_data["scores"][stage]

        if response.strip():
            completed += 1

        write_line("Student Response:", 11, True)
        write_line(response if response else "Not completed")

        write_line("Assistant Feedback:", 11, True)
        write_line(feedback if feedback else "No feedback recorded")

        write_line(f"Score: {score}/5", 11, True)
        y -= 8

    scores = [student_data["scores"][s] for s in STAGES if student_data["scores"][s] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    write_line("Final Summary", 14, True)
    write_line(f"Completed Stages: {completed}/{len(STAGES)}")
    write_line(f"Average Score: {avg_score:.2f}/5")

    write_line("Suggestions for Growth:", 12, True)
    write_line("- Explain the problem clearly before writing code.")
    write_line("- Identify inputs and outputs early.")
    write_line("- Plan the logic before coding.")
    write_line("- Test the program with different cases.")
    write_line("- Reflect on what you learned and what errors you fixed.")

    c.save()
    return filepath

def reset_app():
    global student_data
    student_data = {
        "problem": "",
        "stage": 0,
        "chat": "",
        "responses": {s: "" for s in STAGES},
        "feedback": {s: "" for s in STAGES},
        "scores": {s: 0 for s in STAGES}
    }
    return "", "No problem started yet.", None, ""

with gr.Blocks() as demo:
    gr.Markdown("# AI Programming Learning Companion")
    gr.Markdown("A stage-based AI tutor that guides students step by step and generates a learning report.")

    student_name = gr.Textbox(label="Student Name", placeholder="Enter student name")

    problem = gr.Textbox(
        label="Programming Problem",
        placeholder="Example: Write a Java program to add two numbers",
        lines=2
    )

    start_btn = gr.Button("Start Problem")

    status = gr.Textbox(
        label="Stage Status",
        value="No problem started yet.",
        interactive=False
    )

    chat_display = gr.Textbox(
        label="Tutor Conversation",
        lines=18,
        interactive=False
    )

    answer = gr.Textbox(
        label="Your Answer",
        placeholder="Write your answer for the current stage here...",
        lines=4
    )

    with gr.Row():
        hint_btn = gr.Button("Give me a hint")
        check_btn = gr.Button("Check my answer")
        next_btn = gr.Button("Next step")
        report_btn = gr.Button("Generate report")
        clear_btn = gr.Button("Clear")

    report_file = gr.File(label="Download PDF Report")

    start_btn.click(start_problem, inputs=problem, outputs=[chat_display, status])
    hint_btn.click(give_hint, outputs=[chat_display, status])
    check_btn.click(check_answer, inputs=answer, outputs=[chat_display, status])
    next_btn.click(next_step, outputs=[chat_display, status])
    report_btn.click(generate_report, inputs=student_name, outputs=report_file)
    clear_btn.click(reset_app, outputs=[chat_display, status, report_file, answer])

if __name__ == "__main__":
    demo.launch()