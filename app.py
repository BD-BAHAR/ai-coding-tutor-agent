import os
import gradio as gr
from huggingface_hub import InferenceClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(token=HF_TOKEN)

# 🔥 NEW STAGES
STAGES = [
    "Problem Interpretation",
    "Input and Output Analysis",
    "Solution Strategy",
    "Code Construction",
    "Testing and Debugging",
    "Learning Reflection"
]

# 🔥 STATE
student_data = {
    "problem": "",
    "stage": 0,
    "responses": {s: "" for s in STAGES},
    "feedback": {s: "" for s in STAGES},
    "scores": {s: 0 for s in STAGES}
}

# 🔥 PROMPTS
def get_feedback_and_score(stage, response):
    prompt = f"""
Evaluate a student's answer.

Stage: {stage}
Answer: {response}

Return:
Feedback (1-2 lines)
Score (1-5)

Format:
Feedback: ...
Score: X
"""
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        text = res.choices[0].message.content

        score = 3
        for i in range(1,6):
            if f"{i}" in text:
                score = i

        return text, score
    except:
        return "Feedback unavailable", 3


def get_hint(stage):
    prompt = f"Give a short hint for stage: {stage}"
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80
    )
    return res.choices[0].message.content


# 🔥 MAIN FUNCTIONS

def submit_problem(problem, chat):
    student_data["problem"] = problem
    student_data["stage"] = 0
    return chat + [
        {"role": "assistant",
         "content": f"Stage 1: {STAGES[0]}\nExplain the problem in your own words."}
    ]


def check_answer(answer, chat):
    stage = STAGES[student_data["stage"]]

    feedback, score = get_feedback_and_score(stage, answer)

    student_data["responses"][stage] = answer
    student_data["feedback"][stage] = feedback
    student_data["scores"][stage] = score

    return chat + [
        {"role": "assistant",
         "content": f"{feedback}\nScore: {score}/5"}
    ]


def next_stage(chat):
    if student_data["stage"] < len(STAGES)-1:
        student_data["stage"] += 1
        stage = STAGES[student_data["stage"]]
        return chat + [
            {"role": "assistant",
             "content": f"Stage {student_data['stage']+1}: {stage}\nContinue..."}
        ]
    else:
        return chat + [{"role": "assistant", "content": "All stages completed. Generate report."}]


def hint(chat):
    stage = STAGES[student_data["stage"]]
    h = get_hint(stage)
    return chat + [{"role": "assistant", "content": f"Hint: {h}"}]


# 🔥 PDF
def generate_pdf(name):
    filename = f"/tmp/{name}_report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)

    y = 750

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "AI Programming Learning Companion Report")
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Student: {name}")
    y -= 20
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    y -= 30

    for s in STAGES:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, s)
        y -= 15

        c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Response: {student_data['responses'][s]}")
        y -= 15
        c.drawString(50, y, f"Feedback: {student_data['feedback'][s]}")
        y -= 15
        c.drawString(50, y, f"Score: {student_data['scores'][s]}/5")
        y -= 25

    avg = sum(student_data["scores"].values()) / len(STAGES)

    c.drawString(50, y, f"Final Score: {avg:.2f}/5")
    c.save()

    return filename


# 🔥 UI
with gr.Blocks() as demo:
    gr.Markdown("# AI Programming Learning Companion")

    name = gr.Textbox(label="Student Name")
    problem = gr.Textbox(label="Problem")

    chatbot = gr.Chatbot(type="messages")

    answer = gr.Textbox(label="Your Answer")

    btn_submit = gr.Button("Start Problem")
    btn_check = gr.Button("Check my answer")
    btn_next = gr.Button("Next step")
    btn_hint = gr.Button("Give me a hint")
    btn_report = gr.Button("Generate report")

    file = gr.File()

    btn_submit.click(submit_problem, [problem, chatbot], chatbot)
    btn_check.click(check_answer, [answer, chatbot], chatbot)
    btn_next.click(next_stage, chatbot, chatbot)
    btn_hint.click(hint, chatbot, chatbot)
    btn_report.click(generate_pdf, name, file)

demo.launch()