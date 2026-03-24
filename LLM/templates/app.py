from flask import Flask, request, jsonify, render_template, Response
import requests
import threading
import os
import base64
import json
import time
import csv
import io
import queue
import webbrowser
from threading import Timer
import asyncio
import re
import sys

# ==========================================
# 🔧 Path configuration and dependency injection
# ==========================================
# Add the hardware folder to the system path to ensure
# inter-module imports work properly
# (e.g., tool.py importing agent_client.py)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hardware'))

import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field, create_model
from typing import Literal, Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI  # Added: used for low-level native multimodal vision API calls

# Import specific execution functions from the hardware control module
from hardware.tools import execute_spin_coating, execute_set_temperature, execute_move_robot_arm

app = Flask(__name__)

# ==========================================
# ⚙️ Core configuration parameters
# ==========================================
SILICONFLOW_API_KEY = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"  # ⚠️ Please replace with your real API key
PDF_FOLDER = r"test"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"

# Model path definition
Base_URL = "https://api.siliconflow.cn/v1"
CHAT_API_URL = f"{Base_URL}/chat/completions"

# 🌟 Global model and client instances
# 1. Provider for Pydantic-AI
custom_provider = OpenAIProvider(base_url=Base_URL, api_key=SILICONFLOW_API_KEY)
ai_model = OpenAIChatModel(MODEL_NAME, provider=custom_provider)

# 2. Native AsyncOpenAI client
# (solves the issue that 'OpenAIProvider' has no 'chat' attribute)
async_openai_client = AsyncOpenAI(api_key=SILICONFLOW_API_KEY, base_url=Base_URL)

task_queue = queue.Queue()
task_running = False
cancel_requested = False


# ==========================================
# 🛠️ Helper functions
# ==========================================
def pdf_page_to_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ==========================================
# 🧠 Core: asynchronous dynamic literature extraction
# (based on Pydantic-AI)
# ==========================================
async def async_process_pdf_library(task_description: str, fields: list):
    global task_running, cancel_requested

    # 1. Dynamically generate the Pydantic schema
    fields_def = {f: (str, Field(description=f"Extract and highly condense: {f}")) for f in fields}
    DynamicRecord = create_model('DynamicRecord', **fields_def)

    save_dir = "extract"
    os.makedirs(save_dir, exist_ok=True)
    all_extracted_data = []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)

    for file_idx, filename in enumerate(pdf_files):
        if cancel_requested:
            break

        pdf_path = os.path.join(PDF_FOLDER, filename)
        doc_id = os.path.splitext(filename)[0]
        try:
            with fitz.open(pdf_path) as doc:
                num_pages = len(doc)
            task_queue.put({"type": "progress", "message": f"Processing document {file_idx + 1}/{total_files}: {filename}"})

            for page_num in range(num_pages):
                if cancel_requested:
                    break

                img_base64 = pdf_page_to_image(pdf_path, page_num)
                task_queue.put(
                    {"type": "page_reading", "data": {"filename": filename, "page": page_num + 1, "image": img_base64}}
                )

                # 3. 🌟 Use the native AsyncOpenAI client to process the image
                try:
                    response = await async_openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are a rigorous literature data-cleaning expert. Current task: [{task_description}]\nPlease extract information accurately and strictly output a JSON-format array. Each element of the array must contain the following fields: {fields}. If no matching content is found, output an empty array []. Never output markdown markers."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Extract information from this literature page:"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                                ]
                            }
                        ],
                        temperature=0.1
                    )

                    # Remove possible markdown residue from the model output
                    content = response.choices[0].message.content.strip()
                    clean_json = re.sub(r'```json\n|\n```|```', '', content).strip()

                    try:
                        extracted_list = json.loads(clean_json)
                    except json.JSONDecodeError:
                        extracted_list = []

                    # 4. Use the dynamic Pydantic model for rigorous validation and cleaning
                    for item in extracted_list:
                        if not isinstance(item, dict):
                            continue

                        try:
                            # Use the previously defined DynamicRecord for validation
                            # and filter hallucinated fields
                            record = DynamicRecord(**item)
                            record_dict = record.model_dump()
                            record_dict['_source_doc'] = doc_id
                            all_extracted_data.append(record_dict)

                            task_queue.put({
                                "type": "finding",
                                "data": {
                                    "page": page_num + 1,
                                    "filename": filename,
                                    "details": record_dict
                                }
                            })
                        except Exception as validation_err:
                            print(f"Pydantic validation skipped non-compliant data: {validation_err}")

                except Exception as e:
                    print(f"Page extraction exception: {e}")
                    task_queue.put({"type": "warning", "message": f"Page {page_num + 1} extraction exception: {str(e)}"})

                time.sleep(2.0)
        except Exception as e:
            task_queue.put({"type": "error", "message": f"Failed to process {filename}: {str(e)}"})

    # Export CSV
    os.makedirs("extract", exist_ok=True)
    csv_filename = os.path.join(save_dir, f"Extraction_{time.strftime('%Y%m%d-%H%M%S')}.csv")

    # 🌟 Extension: save both an archived file and a temporary file
    temp_csv = os.path.join(save_dir, "extraction.csv")

    all_keys = list(fields) + ['_source_doc']

    for target_file in [csv_filename, temp_csv]:
        with open(target_file, 'w', newline='', encoding='utf-8') as csvfile:
            if all_extracted_data:
                writer = csv.DictWriter(csvfile, fieldnames=all_keys)
                writer.writeheader()
                for row in all_extracted_data:
                    writer.writerow({k: row.get(k, '') for k in all_keys})
            else:
                csvfile.write(",".join(fields))

    task_queue.put({"type": "complete", "csv": csv_filename, "count": len(all_extracted_data), "fields": fields})
    task_running = False


def process_pdf_library_thread(task_desc, fields):
    """Bridge Flask's synchronous environment and asyncio"""
    global task_running, cancel_requested
    task_running = True
    cancel_requested = False
    asyncio.run(async_process_pdf_library(task_desc, fields))


# ==========================================
# 🌐 Flask route design
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/task_stream')
def task_stream():
    def event_stream():
        while True:
            try:
                msg = task_queue.get(timeout=2)
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get("type") == "complete":
                    break
            except queue.Empty:
                if not task_running:
                    break
                yield ": heartbeat\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/api/cancel_task', methods=['POST'])
def cancel_task():
    global cancel_requested
    cancel_requested = True
    return jsonify({"status": "stopping"})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file received'}), 400
    os.makedirs(PDF_FOLDER, exist_ok=True)
    saved_files = []
    for file in request.files.getlist('files'):
        if file.filename.lower().endswith('.pdf'):
            file.save(os.path.join(PDF_FOLDER, file.filename))
            saved_files.append(file.filename)
    return jsonify({'status': 'success', 'saved': saved_files})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    action = data.get('action', 'chat')

    # 🌟 Workflow: after the user confirms the fields,
    # start the Pydantic-AI extraction thread
    if action == 'start_extraction':
        task_desc, fields = data.get('task_desc'), data.get('fields')
        while not task_queue.empty():
            task_queue.get()
        threading.Thread(target=process_pdf_library_thread, args=(task_desc, fields)).start()
        return jsonify({'type': 'task_trigger', 'reply': "Instruction confirmed! Launching the parsing engine now. Real-time progress is shown below..."})

    # 🌟 Intercept extraction command: let the model infer fields
    if user_message.startswith("Help me search for:"):
        global task_running
        if task_running:
            return jsonify({'type': 'system', 'reply': "⚠️ There is already an extraction task running."})

        task_desc = user_message.replace("Help me search for:", "").strip()
        if not task_desc:
            task_desc = "Passivators specialized for the FAPbI3 perovskite system"
            fields = ["Passivator Name", "Original Sentence", "Mechanism"]
            while not task_queue.empty():
                task_queue.get()
            threading.Thread(target=process_pdf_library_thread, args=(task_desc, fields)).start()
            return jsonify({'type': 'task_trigger', 'reply': f"Default instruction detected. FAPbI3 passivator parsing has been started..."})
        else:
            class FieldAnalysis(BaseModel):
                fields: list[str] = Field(description="List of inferred data column names to extract")

            field_agent = Agent(
                OpenAIChatModel("Qwen/Qwen2.5-72B-Instruct", provider=custom_provider),
                system_prompt="You are a data analysis expert. Based on the user's task description, infer which data column names should be extracted.",
                output_type=FieldAnalysis
            )

            try:
                result = asyncio.run(field_agent.run(f"Task: {task_desc}"))
                fields = result.data.fields
            except Exception as e:
                fields = ["Extraction Target", "Detailed Parameters"]
                print(f"Field inference error: {e}")

            return jsonify({
                'type': 'field_confirm',
                'task_desc': task_desc,
                'fields': fields,
                'reply': f"To complete the extraction, I have planned the following table headers for you:\n`{', '.join(fields)}`\nPlease confirm:"
            })

    # 🌟 Intercept hardware control
    # (based on Pydantic-AI tool calls)
    if user_message.startswith("Hardware control:"):
        cmd_text = user_message.replace("Hardware control:", "").strip()

        # 🌟 Fix: pass the specific execution functions from the hardware module
        # through a list into tools
        hw_agent = Agent(
            ai_model,
            system_prompt="You are a laboratory hardware control agent. Based on the user's instruction, precisely call the appropriate low-level tools (spin coating, temperature control, robotic arm) to complete the task. Return a concise Chinese execution report.",
            tools=[execute_spin_coating, execute_set_temperature, execute_move_robot_arm]
        )
        try:
            result = asyncio.run(hw_agent.run(cmd_text))
            return jsonify({'type': 'system', 'reply': f"🔧 **Hardware Scheduling Result**\n\n{result.data}"})
        except Exception as e:
            return jsonify({'type': 'system', 'reply': f"❌ Hardware scheduling exception: {str(e)}"})

    # 🌟 Intercept software algorithms
    # (based on Pydantic-AI forced structured output)
    if user_message.startswith("Optimization algorithm:"):
        cmd_text = user_message.replace("Optimization algorithm:", "").strip()

        class AlgoDecision(BaseModel):
            action: Literal["call_existing", "generate_new"] = Field(description="Choose whether to call an existing algorithm or generate new code")
            algo_name: Optional[str] = Field(description="If call_existing, specify the algorithm name (e.g. bayes_opt)")
            code: Optional[str] = Field(description="If generate_new, provide the complete Python code")
            reason: str = Field(description="Explanation of the AI decision")

        sw_agent = Agent(
            ai_model,
            system_prompt="You are a top-tier algorithm engineer. You can call existing algorithms (in the software folder), or write a new Python analysis script from scratch according to the requirements.",
            output_type=AlgoDecision
        )

        try:
            decision = asyncio.run(sw_agent.run(cmd_text)).data

            if decision.action == "call_existing":
                return jsonify({
                    'type': 'system',
                    'reply': f"⚙️ **Calling preset algorithm** `{decision.algo_name}`\nReason: {decision.reason}"
                })

            elif decision.action == "generate_new":
                os.makedirs("software", exist_ok=True)
                script_path = os.path.join("software", "dynamic_generated.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(decision.code)

                run_res = subprocess.run(["python", script_path], capture_output=True, text=True)
                output_msg = run_res.stdout if run_res.returncode == 0 else run_res.stderr
                return jsonify({
                    'type': 'system',
                    'reply': f"✨ **Dynamically generated code executed successfully**\n\n**Decision reason**: {decision.reason}\n**Execution log**:\n```text\n{output_msg.strip()}\n```"
                })

        except Exception as e:
            return jsonify({'type': 'system', 'reply': f"❌ Algorithm routing exception: {str(e)}"})

    # 🌟 Normal chat streaming output
    def generate_chat():
        payload = {
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "messages": [{"role": "user", "content": user_message}],
            "stream": True
        }
        try:
            response = requests.post(
                CHAT_API_URL,
                headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
                json=payload,
                stream=True,
                timeout=30
            )
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: ") and "[DONE]" not in decoded:
                        try:
                            content = json.loads(decoded[6:])['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except:
                            pass
        except Exception as e:
            yield f"\n[Network request failed: {str(e)}]"

    return Response(generate_chat(), content_type='text/plain; charset=utf-8')


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == '__main__':
    Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000, threaded=True)