import json
from typing import Optional, List
import os
import base64
import PyPDF2
import fitz  # PyMuPDF
from pydantic_ai import RunContext
import logging
import threading
import json
from agent_client import MQTTConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_client = MQTTConnector()
threading.Thread(target=local_client.connect, kwargs={"timeout": 2}, daemon=True).start()
topic = "do_experiment"
json_path = "bin\\Debug\\net8.0-windows\\reagent_layout.json"

class Deps:
    """Dependency container passed to the agent, not callable"""
    def __init__(self, send_event):
        self.send_event = send_event # async callback to push JSON to WebSocket

async def read_pdf(
    ctx: RunContext[Deps],
    file_path: str,
    page_number: Optional[int] = None
) -> str:
    """
    Extract text from a PDF. If page number is given, also render that page as an image and send it via the dependency
    callback. The image of the page reading will be shown in the window docked to the right of the web page
    """
    await ctx.deps.send_event({
        "type": "tool_call",
        "name": "read_pdf",
        "args": {"file_path": file_path, "page_number": page_number}
    })

    if not os.path.exists(file_path):
        err = f"File not found: {file_path}"
        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": err})
        return err

    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

            if page_number is not None:
                if 1 <= page_number <= num_pages:
                    page = reader.pages[page_number - 1]
                    text = page.extract_text() or ""

                    try:
                        doc = fitz.open(file_path)
                        page_img = doc[page_number - 1]
                        pix = page_img.get_pixmap()
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        await ctx.deps.send_event({
                            "type": "pdf_page_image",
                            "page": page_number,
                            "image": img_base64
                        })
                        doc.close()
                    except Exception as img_err:
                        await ctx.deps.send_event({
                            "type": "warning",
                            "content": f"Could not render page {page_number} as image: {img_err}. Please install PyMuPDF with 'pip install PyMuPDF'."
                        })
                else:
                    text = f"Page {page_number} out of range (1–{num_pages})."
            else:
                text = ""
                for i, page in enumerate(reader.pages):
                    text += f"\n--- Page {i+1} ---\n"
                    text += page.extract_text() or ""

        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": f"reading text: {text[:20]}…"})
        return text

    except Exception as e:
        err = f"Error reading PDF: {str(e)}"
        # await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": err})
        return err

def get_reagent(name:str, path = json_path) -> str:
    """
    Search through reagent_layout.json to find if the reagent we need is already loaded onto the experiment platform.
    Not callable, used in do_experiment()
    :param name: the reagent we want to use
    :param path: the path of reagent_layout.json
    :return: The position in the form of "BPxx" of the reagent on the platform if reagent found, otherwise, raise an
    error and return the description
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            points = data.get("Points", {})
            for point_id, info in points.items():
                reagent_name = info.get("name", "")

                if reagent_name == name:
                    return point_id
            return "Reagent is missing"
    except Exception as e:
        err = str(e)
        return err

async def do_experiment(
        ctx: RunContext[Deps],
        spin_speed:int = 3000,
        spin_acc:int = 1000,
        spin_dur:int = 30000,
        reagent:str = "",
        volume:int = 10
) -> str:
    """
    Tell the platform to conduct a single round of an in-situ spin coating experiment. This function will send all the
    parameters you have set one by one to the emqx server, and then it will pass them on to the platform to start the
    experiment.
    :param spin_speed: spin speed for spin coating, max 6000rpm, default 3000rpm
    :param spin_acc: acceleration of the spin coater, must be integer and default 1000rpm/s
    :param spin_dur: spin duration for spin coating in ms, default 30000ms
    :param reagent: Name of the reagent to be used this round.
    :param volume: The volume of the reagent to be dispensed onto substrate, default 10ul
    :return: Whether there is any errors. No errors will return an "Experiment started" message. If the reagent is not
    ready, will return "Reagent is missing". You can ask the scientists to check the spelling or change your spelling.
    If the server is not connected, will return "Connect server failed". You can ask the scientists to check emqx
    connection.
    """
    try:
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "do_experiment",
            "args": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume
            }
        })

        reagent_pos = get_reagent(reagent)
        if reagent_pos[:2] != "BP":
            return reagent_pos

        if local_client.is_connected:
            local_client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
            msg = (f"✅ Experiment started: seeking {reagent} at {reagent_pos}, {spin_speed} rpm, "
                   f"acc {spin_acc} rpm/s, duration {spin_dur} ms, volume {volume} µl.")
            await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})

            return msg
        else:
            connect_state = local_client.connect()
            if connect_state:
                local_client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
                msg = (f"✅ Experiment started: seeking {reagent} at {reagent_pos}, {spin_speed} rpm, "
                       f"acc {spin_acc} rpm/s, duration {spin_dur} ms, volume {volume} µl.")
                await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})

                return msg
            else:
                return "Connect server failed"
    except Exception as e:
        err = f"Error occurred: {str(e)}"
        return err

def execute_spin_coating(spin_speed: int, spin_acc: int, spin_dur: int, reagent: str, volume: int) -> str:
    payload = {
        "action": "do_experiment",
        "params": {
            "spin_speed": spin_speed,
            "spin_acc": spin_acc,
            "spin_dur": spin_dur,
            "reagent": reagent,
            "volume": volume
        }
    }
    try:
        if not local_client.check_connect():
            local_client.connect(timeout=2)
        local_client.publish("do_experiment", json.dumps(payload))
        return f"Successfully done!"
    except Exception as e:
        return f"ERROR: {str(e)}"

def execute_set_temperature(target: float) -> str:
    try:
        # cmd_list = ["./temp_ctrl", "--set", str(target)]
        # res = subprocess.run(cmd_list, capture_output=True, text=True)
        # return res.stdout.strip()
        return f"Successfully done!"
    except Exception as e:
        return f"ERROR: {str(e)}"

def execute_move_robot_arm(x: float, y: float, z: float) -> str:
    try:
        # res = subprocess.run(["python", "arm_ctrl.py", str(x), str(y), str(z)], capture_output=True, text=True)
        return f"Successfully done!"
    except Exception as e:
        return f"ERROR: {str(e)}"