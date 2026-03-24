# Closed-Loop Experimental Intelligence for Autonomous Materials Discovery and Optimization


Xujie Hui<sup>1</sup>, Wei Meng<sup>1</sup>, Kaixiang Lai<sup>1</sup>, Zhipeng Huang<sup>1</sup>, Feiyue Lu<sup>1</sup>, Jiahao Li<sup>1</sup>, Hongyu Zhang<sup>1</sup>,Ziwen Mo<sup>1</sup>  Jingyan Qi<sup>1</sup>, Ying Shang<sup>1</sup>, Zhipeng Yin<sup>1</sup>, Zhangyu Yuan<sup>1</sup>, Jialin Wu<sup>1</sup>, Ning Li<sup>1</sup>, <sup>2*</sup>

<sup>1</sup>Institute of Polymer Optoelectronic Materials and Devices, Guangdong Basic Research Center of Excellence for Energy and Information Polymer Materials, State Key Laboratory of Luminescent Materials and Devices, South China University of Technology, Guangzhou, China.
<sup>2</sup>Guangdong Provincial Key Laboratory of Luminescence from Molecular Aggregates, South China University of Technology, Guangzhou, 510640 China.

# Abstract

<p style="text-align: justify;">
Materials experimentation increasingly operates in high-dimensional, strongly coupled design spaces, where conventional trial-and-error approaches are inefficient and non-scalable. While automation improves throughput and reproducibility, most platforms remain limited to execution-level automation, with data processing and experimental planning fragmented and human-mediated, thereby limiting adaptive decision-making for complex optimization. Here, we introduce a fully autonomous closed-loop experimental framework that integrates automated experimentation, real-time analysis, and model-guided decision-making into a continuous, machine-readable dataflow, enabling adaptive experimental design without human intervention. By iteratively feeding experimental results back into the decision loop, the system achieves efficient exploration and reliable convergence with substantially fewer experimental trials. Validation on the optimization of optoelectronic thin-films demonstrates accelerated convergence, improved performance and reproducibility, and effective balancing of exploration and exploitation in strongly coupled, non-monotonic parameter landscapes. This work establishes closed-loop experimental intelligence as a generalizable and practical paradigm for the discovery and optimization of advanced optoelectronic materials.
</p>

<p align="center"><img width="801" height="649" alt="image" src="https://github.com/user-attachments/assets/982e24aa-8ae1-4c69-8ded-9e551fd17763" /></p>


# Content
This repository contains the code used to implement the machine learning workflow described in the main manuscript and illustrated in Fig. 2.

The structure of this repository is as follows:

| File Path | Role | Key Capabilities |
|-----------|------|-----------------|
| `BO` |           |                                |
| `templates/index.html` | Frontend interface | Multi-mode interaction, PDF preview, progress visualization, task control |
| `app.py` | Main Flask server | PDF parsing, LLM integration, data extraction/storage, task scheduling, hardware integration |
| `hardware_controller.py` | Hardware control core | Reagent position parsing, experiment instruction integration, LLM routing, low-level hardware control |
| `agent_client.py` | EMQX client module | MQTT communication, experiment command dispatch |
| `temporal/extraction.csv` | Temporary data file | Stores latest extracted parameters for hardware module |
| `extract/` | Archive directory | Timestamped storage of historical results |
| `reagent_layout.json` | Reagent config | Stores physical positions of reagents (BPxx format) |


## ⚙️ Environment Setup

###  1. Create Virtual Environment
```bash
conda create -n SDL_agent python=3.10 -y
conda activate SDL_agent
```
###  2. Install Dependencies
```bash
pip install -r requirements.txt
# flask==2.3.3
# pymupdf==1.23.22
# pillow==10.1.0
# requests==2.31.0
# paho-mqtt==1.6.1
# python-dotenv==1.0.0
```
###  3. Key Configuration Items 
Modify the following configurations in app.py to adapt to the local environment:
```bash
# LLM API Configuration
SILICONFLOW_API_KEY = "your SiliconFlow API key"
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# PDF storage directory
PDF_FOLDER = r"本地PDF文件夹路径"

# MQTT server configuration (hardware/tool.py)
class Client_Conf:
    def __init__(self):
        self.client_id = "your_custom_client_id"
        self.usr_name = "your_mqtt_username"
        self.password = "your_mqtt_password"
        self.ip = "your_mqtt_server_ip"
        self.port = 1883
```

## 🚀 Quick Start

1. Configure API key, PDF directory, and MQTT server settings.
2. Start the Flask server:

```bash
python app.py
```
3.Open your browser and navigate to:

```bash
[python app.py](http://127.0.0.1:5000)
```

Then enter the "AI Lab Smart Control Panel" interface.

# Contributing

Contributions to AI-driven-closed-loop are welcome! The following individuals are currently involved in the project:

Kaixiang Lai

Zhipeng Huang 

Ziwen Mo

Xujie Hui






