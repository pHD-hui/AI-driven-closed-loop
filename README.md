# Closed-Loop Experimental Intelligence for Autonomous Materials Discovery and Optimization


Xujie Hui<sup>1</sup>, Wei Meng<sup>1</sup>, Kaixiang Lai<sup>1</sup>, Zhipeng Huang<sup>1</sup>, Feiyue Lu<sup>1</sup>, Jiahao Li<sup>1</sup>, Hongyu Zhang<sup>1</sup>, Jingyan Qi<sup>1</sup>, Ying Shang<sup>1</sup>, Zhipeng Yin<sup>1</sup>, Zhangyu Yuan<sup>1</sup>, Jialin Wu<sup>1</sup>, Ning Li<sup>1</sup>, <sup>2*</sup>

<sup>1</sup>Institute of Polymer Optoelectronic Materials and Devices, Guangdong Basic Research Center of Excellence for Energy and Information Polymer Materials, State Key Laboratory of Luminescent Materials and Devices, South China University of Technology, Guangzhou, China.
<sup>2</sup>Guangdong Provincial Key Laboratory of Luminescence from Molecular Aggregates, South China University of Technology, Guangzhou, 510640 China.
Email: ningli2022@scut.edu.cn
# Abstract
Materials experimentation increasingly operates in high-dimensional, strongly coupled design spaces, where conventional trial-and-error approaches are inefficient and non-scalable. While automation improves throughput and reproducibility, most platforms remain limited to execution-level automation, with data processing and experimental planning fragmented and human-mediated, thereby limiting adaptive decision-making for complex optimization. Here, we introduce a fully autonomous closed-loop experimental framework that integrates automated experimentation, real-time analysis, and model-guided decision-making into a continuous, machine-readable dataflow, enabling adaptive experimental design without human intervention. By iteratively feeding experimental results back into the decision loop, the system achieves efficient exploration and reliable convergence with substantially fewer experimental trials. Validation on the optimization of optoelectronic thin-films demonstrates accelerated convergence, improved performance and reproducibility, and effective balancing of exploration and exploitation in strongly coupled, non-monotonic parameter landscapes. This work establishes closed-loop experimental intelligence as a generalizable and practical paradigm for the discovery and optimization of advanced optoelectronic materials.

<p align="center"><img width="801" height="649" alt="image" src="https://github.com/user-attachments/assets/982e24aa-8ae1-4c69-8ded-9e551fd17763" /></p>

# Content
This repository contains the code used to implement the machine learning workflow described in the main manuscript and illustrated in Fig. 2.

The structure of this repository is as follows:
## Repository Structure

- `experiments/`  
  Folder containing the experiments completed by our platform, along with some test cases.

- `figures/`  
  Folder containing high-resolution figures as reported in the main paper, as well as instructions to reproduce them.

- `robotexperiments/`  
  Main module containing the core implementation of the package (`.py` files).

- `script/`  
  Folder containing scripts for:
  - setting up experiments
  - running experiments
  - plotting results  
  (see `experiments/`, `cycles/`, `plots/`)

- `environment.yml`  
  Environment configuration file used to install dependencies.

- `setup.py`  
  Installation script for the package.
  # Installation
