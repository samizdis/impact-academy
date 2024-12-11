# Auto-Enhance

![image](https://github.com/user-attachments/assets/24f50197-e7d9-47be-9449-a6599a5e4dec)


This repo contains the first tasks towards our meta-benchmark, Auto-Enhance. We measure the capability of "top-level" agents (i.e. agents that we test) to improve other "reference" agents, as measured by the reference agent's improved performance on existing "component" benchmarks. 
We build the tasks as [METR tasks](https://github.com/METR/task-standard).

Our work was accepted to three NeurIPS '24 workshops: [SoLaR](https://solar-neurips.github.io/), [SafeGenAi](https://safegenaiworkshop.github.io/) and [Towards Safe and Trustworthy Agents](https://www.mlsafety.org/events/neurips/2024). Check out the write-up [here](https://openreview.net/forum?id=8WM3sqWdQ4).

## Tasks

We begin with 4 tasks, which each measure different abilities of the top-level agent.

### Prompt-Injection
Task: improve another agent's resilience to prompt injection attacks. Based on the [CyberSecEval2](https://arxiv.org/abs/2404.13161) benchmark. 

[Task Implementation](./prompt_injection_uplift/README.md)

### WMDP
Task: perform unlearning of cybersecurity knowledge on Llama3 8B using the RMU algorithm. Based on the [WMDP](https://arxiv.org/abs/2403.03218) benchmark. 

[Task Implemnetation](./wmdp/README.md)

### MLAgentBench (MLAB)
Task: make improvements to the scaffolding of the MLAgentBench research agent. Based on the [MLAgentBench](https://arxiv.org/abs/2310.03302) benchmark.

[Task Implementation](./mlagentbench/README.md)

### SWE-Bench
Task: select the optimal LLM, which achieves best performance while operating a given scaffold to solve Github issues. Based on the [SWE-bench](https://arxiv.org/abs/2310.06770) benchmark.

[Task Implementation](./swe_bench_enhance/README.md)


## Setup
Make sure docker-engine is running. Then run:

```
git clone https://github.com/samizdis/impact-academy
cd drivers && npm install
cd ..\workbench && npm install
``` 
