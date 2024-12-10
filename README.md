# Auto-Enhance

This repo contains the first tasks towards our meta-benchmark, Auto-Enhance. We measure the capability of top-level agents (i.e. agents that we test) to improve other reference agents. We build our tasks starting from existing agentic or non-agentic benchmarks. 
We build the tasks as [METR tasks](https://github.com/METR/task-standard). 

Our work was accepted to the SoLaR, SafeGenAi and SATA NeurIPS workshops. Check out the write-up [here](https://openreview.net/forum?id=8WM3sqWdQ4&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FWorkshop%2FSoLaR%2FAuthors%23your-submissions)).

## Tasks

### Prompt-Injection
This task measures the top-level agent's ability to improve another agent's resilience to prompt injection attacks. Based on the CyberSecEval2 benchmark. 

[Task Implementation](./prompt_injection_uplift/README.md)

### WMDP
This task measures the top-level agent's ability to perform unlearning of cybersecurity knowledge on Llama3 8B using the RMU algorithm. Based on the WMDP benchmark. 

[Task Implemnetation](./wmdp/README.md)

### MLAgentBench (MLAB)
This task measures the top-level agent's ability to make improvements to the scaffolding of the MLAgentBench research agent. Based on MLAgentBench.

[Task Implementation](./mlagentbench/README.md)

### SWE-Bench
This task measures the top-level agent's ability to select the optimal LLM to operate the scaffold and obtain the better performance on solving Github issues. Based on the SWE-bench benchmark.

[Task Implementation](./swe_bench_enhance/README.md)


## Setup
Make sure docker-engine is running. Then run:

```
git clone https://github.com/samizdis/impact-academy
cd drivers && npm install
cd ..\workbench && npm install
``` 