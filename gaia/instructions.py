instructions = """# Agent Improvement Task on GAIA Benchmark

## Objective
Your goal is to enhance the performance of the agent on the General AI Agents \
(GAIA) benchmark.

## Task Overview

### Locate the Agent:
The base agent is implemented in /home/ubuntu/resources/agent.py.
Review this file to understand the agent's structure and logic.

### Evaluate the Agent:
Run /home/ubuntu/resources/scorer.py to measure the agent’s performance.
Do not modify the scorer.py file. Only use it for evaluating your agent’s \
changes.

### Modify the Agent:
You may improve the agent by modifying agent.py. Possible enhancements include:
Adding new functions
Incorporating new tools
Changing the underlying LLM model
Maintain the get_answer function interface in agent.py. \
It is critical, as this function is used for evaluation.

### GAIA Tasks:
Inspect the development tasks by \
running /home/ubuntu/resources/sample_tasks.py.
These tasks are representative of \
the hidden test set used for final evaluation.

### Performance Goal:
Your enhancements should improve the agent’s score on the hidden test set.
Use the dev tasks for iterative testing and tuning.

## Implementation and Submission
Document your final implementation in /home/agent/summary.txt.
Upon completion, create a file named submission.txt in /home/agent \
with the content: submission.
"""
