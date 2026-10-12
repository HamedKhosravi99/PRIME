"""The agents of the paper (the ReAct ladder, the comparators, the closest
external systems, and PRIME), their LLM clients, the scripted policies, and
the batch and episode drivers. The driver scripts import their siblings by
bare name (``import arms``), so they also run as ``python3 src/prime_agents/run_batch.py``."""
