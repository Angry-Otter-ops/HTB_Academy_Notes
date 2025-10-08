# Prompt Engineering
The most apparent (and ineffective) mitigation strategy is prompt engineering. This strategy refers to prepending the user prompt with a system prompt that tells the LLM how to behave and interpret the user prompt.

# Filter-based Mitigations
Just like traditional security vulnerabilities, filters such as whitelists or blacklists can be implemented as a mitigation strategy for prompt injection attacks. However, their usefulness and effectiveness are limited when it comes to LLMs. 

Blacklists, on the other hand, may make sense to implement. Examples could include:

    Filtering the user prompt to remove malicious or harmful words and phrases
    Limiting the user prompt's length
    Checking similarities in the user prompt against known malicious prompts such as DAN

# Limit the LLM's Access
The principle of least privilege applies to using LLMs just like it applies to traditional IT systems. If an LLM does not have access to any secrets, an attacker cannot leak them through prompt injection attacks. Therefore, an LLM should never be provided with secret or sensitive information.