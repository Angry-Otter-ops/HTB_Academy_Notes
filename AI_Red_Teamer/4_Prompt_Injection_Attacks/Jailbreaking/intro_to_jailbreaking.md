# Types of Jailbreak Prompts

There are different types of jailbreak prompts, each with a different idea behind it:

1. **Do Anything Now (DAN):** These prompts aim to bypass all LLM restrictions. There are many different versions and variants of DAN prompts. Check out this GitHub repository for a collection of DAN prompts.
2. **Roleplay:** The idea behind roleplaying prompts is to avoid asking a question directly and instead ask the question indirectly through a roleplay or fictional scenario. Check out this paper for more details on roleplay-based jailbreaks.
3. **Fictional Scenarios:** These prompts aim to convince the LLM to generate restricted information for a fictional scenario. By convincing the LLM that we are only interested in a fictional scenario, an LLM's resilience might be bypassed.
4. **Token Smuggling:** This technique attempts to hide requests for harmful or restricted content by manipulating input tokens, such as splitting words into multiple tokens or using different encodings, to avoid initial recognition of blocked words.
5. **Suffix & Adversarial Suffix:** Since LLMs are text completion algorithms at their core, an attacker can append a suffix to their malicious prompt to try to nudge the model into completing the request. Adversarial suffixes are advanced variants computed specifically to coerce LLMs to ignore restrictions. They often look non-nonsensical to the human eye. For more details on the adversarial suffix technique, check out this paper.
6. **Opposite/Sudo Mode:** Convince the LLM to operate in a different mode where restrictions do not apply.
