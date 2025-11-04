# Overview of Application & System Components

Components of AI Models:

1. Model
2. Data
3. Application
4. System

**MCP (Model Context Protocol)** is an open-source standard for connecting AI applications to external systems. MCP provides a standardized interface between LLM applications and external resources. As LLMs become increasingly capable, ensuring that these models can consistently interpret, retain, and apply context is critical for a satisfying performance. 
(https://modelcontextprotocol.io/docs/getting-started/intro)

What can MCP enable?

- Agents can access your Google Calendar and Notion, acting as a more personalized AI assistant.
- Claude Code can generate an entire web app using a Figma design.
- Enterprise chatbots can connect to multiple databases across an organization, empowering users to analyze data using chat.
- AI models can create 3D designs on Blender and print them out using a 3D printer.

## Application Component

The interface layer connecting users to the underlying model and its capabilities. It comprises all applications that the AI deployment interacts with, including web applications, mobile apps, APIs, databases, and integrated services such as plugins and autonomous agents. 

### Common application-component attacks:

- **Injection attacks**, such as SQL injection or command injection. Injection vulnerabilities can lead to loss of data or complete system takeover.
- **Access control vulnerabilities**, potentially enabling unauthorized attackers to access sensitive data or functionality.
- **Denial of ML-Service**, potentially impairing the availability of the AI deployment.
- **Rogue Actions**: If a model has excessive agency and can access functions or data it does not necessarily need to access, the model may trigger unintended actions impacting systems it interfaces with. Such rogue actions may result from malicious actions or inadvertently occur due to unexpected model interactions. For instance, if the model can issue arbitrary SQL queries in a connected database, a model response may result in data loss if all tables are dropped. Such a query can either be issued maliciously by an attacker or accidentally caused by unexpected user input.
- **Model Reverse Engineering**: An attacker may be able to replicate the model by analyzing inputs and outputs for a vast number of input data points. If the application does not implement a rate limit, malicious actors might frequently query the model to reverse engineer it.
- **Vulnerable Agents or Plugins**: Vulnerabilities in custom agents or plugins integrated into the deployment may perform unintended actions or exfiltrate model interactions to malicious actors.
Logging of sensitive data: If the application logs sensitive data from user input or model interactions, sensitive information may be disclosed to unauthorized actors through application logs.

## System Component

Encompasses everything infrastructure-related, including deployment platforms, code, data storage, and hardware. 

Common system-component vulnerabilities include:

- **Misconfigured Infrastructure**: If infrastructure used during training or inference is misconfigured to expose data or services to the public inadvertently, unauthorized actors may be able to steal training data, user data, the model itself, or configuration secrets.
- **Improper Patch Management**: Issues in an AI deployment application's patch management process may result in unpatched public vulnerabilities in different system components, from the operating system to the ML stack. These vulnerabilities can range from privilege escalation vectors to remote code execution flaws and may result in total system compromise.
- **Network Security**: Since generative AI deployments typically interface with different systems over internal networks, proper network security is crucial to mitigate security threats. Common network security measures include network segmentation, encryption, and monitoring to thwart lateral movement.
- **Model Deployment Tampering**: If threat actors manipulate the deployment process, they may be able to maliciously modify model behavior. They can achieve this by manipulating the source code or exploiting vulnerabilities.
Excessive Data Handling: Applications processing and storing data excessively may run into legal issues if this data includes user-related information. Furthermore, excessive data handling increases the impact of data storage vulnerabilities as more data is at risk of being leaked or stolen.

