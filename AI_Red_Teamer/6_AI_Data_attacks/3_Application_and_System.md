# Model Deployment Tampering
Model deployment tampering attack can occur in various stages of the ML lifecycle.
- when models are transered
- when models are integrated
- when models are hosted on untrusted infrastructure. 



## Model Deployment Tampering Attacks
Can exploit broken access ontrol in the ML application to gain access to the model files or training data. If unauthorized access  to model files, they can directly alter the model’s internal parameters, such as weights and biases. 

## Compromising the Server Infrastructure
Security Issue types:

1. Misconfigured Management API enabling unauthorized remote access
2. Server-Side Request Forgery (SSRF) vulnerability enables downloading remote files
3. Usage of an insecure library containing a public deserialization vulnerability leading to remote code execution