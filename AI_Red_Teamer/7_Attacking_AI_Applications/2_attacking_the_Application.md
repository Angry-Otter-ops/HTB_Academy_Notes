# Model Reverse Engineering

Model reverse engineering is an attack on an AI application in which an adversary attempts to reconstruct or approximate the deployed model. By systematically sending inputs to the model through an exposed API and observing the outputs, the adversary collects enough input-output data points to train a *surrogate model* that mimics the original model's behavior. 

Potential risk factors: 
- Intellectual Property Theft
- Model inversion attacks (the adversary attempts to reconstruct sensitive information about the training data.)

## Reverse Engineering an ML Model

### The Classifier


## Denial of ML Service

**Sponge Examples**, as introduced in the (https://arxiv.org/pdf/2006.03463) paper, are specifically crafted adversarial inputs that maximize energy consumption and latency in the ML mode.

Two principles for text-based sponge:

- **The output sequence length** If a model generates more output tokens, more processing power is required to generate these tokens. As such, sponge examples aim to result in a response that is as long as possible. 
- **The number of input tokens**. Before processing a text response, each model represents the input as tokens. These tokens are learned and optimized in the training process to increase efficiency. Generally, if the number of tokens is larger, the model needs to process more data, which will require more processing power. 

## Mitigations

- Rate limiting
- Anomaly detection
- Query monitoring

# Insecure Integrated Components

# Rogue Actions

**rogue actions** refer to unintended behaviors or operations carried out via system extensions, such as LLM plugins or agents. 