# Black-Box GoodWords Attack

Attacker can only submit messages to the classifier and observe returned confidence scores or probability estimates. No access the model's architecture, parameters, or training data.

## Objective

find the smallest set of words that, when added to a spam message, minimizes the spam score returned by the classifier.

### Exploration vs. Exploitation Trade-off
Must balance between testing new, untested words to discover potentially effective additions (exploration) and using known effective words to achieve immediate evasion success (exploitation).

# Black-Box Core Components

The black-box attack scenario simulates realistic conditions where attackers have only query access to the target model. 

# Black-Box Adaptive Discovery Methods

We can employ adaptive learning techniques to efficiently discover effective words within query constraints. To this end, we'll implement epsilon-greedy selection and exponential moving averages to balance exploration of new candidates with exploitation of proven performers.

