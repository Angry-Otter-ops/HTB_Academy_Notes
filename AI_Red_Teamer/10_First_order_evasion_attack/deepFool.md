# DeepFool and the Quest for Minimality

**Fast Gradient Sign** Method demonstrated efficient adversarial generation through a single gradient step. 
- One backward pass.
- One step along the gradient sign. 

**DeepFool**
treats adversarial example generation as a geometric problem: find the shortest path from a data point to the closest decision boundary.

# Linear Classifiers and Optimal Projections
Formula:
```
r f(x)=wTx+b
```

finding the minimal L2 perturbation is straightforward. 

Breakdown:
- **f(xi)** - the network’s current output
- **∇f(xi)** -  the gradient (which tells how the output changes as each input feature is modified)
- **(x−xi)** - is the displacement from the current position

# Multi-Class Formulation

# The Overshoot Parameter
DeepFool includes an overshoot parameter (typically 0.02) that slightly overshoots the decision boundary: the actual step taken is (1+overshoot)×ri

# Comparison with Iterative FGSM

**Iterative FGSM** takes uniform steps in the direction of the gradient sign. 