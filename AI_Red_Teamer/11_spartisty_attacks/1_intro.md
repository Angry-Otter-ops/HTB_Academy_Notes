# Sparsity attacks 
Seek misclassification by changing as few input dimensions as possible.

```
‖xadv−x‖0=|{i∣(xadv)i≠xi}|
```

## Why Sparsity Attacks Matter

Sparse edits align with real constraints. An attacker may only be able to flip a few bits in a binary, touch a handful of pixels due to rendering limits, or change a small number of tokens in text. Sparse perturbations can be harder to detect with defenses tuned to global noise statistics, and they reveal which features the model treats as most decisive. For defenders, reproducing EAD and JSMA establishes baselines for L1‑induced sparsity and explicit L0 control, which together expose different failure modes than L∞ or L2 attacks.

## ElasticNet

The Elastic-net Attacks to Deep neural networks (EAD) creates perturbations that are simultaneously sparse (changing few pixels) and smooth (making small, coordinated changes)

ElasticNet attacks minimize a mixed-norm distance function combining L2 and L1 components. The attack seeks perturbations that cause misclassification while keeping both the total perturbation energy (via L2) and the number of modified pixels (via L1) small. The β parameter controls the relative importance of sparsity versus smoothness.

# Disctance Metrics

To balance sparsity against smoothness, ElasticNet needs three distance computations: L1 for counting total change, squared L2 for measuring energy, and their weighted combination for the actual optimization objective. 

