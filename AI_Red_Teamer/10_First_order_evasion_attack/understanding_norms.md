# Norm

A mathematical tool that assigns a "size" or "length" to changes we make to an input.

## The Three Rules Every Norm Must Follow

1. **Zero means zero**: The only thing with zero length is... nothing. If your measurement says something has zero size, it better actually be zero. This prevents our "ruler" from lying to us.
2.**Doubling means doubling**: If you make a change twice as big, the measurement should also be twice as big. This keeps our measurements consistent and predictable.
3. **Shortcuts don't exist**: The direct path between two points is never longer than going the roundabout way. In math terms, this is called the "triangle inequality" - imagine a triangle where one side can't be longer than the other two sides combined.

**p-norms** - The "p" is just a number that determines which family member we're using

Formula:
```
∥x∥p=(∑i=1n|xi|p)1/p
```

which means "Take each change, raise it to the power p, add them all up, then take the p-th root." 

## L0 Norm
The **L0 norm** is the simplest conceptually - it just counts how many pixels (or features) you changed. Period. It doesn’t care if you changed a pixel by a tiny amount or completely flipped it from black to white. Changed is changed.

Formula:
```
∥δ∥0=∑i𝟙[δi≠0]
```

## L1 Norm
The **L1 norm** adds up the absolute value of all changes. It’s like having a budget for how much total change you can make, and you can spread it around however you want.

Formula:
```
∥δ∥1=∑i|δi|
```

## L2 Norm
The **L2 norm** is the famous Euclidean distance - straight-line distance in space. It measures perturbations by taking the square root of the sum of squared changes:

Formula:
```
∥δ∥2=∑iδi2
```

### Advantages:

    Creates the smoothest, most imperceptible perturbations
    Mathematically well‑behaved (convex and smooth away from 0). In practice, many methods use the squared L2 norm, which is differentiable everywhere and especially convenient for optimization.
    Well-understood optimization properties

### Disadvantages:

    Changes everything at least a little bit
    Less interpretable - harder to see which features matter most
    May create a visible "haze" over the entire image
