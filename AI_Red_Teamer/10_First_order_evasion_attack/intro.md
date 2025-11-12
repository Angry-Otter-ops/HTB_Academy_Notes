**First-order attacks** use gradient information to craft adversarial examples. During normal training, gradients tell us how to adjust model weights to reduce error. During an attack, gradients tell us how to adjust inputs to increase error.

**FGSM (Fast Gradient Sign Method)** takes the direct approach. You decide upfront how much you're willing to change the input, then compute the gradient and move each pixel in the direction that increases loss.  

First-order attacks are also remarkably transferable. An adversarial example crafted against one model often fools other models, even those with different architectures or training procedures. This enables realistic attacks where the adversary doesn't need full access to the target model.