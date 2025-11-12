import os
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


# Import common utilities from HTB Evasion Library
from htb_ai_library import (
    set_reproducibility,
    SimpleCNN,
    get_mnist_loaders,
    mnist_denormalize,
    train_model,
    evaluate_accuracy
)

# Configure reproducibility
set_reproducibility(1337)

# Configure computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data loaders using library function (normalized space)
train_loader, test_loader = get_mnist_loaders(batch_size=128, normalize=True)

# Initialize model using library's SimpleCNN
model = SimpleCNN().to(device)

# Train the model using library function
trained_model = train_model(model, train_loader, test_loader, epochs=1, device=device)

# Evaluate baseline accuracy using library function
baseline_acc = evaluate_accuracy(trained_model, test_loader, device)
print(f"Baseline test accuracy: {baseline_acc:.2f}%")

def _forward_and_loss(model: nn.Module, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Forward pass and cross-entropy loss without side effects.

    Args:
        model: Neural network classifier
        x: Input images tensor
        y: Target labels tensor

    Returns:
        tuple[Tensor, Tensor]: Model logits and scalar loss value
    """
    if getattr(model, "training", False):
        raise RuntimeError("Expected model.eval() for attack computations to avoid BN/Dropout state updates")
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return logits, loss

def _input_gradient(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Return gradient of loss with respect to input tensor x.

    Args:
        model: Neural network in evaluation mode
        x: Input images to compute gradients for
        y: True labels for loss computation

    Returns:
        Tensor: Gradient tensor with same shape as x
    """
    x_req = x.clone().detach().requires_grad_(True)
    _, loss = _forward_and_loss(model, x_req, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return x_req.grad.detach()

def fgsm_attack(model: nn.Module,
                images: Tensor,
                labels: Tensor,
                epsilon: float,
                targeted: bool = False) -> Tensor:

    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not images.is_floating_point():
        raise ValueError("images must be floating point tensors")

    grad = _input_gradient(model, images, labels)
    step_dir = -1.0 if targeted else 1.0
    x_adv = images + step_dir * epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, MNIST_NORM_MIN, MNIST_NORM_MAX)
    return x_adv.detach()

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
# Epsilon in normalized space (≈0.25 in pixel space)
epsilon = 0.8
with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv = fgsm_attack(model, images, labels, epsilon)
with torch.no_grad():
    adv_pred = model(x_adv).argmax(dim=1)

originally_correct = (clean_pred == labels)
flipped = (adv_pred != labels) & originally_correct
success = flipped.sum().item() / max(int(originally_correct.sum().item()), 1)
print(f"FGSM flips (first batch): {success:.2%}")

def _norm_params(images: Tensor, mean: list, std: list) -> tuple[Tensor, Tensor]:
    """Convert normalization parameters to broadcastable tensors.

    Args:
        images: Input images tensor with shape (N, C, H, W)
        mean: Normalization mean per channel as list
        std: Normalization std per channel as list

    Returns:
        tuple[Tensor, Tensor]: Mean and std tensors with shape (1, C, 1, 1)
    """
    device, dtype, C = images.device, images.dtype, images.shape[1]
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    if mean_t.shape[1] != C or std_t.shape[1] != C:
        raise ValueError("mean/std channels must match images")
    return mean_t, std_t

def fgsm_pixel_space(model: nn.Module,
                     images: Tensor,
                     labels: Tensor,
                     epsilon: float,
                     mean: list,
                     std: list,
                     targeted: bool = False) -> Tensor:
    """FGSM for pixel-space inputs attacking normalized models.

    This variant accepts images in [0,1] pixel space rather than normalized
    space. It normalizes inputs internally for the model, converts gradients
    back to pixel space, and returns adversarials in [0,1] pixel space.

    Args:
        model: Model expecting normalized inputs
        images: Clean images in [0,1] pixel space (unnormalized)
        labels: Target labels
        epsilon: Max perturbation in pixel space (e.g., 8/255)
        mean: Normalization mean per channel
        std: Normalization std per channel
        targeted: If True, minimize loss towards labels

    Returns:
        Tensor: Adversarial images in [0,1] pixel space (unnormalized)
    """
    mean_t, std_t = _norm_params(images, mean, std)
    x = images.clone().detach()
    x_norm = (x - mean_t) / std_t
    x_norm.requires_grad_(True)

    _, loss = _forward_and_loss(model, x_norm, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    # Convert gradient from normalized space to image space
    grad_img = x_norm.grad / std_t
    step_dir = -1.0 if targeted else 1.0
    x_adv = torch.clamp(x + step_dir * epsilon * grad_img.sign(), 0.0, 1.0)
    return x_adv.detach()

# Example: Starting with pixel-space images
epsilon_px = 8 / 255  # pixel-space epsilon (≈0.031)
mean, std = [0.1307], [0.3081]

# Denormalize existing normalized images to get pixel-space images
mean_t, std_t = _norm_params(images, mean, std)
pixel_images = images * std_t + mean_t
pixel_images = torch.clamp(pixel_images, 0.0, 1.0)

# Attack in pixel space
x_adv_pixel = fgsm_pixel_space(model, pixel_images, labels, epsilon_px, mean, std)

# x_adv_pixel is in [0,1] and can be displayed or saved directly
# If you need to pass to the model again, normalize it first:
x_adv_norm = (x_adv_pixel - mean_t) / std_t

from typing import Dict

def evaluate_attack(model: nn.Module,
                   clean_images: Tensor,
                   adversarial_images: Tensor,
                   true_labels: Tensor) -> Dict[str, float]:
    """Compute accuracy, success rate, confidence shift, and norms.

    Args:
        model: Evaluated classifier in evaluation mode
        clean_images: Clean inputs in the model's expected domain (e.g., normalized MNIST)
        adversarial_images: Adversarial counterparts in the same domain as `clean_images`
        true_labels: Ground-truth labels

    Returns:
        Dict[str, float]: Aggregated metrics summarizing attack impact
    """
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean_images)
        adv_logits = model(adversarial_images)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)
        clean_correct = (clean_pred == true_labels)
        adv_correct = (adv_pred == true_labels)

        originally_correct = clean_correct
        flipped = (~adv_correct) & originally_correct
        conf_clean = clean_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        conf_adv = adv_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        l2 = (adversarial_images - clean_images).view(clean_images.size(0), -1).norm(p=2, dim=1)
        linf = (adversarial_images - clean_images).abs().amax()

        return {
            "clean_accuracy": clean_correct.float().mean().item(),
            "adversarial_accuracy": adv_correct.float().mean().item(),
            # Success rate among originally correct samples only
            "attack_success_rate": (
                flipped.float().sum() / originally_correct.float().sum().clamp_min(1.0)
            ).item(),
            "avg_clean_confidence": conf_clean.mean().item(),
            "avg_adv_confidence": conf_adv.mean().item(),
            "avg_confidence_drop": (conf_clean - conf_adv).mean().item(),
            "avg_l2_perturbation": l2.mean().item(),
            "max_linf_perturbation": linf.item(),
        }
# Assume images, labels, x_adv from the Core Implementation section
metrics = evaluate_attack(model, images, x_adv, labels)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


# Assume images, labels, x_adv from the Core Implementation section
metrics = evaluate_attack(model, images, x_adv, labels)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
import matplotlib.pyplot as plt
import numpy as np

# Colors imported from library
from htb_ai_library import (
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

def _style_axes(ax: plt.Axes) -> None:
    """Apply Hack The Box dark theme to an axes instance.

    Args:
        ax: Matplotlib axes to style
    """
    ax.set_facecolor(NODE_BLACK)
    ax.tick_params(colors=HACKER_GREY)
    for spine in ax.spines.values():
        spine.set_color(HACKER_GREY)
    ax.grid(True, color=HACKER_GREY, linestyle="--", alpha=0.25)

def visualize_attack(model: nn.Module,
                    image: Tensor,
                    label: Tensor,
                    make_adv,
                    title: str,
                    num_classes: int = 10,
                    targeted: bool = False,
                    target_class: int | None = None) -> None:
    """HTB-styled visualization for adversarial examples.

    Args:
        model: Classifier in evaluation mode
        image: Single image in normalized space, shape (C,H,W)
        label: Scalar true label tensor
        make_adv: Callable (model, image_batch, label_batch) -> adv_batch in normalized space
        title: Figure title
        num_classes: Number of classes to show in probability bars
        targeted: Whether the attack is targeted
        target_class: Optional target class to annotate
    """
    model.eval()
    dev = next(model.parameters()).device
    image_dev = image.to(dev)
    label_dev = label.to(dev)

    # Compute clean predictions
    with torch.no_grad():
        clean_probs = F.softmax(model(image_dev.unsqueeze(0)), dim=1).squeeze(0)
        clean_pred = int(clean_probs.argmax().item())

    # Generate adversarial example
    x_adv_dev = make_adv(model, image_dev.unsqueeze(0), label_dev.unsqueeze(0)).squeeze(0)
    perturbation_dev = x_adv_dev - image_dev

    # Compute adversarial predictions
    with torch.no_grad():
        adv_probs = F.softmax(model(x_adv_dev.unsqueeze(0)), dim=1).squeeze(0)
        adv_pred = int(adv_probs.argmax().item())

    # Denormalize for visualization
    image_vis = mnist_denormalize(image_dev.unsqueeze(0)).squeeze(0).detach().cpu()
    x_adv_vis = mnist_denormalize(x_adv_dev.unsqueeze(0)).squeeze(0).detach().cpu()
    perturbation_vis = (x_adv_vis - image_vis)
    # Create figure with grid layout
    fig = plt.figure(figsize=(16, 10), facecolor=NODE_BLACK)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    # Original image panel
    ax1 = fig.add_subplot(gs[0, 0])
    _style_axes(ax1)
    if image_vis.shape[0] == 1:
        ax1.imshow(image_vis.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax1.imshow(image_vis.permute(1, 2, 0))
    ax1.set_title(f"Original | class={clean_pred} | p={clean_probs[clean_pred]:.2%}",
                  color=HTB_GREEN, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Adversarial image panel
    ax2 = fig.add_subplot(gs[0, 1])
    _style_axes(ax2)
    if x_adv_vis.shape[0] == 1:
        ax2.imshow(x_adv_vis.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax2.imshow(x_adv_vis.permute(1, 2, 0))
    title_color = MALWARE_RED if adv_pred != int(label.item()) else HTB_GREEN
    adv_title = f"Adversarial | class={adv_pred} | p={adv_probs[adv_pred]:.2%}"
    if targeted and target_class is not None:
        adv_title += f" | target={target_class}"
    ax2.set_title(adv_title, color=title_color, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Perturbation panel (scaled for visibility)
    ax3 = fig.add_subplot(gs[0, 2])
    _style_axes(ax3)
    pert_scaled = (perturbation_vis * 10 + 0.5).clamp(0, 1)
    if pert_scaled.shape[0] == 1:
        ax3.imshow(pert_scaled.squeeze(0), cmap='gray', vmin=0, vmax=1)
    else:
        ax3.imshow(pert_scaled.permute(1, 2, 0))
    ax3.set_title("Perturbation (x10)", color=NUGGET_YELLOW, fontweight="bold")
    ax3.set_xticks([])
    ax3.set_yticks([])
    # Class probability comparison
    ax4 = fig.add_subplot(gs[1, :])
    _style_axes(ax4)
    x = np.arange(num_classes)
    width = 0.4
    ax4.bar(x - width/2, clean_probs[:num_classes].cpu(), width,
            color=AZURE, label="clean")
    ax4.bar(x + width/2, adv_probs[:num_classes].cpu(), width,
            color=MALWARE_RED, label="adv")
    ax4.set_xlabel("Class", color=WHITE)
    ax4.set_ylabel("Probability", color=WHITE)
    legend = ax4.legend(facecolor=NODE_BLACK, edgecolor=HACKER_GREY)
    for text in legend.get_texts():
        text.set_color(WHITE)
    ax4.set_title("Class probabilities", color=HTB_GREEN, fontweight="bold")
    for text in ax4.get_xticklabels() + ax4.get_yticklabels():
        text.set_color(HACKER_GREY)

    # Add main title and display
    fig.suptitle(title, color=HTB_GREEN, fontweight="bold", fontsize=24, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()
def visualize_fgsm_attack(model: nn.Module,
                         image: Tensor,
                         label: Tensor,
                         epsilon: float,
                         num_classes: int = 10,
                         targeted: bool = False,
                         target_class: int | None = None) -> None:
    """Wrapper for visualize_attack using FGSM.

    Args:
        model: Classifier model
        image: Single image tensor
        label: True label
        epsilon: Perturbation budget
        num_classes: Classes to display
        targeted: If True, targeted attack
        target_class: Target class for targeted attack
    """
    def _make_adv(m, xb, yb):
        if targeted and target_class is None:
            raise ValueError("target_class must be provided when targeted=True")
        y_used = yb if not targeted else torch.full_like(yb, target_class)
        return fgsm_attack(m, xb, y_used, epsilon, targeted=targeted)

    mode = "Targeted" if targeted else "Untargeted"
    visualize_attack(model, image, label, _make_adv,
                    title=f"FGSM {mode}",
                    num_classes=num_classes,
                    targeted=targeted,
                    target_class=target_class)
eps_candidates = [0.5, 0.8, 1.0]
success_image, success_label, success_eps = None, None, None

model.eval()
candidate, candidate_label = None, None

for xb, yb in test_loader:
    xb, yb = xb.to(device), yb.to(device)
    match_indices = (yb == 1).nonzero(as_tuple=True)[0]
    if len(match_indices) == 0:
        continue

    # Check predictions for all digit 1s in this batch
    with torch.no_grad():
        preds = model(xb[match_indices]).argmax(dim=1)
        correct_mask = (preds == 1)
        if correct_mask.any():
            # Take first correctly classified digit 1
            local_idx = correct_mask.nonzero(as_tuple=True)[0][0].item()
            idx = match_indices[local_idx].item()
            candidate = xb[idx]
            candidate_label = yb[idx]
            break

if candidate is None:
    raise RuntimeError("Could not find a correctly classified digit 1 in test set")

target_label = torch.tensor([7], device=device)

for eps_try in eps_candidates:
    x_adv = fgsm_attack(
        model,
        candidate.unsqueeze(0),
        target_label,
        epsilon=eps_try,
        targeted=True,
    )
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()
    print(f"epsilon={eps_try:.2f} -> predicted {pred}")

    if pred == 7:
        success_image = candidate
        success_label = candidate_label
        success_eps = eps_try
        break
if success_image is None:
    raise RuntimeError("Targeted FGSM did not achieve 1 -> 7 within the tested epsilons.")

_ = visualize_fgsm_attack(
    model,
    success_image.detach().cpu(),
    success_label.detach().cpu(),
    success_eps,
    targeted=True,
    target_class=7,
)
def iterative_fgsm(model: nn.Module,
                   images: Tensor,
                   labels: Tensor,
                   epsilon: float,
                   num_iter: int,
                   alpha: float | None = None,
                   targeted: bool = False,
                   random_start: bool = False) -> Tensor:
    """Iterative FGSM (Basic Iterative Method) with projection.

    Args:
        model: Target classifier in evaluation mode
        images: Clean images (normalized)
        labels: Ground-truth or target labels
        epsilon: L_infinity budget (in normalized space)
        num_iter: Number of iterations
        alpha: Step size per iteration (defaults to epsilon/T)
        targeted: If True, targeted attack
        random_start: If True, initialize within the epsilon ball

    Returns:
        Tensor: Adversarial images (normalized)
    """
    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if alpha is None:
        alpha = epsilon / max(num_iter, 1)
    if random_start:
        torch.manual_seed(1337)
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(images + delta, MNIST_NORM_MIN, MNIST_NORM_MAX)
    else:
        x_adv = images.clone()

    for _ in range(num_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        step_dir = -1.0 if targeted else 1.0
        x_adv = x_adv + step_dir * alpha * x_adv.grad.sign()
        x_adv = torch.clamp(images + (x_adv - images).clamp(-epsilon, epsilon), MNIST_NORM_MIN, MNIST_NORM_MAX)

    return x_adv.detach()
# Assume model, test_loader, device from FGSM Setup
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

epsilon = 0.8
num_iter = 10
alpha = epsilon / num_iter  # alpha = 0.08

with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv_ifgsm = iterative_fgsm(
    model, images, labels,
    epsilon=epsilon,
    num_iter=num_iter,
    alpha=alpha,
    targeted=False,
    random_start=True
)

with torch.no_grad():
    adv_pred_ifgsm = model(x_adv_ifgsm).argmax(dim=1)

originally_correct = clean_pred == labels
flipped_ifgsm = (adv_pred_ifgsm != labels) & originally_correct
print(
    f"I-FGSM flips (first batch): "
    f"{(flipped_ifgsm.float().sum() / originally_correct.float().sum().clamp_min(1.0)).item():.2%}"
)
# Reuse evaluate_attack function from the Evaluation Metrics section
metrics_ifgsm = evaluate_attack(model, images, x_adv_ifgsm, labels)
for k, v in metrics_ifgsm.items():
    print(f"{k}: {v:.4f}")
def visualize_ifgsm(model: nn.Module,
                    image: Tensor,
                    label: Tensor,
                    epsilon: float,
                    num_iter: int,
                    targeted: bool = False,
                    target_class: int | None = None) -> None:
    """Wrapper for visualize_attack using I-FGSM.

    Args:
        model: Classifier model
        image: Single image tensor [C,H,W]
        label: True label
        epsilon: Perturbation budget
        num_iter: Number of iterations
        targeted: If True, targeted attack
        target_class: Target class for targeted attacks
    """
    alpha = epsilon / max(num_iter, 1)

    def _make_adv(m, xb, yb):
        y_used = yb if not targeted else torch.full_like(yb, target_class)
        return iterative_fgsm(
            m, xb, y_used,
            epsilon, num_iter, alpha,
            targeted=targeted,
            random_start=True
        )

    mode = "Targeted" if targeted else "Untargeted"
    visualize_attack(
        model, image, label, _make_adv,
        title=f"I-FGSM {mode}",
        targeted=targeted,
        target_class=target_class
    )

# Visualize first sample from test batch
_ = visualize_ifgsm(
    model,
    images[0].detach().cpu(),
    labels[0].detach().cpu(),
    epsilon,
    num_iter,
    targeted=False
)
# Find one sample of '1'
one_img, one_lbl = None, None
for xb, yb in test_loader:
    m = (yb == 1)
    if m.any():
        j = m.nonzero(as_tuple=True)[0][0].item()
        one_img = xb[j].to(device)
        one_lbl = yb[j].to(device)
        break

# Try increasing epsilon values until successful
for eps_try in [0.5, 0.8, 1.0]:
    x_adv = iterative_fgsm(
        model,
        one_img.unsqueeze(0),
        torch.tensor(7, device=device).unsqueeze(0),  # target label
        epsilon=eps_try,
        num_iter=num_iter,
        alpha=eps_try / max(num_iter, 1),
        targeted=True,
        random_start=True,
    )
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()
    print(f"epsilon={eps_try:.2f} -> predicted {pred}")

    if pred == 7:
        _ = visualize_ifgsm(
            model,
            one_img.detach().cpu(),
            one_lbl.detach().cpu(),
            eps_try,
            num_iter,
            targeted=True,
            target_class=7,
        )
        break
# Compare FGSM (one-step) and I-FGSM on the same batch
# Run both attacks with same epsilon
epsilon = 0.7
x_adv_fgsm = fgsm_attack(model, images, labels, epsilon)
x_adv_ifgsm = iterative_fgsm(
    model, images, labels,
    epsilon, num_iter=10,
    random_start=True
)

# Compare success rates
with torch.no_grad():
    fgsm_pred = model(x_adv_fgsm).argmax(dim=1)
    ifgsm_pred = model(x_adv_ifgsm).argmax(dim=1)

orig_correct = clean_pred == labels
fgsm_success = (
    ((fgsm_pred != labels) & orig_correct).float().sum()
    / orig_correct.float().sum().clamp_min(1.0)
)
ifgsm_success = (
    ((ifgsm_pred != labels) & orig_correct).float().sum()
    / orig_correct.float().sum().clamp_min(1.0)
)

print(f"FGSM success rate: {fgsm_success:.1%}")
print(f"I-FGSM success rate: {ifgsm_success:.1%}")
print(f"Improvement: {(ifgsm_success - fgsm_success) / fgsm_success:.1%}")

from htb_ai_library import (
    set_reproducibility,
    MNISTClassifierWithDropout,
    get_mnist_loaders,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
    analyze_model_confidence,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

set_reproducibility(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNISTClassifierWithDropout is imported from htb_ai_library
# The architecture internally defines:
# - Conv1: 1->32 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - Conv2: 32->64 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - FC1: 3136->128, ReLU, 50% dropout
# - FC2: 128->10 (logits)

model = MNISTClassifierWithDropout().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

model_path = 'output/mnist_model.pth'
os.makedirs('output', exist_ok=True)

# Try loading cached model
if os.path.exists(model_path):
    print(f"Found cached model at {model_path}")
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()

    # Validate cached model
    _, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Cached model accuracy: {accuracy:.2f}%")

    if accuracy < 90.0:
        print("Accuracy below threshold, retraining required")
        model = None
else:
    model = None

# Train if needed
if model is None:
    print("Training new model...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64, normalize=True)
    model = MNISTClassifierWithDropout().to(device)

    model = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device
    )
    # Evaluate and cache
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

    save_model({
        'model': model,
        'architecture': 'MNISTClassifierWithDropout',
        'accuracy': accuracy,
        'training_config': {
            'epochs': 5,
            'batch_size': 64,
            'device': str(device)
        }
    }, model_path)

def deepfool(image: torch.Tensor,
             net: nn.Module,
             num_classes: int = 10,
             overshoot: float = 0.02,
             max_iter: int = 50,
             device: str = 'cuda') -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
    """
    Generate minimal adversarial perturbation using DeepFool algorithm.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        net (nn.Module): Target neural network in evaluation mode
        num_classes (int): Number of top-scoring classes to consider (default: 10)
        overshoot (float): Overshoot parameter for boundary crossing (default: 0.02)
        max_iter (int): Maximum iterations before terminating (default: 50)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - r_tot (torch.Tensor): Total accumulated perturbation
            - loop_i (int): Number of iterations performed
            - label (int): Original predicted class
            - k_i (int): Final adversarial class
            - pert_image (torch.Tensor): Final perturbed image
    """
    image = image.to(device)
    net = net.to(device)
def deepfool(image: torch.Tensor,
             net: nn.Module,
             num_classes: int = 10,
             overshoot: float = 0.02,
             max_iter: int = 50,
             device: str = 'cuda') -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
    """
    Generate minimal adversarial perturbation using DeepFool algorithm.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        net (nn.Module): Target neural network in evaluation mode
        num_classes (int): Number of top-scoring classes to consider (default: 10)
        overshoot (float): Overshoot parameter for boundary crossing (default: 0.02)
        max_iter (int): Maximum iterations before terminating (default: 50)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - r_tot (torch.Tensor): Total accumulated perturbation
            - loop_i (int): Number of iterations performed
            - label (int): Original predicted class
            - k_i (int): Final adversarial class
            - pert_image (torch.Tensor): Final perturbed image
    """
    image = image.to(device)
    net = net.to(device)

    # Original prediction and class ordering (descending score)
    f_image = net(image).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    label = I[0]

    # Working tensors and accumulators
    input_shape = image.shape
    pert_image = image.clone()
    r_tot = torch.zeros(input_shape).to(device)
    loop_i = 0

    # Iterate until a successful perturbation is found or the limit is reached
    while loop_i < max_iter:
        x = pert_image.clone().requires_grad_(True)
        fs = net(x)
        # Current top prediction at x
        k_i = fs.data.cpu().numpy().flatten().argsort()[::-1][0]

        # Stop when the prediction changes
        if k_i != label:
            break

        # Initialize the best candidate step for this iteration
        pert = float('inf')
        w = None
        # Search minimal step among candidate classes
        for k in range(1, num_classes):
            if I[k] == label:
                continue
            # Compute gradient for candidate class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, I[k]].backward(retain_graph=True)
            grad_k = x.grad.data.clone()
            # Compute gradient for original class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, label].backward(retain_graph=True)
            grad_label = x.grad.data.clone()
            # Direction and distance under linearization
            w_k = grad_k - grad_label
            f_k = (fs[0, I[k]] - fs[0, label]).data.cpu().numpy()
            pert_k = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-10)
            if pert_k < pert:
                pert = pert_k
                w = w_k
        # Minimal step for the selected direction
        r_i = (pert + 1e-4) * w / (torch.norm(w.flatten()) + 1e-10)
        r_tot = r_tot + r_i

        # Apply with overshoot to ensure crossing
        pert_image = image + (1 + overshoot) * r_tot
        loop_i += 1

    return r_tot, loop_i, label, k_i, pert_image

# Load trained model
model_path = 'output/mnist_model.pth'
if os.path.exists(model_path):
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()
else:
    raise FileNotFoundError("Model not found.")

# Get single test sample
_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
dataiter = iter(test_loader)
image, true_label = next(dataiter)
image = image.to(device)

print(f"True label: {true_label.item()}")

# Baseline classification
with torch.no_grad():
    original_output = model(image)
    original_pred = original_output.argmax(dim=1).item()
    original_confidence = F.softmax(original_output, dim=1).max().item()

print(f"Original: class {original_pred} (confidence: {original_confidence:.3f})")

# Execute DeepFool attack
r_total, iterations, orig_label, pert_label, pert_image = deepfool(
    image, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
)

print(f"Attack: {orig_label} → {pert_label} in {iterations} iterations")

# Compute perturbation norms
perturbation_norm_l2 = torch.norm(r_total).item()
perturbation_norm_linf = torch.abs(r_total).max().item()
relative_perturbation = perturbation_norm_l2 / torch.norm(image).item()

# Evaluate adversarial confidence
with torch.no_grad():
    adv_output = model(pert_image)
    adv_confidence = F.softmax(adv_output, dim=1).max().item()

# Display results
print(f"\n=== Attack Results ===")
print(f"L2 norm: {perturbation_norm_l2:.4f}")
print(f"L∞ norm: {perturbation_norm_linf:.4f}")
print(f"Relative perturbation: {relative_perturbation:.2%}")
print(f"Original confidence: {original_confidence:.3f}")
print(f"Adversarial confidence: {adv_confidence:.3f}")

# Prepare images for visualization
original_img = mnist_denormalize(image.squeeze()).cpu().numpy()
adversarial_img = mnist_denormalize(pert_image.squeeze()).cpu().numpy()
perturbation = r_total.cpu().squeeze().numpy()

# Normalize perturbation for visibility (amplify minimal changes)
pert_display = perturbation - perturbation.min()
if pert_display.max() > 0:
    pert_display = pert_display / pert_display.max()

# Create four-panel visualization
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)

# Panel 1: Original clean image
axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f"Original\nClass: {original_pred}",
                  color=HTB_GREEN, fontweight='bold')
axes[0].axis('off')

# Panel 2: Amplified perturbation pattern
axes[1].imshow(pert_display, cmap='inferno')
axes[1].set_title("Perturbation\n(amplified)",
                  color=NUGGET_YELLOW, fontweight='bold')
axes[1].axis('off')

# Panel 3: Perturbation magnitude heatmap
im = axes[2].imshow(np.abs(perturbation), cmap='viridis')
axes[2].set_title(f"Magnitude\nL2: {perturbation_norm_l2:.4f}",
                  color=AZURE, fontweight='bold')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Panel 4: Adversarial result
title_color = HTB_GREEN if pert_label != original_pred else MALWARE_RED
axes[3].imshow(adversarial_img, cmap='gray', vmin=0, vmax=1)
axes[3].set_title(f"Adversarial\nClass: {pert_label}",
                  color=title_color, fontweight='bold')
axes[3].axis('off')

# Summary metrics
metrics_text = (
    f"Iterations: {iterations}  |  "
    f"Relative pert: {relative_perturbation:.2%}  |  "
    f"Confidence: {original_confidence:.3f} → {adv_confidence:.3f}"
)
fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, color=WHITE)

plt.suptitle("DeepFool Attack Visualization", fontsize=14,
             color=HTB_GREEN, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

num_examples = 20
print(f"\nGenerating {num_examples} adversarial examples using DeepFool...")

_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
model.eval()

results = []
success_count = 0

print(f"Test loader ready with {len(test_loader.dataset)} samples")
print(f"Will process first {num_examples} samples")
print("Starting batch attack generation...")

for idx, (data, target) in enumerate(test_loader):
    if idx >= num_examples:
        break

    data = data.to(device)

    # Execute DeepFool attack
    r, iterations, orig_label, adv_label, pert_image = deepfool(
        data, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
    )

    # Track success and store metrics
    success = (orig_label != adv_label)
    if success:
        success_count += 1

    results.append({
        'original_image': data.cpu(),
        'perturbation': r.cpu(),
        'perturbed_image': pert_image.cpu(),
        'original_label': orig_label,
        'adversarial_label': adv_label,
        'iterations': iterations,
        'true_label': target.item(),
        'l2_norm': torch.norm(r.cpu()).item(),
        'success': success
    })

    # Progress feedback
    print(f"  Example {idx+1}: True={target.item()}, Orig={orig_label}, "
          f"Adv={adv_label}, Iter={iterations}, L2={torch.norm(r.cpu()).item():.4f}")

print(f"\nAttack Success Rate: {success_count}/{num_examples} "
      f"({100*success_count/num_examples:.1f}%)")
print(f"Average L2 norm: {np.mean([r['l2_norm'] for r in results]):.4f}")
print(f"Average iterations: {np.mean([r['iterations'] for r in results]):.1f}")

def visualize_attack_grid(results, save_dir='output'):
    """
    Create grid visualization showing original and adversarial images side-by-side.

    Args:
        results (list): Attack results from batch generation
        save_dir (str): Directory to save visualization
    """
    print("\nGenerating attack grid visualization...")

    num_examples = min(10, len(results))
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.patch.set_facecolor(NODE_BLACK)

    for ax in axes.flatten():
        ax.set_facecolor(NODE_BLACK)
        for spine in ax.spines.values():
            spine.set_edgecolor(HACKER_GREY)

    for idx in range(num_examples):
        row = idx // 5
        col = idx % 5

        # Original image (top row for this column)
        ax_original = axes[row * 2, col]
        img = mnist_denormalize(results[idx]['original_image'].squeeze()).numpy()
        ax_original.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax_original.set_title(f"Original: {results[idx]['original_label']}",
                            color=HTB_GREEN, fontsize=10)
        ax_original.axis('off')

        # Adversarial image (bottom row for this column)
        ax_adv = axes[row * 2 + 1, col]
        adv_img = mnist_denormalize(results[idx]['perturbed_image'].squeeze()).numpy()
        ax_adv.imshow(adv_img, cmap='gray', vmin=0, vmax=1)

        title_color = MALWARE_RED if results[idx]['success'] else HACKER_GREY
        ax_adv.set_title(f"Adversarial: {results[idx]['adversarial_label']}",
                        color=title_color, fontsize=10)
        ax_adv.axis('off')

    plt.suptitle('DeepFool Attack: Original vs Adversarial Examples',
                color=HTB_GREEN, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepfool_examples.png'),
                facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Grid visualization saved to {save_dir}/deepfool_examples.png")

# Generate the grid visualization
visualize_attack_grid(results, save_dir='output')

def visualize_perturbation_analysis(results, save_dir='output'):
    """
    Analyze and visualize perturbation characteristics across samples.

    Creates two-row visualization: top shows raw perturbation heatmaps,
    bottom shows amplified differences overlaid on originals.

    Args:
        results (list): Attack results
        save_dir (str): Output directory
    """
    print("\nGenerating perturbation analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor(NODE_BLACK)

    for ax in axes.flatten():
        ax.set_facecolor(NODE_BLACK)
        for spine in ax.spines.values():
            spine.set_edgecolor(HACKER_GREY)

    # Select first 3 successful attacks
    successful_attacks = [r for r in results if r['success']][:3]

    for idx, result in enumerate(successful_attacks):
        # Top row: Raw perturbation heatmap
        ax_top = axes[0, idx]
        pert = result['perturbation'].squeeze().numpy()
        vmax = np.abs(pert).max() or 1e-6
        im_top = ax_top.imshow(pert, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_top.set_title(f'Perturbation (L2={result["l2_norm"]:.3f})',
                        color=HTB_GREEN, fontsize=10)
        ax_top.axis('off')

        cbar_top = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.04)
        cbar_top.outline.set_edgecolor(HACKER_GREY)
        cbar_top.ax.tick_params(colors=WHITE)

        # Bottom row: Amplified difference visualization
        ax_bottom = axes[1, idx]
        orig_img = result['original_image'].squeeze().numpy()
        adv_img = result['perturbed_image'].squeeze().detach().numpy()
        diff_amplified = (adv_img - orig_img) * 10  # 10x amplification for visibility

        im_bottom = ax_bottom.imshow(diff_amplified, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax_bottom.set_title(f"{result['original_label']} → {result['adversarial_label']} "
                           f"({result['iterations']} iters)",
                           color=NUGGET_YELLOW, fontsize=10)
        ax_bottom.axis('off')

        cbar_bottom = plt.colorbar(im_bottom, ax=ax_bottom, fraction=0.046, pad=0.04)
        cbar_bottom.outline.set_edgecolor(HACKER_GREY)
        cbar_bottom.ax.tick_params(colors=WHITE)

    plt.suptitle('DeepFool Perturbation Analysis', color=HTB_GREEN, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deepfool_perturbations.png'),
                facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Perturbation analysis saved to {save_dir}/deepfool_perturbations.png")

# Generate the perturbation analysis visualization
visualize_perturbation_analysis(results, save_dir='output')

print("\nGenerating attack metrics visualization...")

# Setup three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)
    ax.tick_params(colors=WHITE)
    ax.grid(True, alpha=0.3, color=HACKER_GREY, linestyle='--')

# Panel 1: L2 Norm Distribution
l2_norms = [r['l2_norm'] for r in results]
axes[0].hist(l2_norms, bins=15, color=HTB_GREEN, alpha=0.7, edgecolor=HACKER_GREY)
axes[0].set_xlabel('L2 Norm', color=WHITE)
axes[0].set_ylabel('Frequency', color=WHITE)
axes[0].set_title('Perturbation Magnitude Distribution', color=HTB_GREEN)
print(f"L2 norm range: [{min(l2_norms):.4f}, {max(l2_norms):.4f}]")

# Panel 2: Iteration Count Distribution
iterations = [r['iterations'] for r in results]
axes[1].hist(iterations, bins=range(1, max(iterations)+2),
            color=AZURE, alpha=0.7, edgecolor=HACKER_GREY)
axes[1].set_xlabel('Iterations', color=WHITE)
axes[1].set_ylabel('Frequency', color=WHITE)
axes[1].set_title('Iterations Required', color=HTB_GREEN)
print(f"Iteration range: [{min(iterations)}, {max(iterations)}]")

# Panel 3: Per-Class Success Rates
class_success = {}
for r in results:
    orig = r['original_label']
    if orig not in class_success:
        class_success[orig] = {'total': 0, 'success': 0}
    class_success[orig]['total'] += 1
    if r['success']:
        class_success[orig]['success'] += 1

classes = sorted(class_success.keys())
success_rates = [
    class_success[c]['success'] / class_success[c]['total'] * 100
    if class_success[c]['total'] > 0 else 0
    for c in classes
]

bars = axes[2].bar(classes, success_rates, color=NUGGET_YELLOW,
                   alpha=0.7, edgecolor=HACKER_GREY)
axes[2].set_xlabel('Original Class', color=WHITE)
axes[2].set_ylabel('Success Rate (%)', color=WHITE)
axes[2].set_title('Attack Success by Class', color=HTB_GREEN)
axes[2].set_ylim(0, 105)

# Add percentage labels on bars
for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax_x = bar.get_x() + bar.get_width() / 2.0
    axes[2].text(ax_x, height + 1, f'{rate:.0f}%',
                 ha='center', va='bottom', color=WHITE, fontsize=8)

# Save visualization
plt.suptitle('DeepFool Attack Metrics', color=HTB_GREEN, fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('output/deepfool_metrics.png',
            facecolor=NODE_BLACK, dpi=150, bbox_inches='tight')
plt.close()

print("Metrics visualization saved to output/deepfool_metrics.png")

def print_summary_statistics(results):
    """
    Print summary statistics for attack results.

    Computes and displays success rate, perturbation statistics, iteration
    statistics, and common class transitions.

    Args:
        results (list): Attack results
    """
    print("\n" + "="*60)
    print("Attack Summary Statistics")
    print("="*60)

    successful_attacks = [r for r in results if r['success']]

    if successful_attacks:
        avg_l2 = np.mean([r['l2_norm'] for r in successful_attacks])
        avg_iterations = np.mean([r['iterations'] for r in successful_attacks])
        min_l2 = min([r['l2_norm'] for r in successful_attacks])
        max_l2 = max([r['l2_norm'] for r in successful_attacks])

        print(f"Success Rate: {len(successful_attacks)}/{len(results)} "
              f"({100*len(successful_attacks)/len(results):.1f}%)")
        print(f"Average L2 Norm: {avg_l2:.4f}")
        print(f"L2 Range: [{min_l2:.4f}, {max_l2:.4f}]")
        print(f"Average Iterations: {avg_iterations:.1f}")

        # Class transition analysis
        transitions = {}
        for r in successful_attacks:
            key = f"{r['original_label']}→{r['adversarial_label']}"
            transitions[key] = transitions.get(key, 0) + 1

        print(f"\nMost Common Misclassifications:")
        for trans, count in sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {trans}: {count} times")
    else:
        print("No successful attacks generated")

    print("="*60)

# Generate summary
print_summary_statistics(results)
