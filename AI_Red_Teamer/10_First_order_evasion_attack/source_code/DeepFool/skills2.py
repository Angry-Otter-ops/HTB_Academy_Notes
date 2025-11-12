#!/usr/bin/env python3

import argparse
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import sys
import os

def base64_to_tensor(base64_str: str) -> torch.Tensor:
    """
    Convert base64-encoded PNG to tensor.

    Args:
        base64_str: Base64-encoded PNG string

    Returns:
        torch.Tensor: Image tensor in [0,1] range with shape (C, H, W)
    """
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    tensor = transforms.ToTensor()(img)
    return tensor


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """
    Convert tensor to base64-encoded PNG.

    Args:
        tensor: Image tensor in [0,1] range with shape (C, H, W)

    Returns:
        str: Base64-encoded PNG string
    """
    img_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

class CIFAR10CNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification.

    Architecture:
        - Conv block 1: 3→32 channels, BatchNorm, ReLU, MaxPool
        - Conv block 2: 32→64 channels, BatchNorm, ReLU, MaxPool
        - FC1: 64*8*8 → 128, ReLU, Dropout(0.5)
        - FC2: 128 → 10 (logits)
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize the CNN.

        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
        """
        super(CIFAR10CNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def load_model(model_path: str, device: str = 'cuda') -> CIFAR10CNN:
    """
    Load trained CIFAR-10 model.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        CIFAR10CNN: Loaded model in eval mode
    """
    model = CIFAR10CNN(num_classes=10)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both direct state_dict and checkpoint dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model

def deepfool_attack(
    model: nn.Module,
    image: torch.Tensor,
    mean: list,
    std: list,
    num_classes: int = 10,
    overshoot: float = 0.02,
    max_iter: int = 50,
    device: str = 'cuda'
) -> tuple:
    """
    DeepFool minimal perturbation attack.

    This implements the complete DeepFool algorithm:
    1. Linearize the classifier around the current point
    2. For each class k != current_class:
       - Compute gradient direction w_k that increases class k score
       - Compute distance to the linearized decision boundary
    3. Find the closest decision boundary
    4. Take minimal step to reach that boundary (with overshoot)
    5. Re-linearize and repeat until misclassification

    Args:
        model: Trained CIFAR-10 model
        image: Clean image in [0,1] range, shape (3, 32, 32)
        mean: Normalization mean per channel
        std: Normalization std per channel
        num_classes: Number of classes to consider
        overshoot: Overshoot parameter to ensure crossing boundary
        max_iter: Maximum iterations
        device: Device to run attack on

    Returns:
        tuple: (adversarial_image, total_perturbation, iterations, final_class)
    """
    # Prepare normalization tensors
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t = torch.tensor(std, device=device).view(3, 1, 1)

    # Work in pixel space [0,1]
    x = image.clone().to(device)
    x_orig = image.clone().to(device)

    # Normalize for model
    x_norm = (x - mean_t) / std_t

    # Get initial prediction
    with torch.no_grad():
        logits = model(x_norm.unsqueeze(0))
        current_class = logits.argmax(dim=1).item()

    original_class = current_class
    r_total = torch.zeros_like(x)

    print(f"\n{'='*60}")
    print(f"DeepFool Attack")
    print(f"{'='*60}")
    print(f"Original class: {original_class}")
    print(f"Num classes: {num_classes}")
    print(f"Overshoot: {overshoot}")
    print(f"Max iterations: {max_iter}")
    print(f"{'='*60}\n")

    for iteration in range(max_iter):
        # Enable gradients
        x_norm = (x - mean_t) / std_t
        x_norm.requires_grad = True

        # Forward pass
        logits = model(x_norm.unsqueeze(0))
        current_class = logits.argmax(dim=1).item()

        # Check if misclassified
        if current_class != original_class:
            print(f"  Misclassification achieved at iteration {iteration + 1}")
            print(f"  New class: {current_class}")
            break

        # Find closest decision boundary
        min_dist = float('inf')
        best_w = None
        best_f = None

        for k in range(num_classes):
            if k == current_class:
                continue

            # Zero gradients
            if x_norm.grad is not None:
                x_norm.grad.zero_()

            # Gradient for class k
            logits[0, k].backward(retain_graph=True)
            grad_k = x_norm.grad.clone()

            # Zero gradients
            if x_norm.grad is not None:
                x_norm.grad.zero_()

            # Gradient for current class
            logits[0, current_class].backward(retain_graph=True)
            grad_current = x_norm.grad.clone()

            # Direction: increases k, decreases current
            w_k = grad_k - grad_current

            # Score gap
            f_k = (logits[0, k] - logits[0, current_class]).item()

            # Distance to boundary in normalized space
            w_norm = torch.norm(w_k)
            dist = abs(f_k) / (w_norm + 1e-10)

            # Track minimum
            if dist < min_dist:
                min_dist = dist
                best_w = w_k
                best_f = f_k

        # Compute perturbation in normalized space
        w_norm_sq = torch.norm(best_w)**2
        r_i = (abs(best_f) / (w_norm_sq + 1e-10)) * best_w

        # Convert to pixel space (chain rule)
        r_i_pixel = r_i * std_t

        # Apply perturbation with overshoot
        r_total = r_total + (1 + overshoot) * r_i_pixel
        x = x_orig + r_total

        # Clip to valid range
        x = torch.clamp(x, 0.0, 1.0)

        if (iteration + 1) % 10 == 0:
            # Check L2 in normalized space
            with torch.no_grad():
                x_norm_check = (x - mean_t) / std_t
                orig_norm = (x_orig - mean_t) / std_t
                l2 = torch.norm(x_norm_check - orig_norm).item()
            print(f"  Iteration {iteration + 1}/{max_iter} - L2 norm: {l2:.4f}")

    print(f"\n{'='*60}")
    print(f"Attack complete!")
    print(f"{'='*60}\n")

    return x.detach().cpu(), r_total.detach().cpu(), iteration + 1, current_class
def test_api_robustness(host: str, image: torch.Tensor) -> bool:
    """
    Test API robustness against invalid inputs.

    Args:
        host: Challenge server host URL
        image: Clean challenge image

    Returns:
        bool: True if all robustness tests pass
    """
    print(f"\n{'='*60}")
    print("API Robustness Tests")
    print(f"{'='*60}\n")

    all_passed = True

    # Test 1: Invalid JSON
    print("[Test 1] Invalid JSON format...")
    try:
        response = requests.post(
            f"{host}/submit",
            data="invalid json",
            headers={'content-type': 'application/json'}
        )
        if response.status_code == 400:
            print("  [+] Correctly rejected invalid JSON")
        else:
            print(f"  [-] Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    # Test 2: Missing image field
    print("\n[Test 2] Missing 'image' field...")
    try:
        response = requests.post(
            f"{host}/submit",
            json={"wrong_field": "data"}
        )
        if response.status_code == 400:
            result = response.json()
            if 'error' in result and 'image' in result['error'].lower():
                print("  [+] Correctly rejected missing image field")
            else:
                print(f"  [-] Expected error message about missing image field")
                all_passed = False
        else:
            print(f"  [-] Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    # Test 3: Invalid base64
    print("\n[Test 3] Invalid base64 encoding...")
    try:
        response = requests.post(
            f"{host}/submit",
            json={"image": "not_valid_base64!!!"}
        )
        if response.status_code == 400:
            print("  [+] Correctly rejected invalid base64")
        else:
            print(f"  [-] Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    # Test 4: Corrupted image data
    print("\n[Test 4] Corrupted image data...")
    try:
        # Send corrupted base64 that decodes but isn't a valid PNG
        corrupted_base64 = base64.b64encode(b"not a valid png image").decode('utf-8')
        response = requests.post(
            f"{host}/submit",
            json={"image": corrupted_base64}
        )
        if response.status_code == 400:
            print("  [+] Correctly rejected corrupted image data")
        else:
            print(f"  [-] Expected 400, got {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    print(f"\n{'='*60}")
    print(f"API Robustness: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}\n")

    return all_passed

def test_anti_cheat(host: str, model: nn.Module, image: torch.Tensor,
                    l2_threshold: float, mean: list, std: list,
                    device: str) -> bool:
    """
    Test anti-cheat mechanisms for DeepFool.

    Args:
        host: Challenge server host URL
        model: Trained model
        image: Clean challenge image
        l2_threshold: L2 constraint threshold
        mean: Normalization mean
        std: Normalization std
        device: Device to run on

    Returns:
        bool: True if all anti-cheat tests pass
    """
    print(f"\n{'='*60}")
    print("Anti-Cheat Mechanism Tests")
    print(f"{'='*60}\n")

    all_passed = True
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    # Test 1: Clean image (no modification)
    print("[Test 1] Submitting clean image (should fail)...")
    try:
        clean_base64 = tensor_to_base64(image)
        response = requests.post(f"{host}/submit", json={"image": clean_base64})
        result = response.json()
        if not result['success'] and not result['validation']['misclassification']:
            print("  [+] Correctly rejected clean image")
        else:
            print("  [-] Should reject clean image (no misclassification)")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    # Test 2: L2 perturbation too large
    print("\n[Test 2] L2 perturbation exceeding threshold (should fail)...")
    try:
        large_pert = image.clone()
        # Add large perturbation
        large_pert = torch.clamp(large_pert + 0.5, 0, 1)
        large_base64 = tensor_to_base64(large_pert)
        response = requests.post(f"{host}/submit", json={"image": large_base64})
        result = response.json()
        if not result['success'] and not result['validation']['l2_satisfied']:
            print("  [+] Correctly rejected oversized L2 perturbation")
        else:
            print("  [-] Should reject perturbation exceeding L2 threshold")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    # Test 3: I-FGSM attack (L∞-based, should fail L∞ check)
    print("\n[Test 3] I-FGSM attack (L∞-based, should fail)...")
    try:
        x_fgsm = image.clone().to(device)
        x_norm = (x_fgsm - mean_t.to(device)) / std_t.to(device)
        x_norm.requires_grad = True

        outputs = model(x_norm.unsqueeze(0))
        target = torch.tensor([0], device=device)  # Arbitrary target

        # Targeted I-FGSM
        for _ in range(20):
            loss = F.cross_entropy(outputs, target)
            model.zero_grad()
            loss.backward()

            grad_norm = x_norm.grad
            grad_pixel = grad_norm / std_t.to(device)

            x_fgsm = x_fgsm - 0.01 * grad_pixel.sign()
            x_fgsm = torch.clamp(x_fgsm, 0, 1)

            x_norm = (x_fgsm - mean_t.to(device)) / std_t.to(device)
            x_norm.requires_grad = True
            outputs = model(x_norm.unsqueeze(0))

        x_fgsm = x_fgsm.cpu()
        fgsm_base64 = tensor_to_base64(x_fgsm)
        response = requests.post(f"{host}/submit", json={"image": fgsm_base64})
        result = response.json()

        if not result['success'] and not result['validation'].get('reasonable_linf', True):
            print("  [+] Correctly rejected I-FGSM attack (L∞ too large)")
        else:
            print("  [-] Should reject I-FGSM attacks")
            all_passed = False
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    # Test 4: Random perturbation (wrong direction)
    print("\n[Test 4] Random perturbation (wrong direction, should fail)...")
    try:
        random_pert = image.clone()
        # Small random perturbation that causes misclassification
        noise = torch.randn_like(random_pert) * 0.03
        random_pert = torch.clamp(random_pert + noise, 0, 1)

        # Check if it causes misclassification
        with torch.no_grad():
            orig_norm = (image - mean_t) / std_t
            rand_norm = (random_pert - mean_t) / std_t
            orig_pred = model(orig_norm.unsqueeze(0).to(device)).argmax().item()
            rand_pred = model(rand_norm.unsqueeze(0).to(device)).argmax().item()

        if rand_pred != orig_pred:
            random_base64 = tensor_to_base64(random_pert)
            response = requests.post(f"{host}/submit", json={"image": random_base64})
            result = response.json()

            if not result['success'] and not result['validation'].get('deepfool_direction', True):
                print("  [+] Correctly rejected random perturbation")
            else:
                print("  [-] Should reject random perturbations")
                all_passed = False
        else:
            print("  [!] Random perturbation didn't cause misclassification, skipping")
    except Exception as e:
        print(f"  [-] Exception: {e}")
        all_passed = False

    print(f"\n{'='*60}")
    print(f"Anti-Cheat Tests: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}\n")

    return all_passed

def solve_challenge(host: str, device: str = 'cuda', run_tests: bool = False):
    """
    Main solver function.

    Args:
        host: Challenge server host URL
        device: Device for attack ('cuda' or 'cpu')
        run_tests: Whether to run robustness and anti-cheat tests
    """
    print(f"\n{'='*60}")
    print(f"Skills Assessment 2")
    print(f"{'='*60}\n")

    # Determine model path (assume running from solver/ directory)
    solver_dir = os.path.dirname(os.path.abspath(__file__))
    challenge_dir = os.path.dirname(solver_dir)
    assets_dir = os.path.join(challenge_dir, "assets")
    model_path = os.path.join(assets_dir, "cifar10_model_best.pth")

    # Ensure assets directory exists
    os.makedirs(assets_dir, exist_ok=True)

    # Fetch model if missing
    if not os.path.exists(model_path):
        print(f"[1/5] Model not found at {model_path}")
        weights_url = f"{host}/model/weights"  # Adjust if your endpoint differs
        print(f"Attempting to download weights from: {weights_url}")

        try:
            resp = requests.get(weights_url, timeout=30)
            resp.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(resp.content)
            print(f"  Weights downloaded and saved to {model_path}")
        except requests.exceptions.RequestException as e:
            print(f"[-] Error: Failed to download model weights from {weights_url}")
            print(f"Details: {e}")
            print("Hint: Ensure the challenge server exposes /model/weights and is reachable.")
            raise

    # Step 1: Load model
    print("[1/5] Loading CIFAR-10 model...")
    model = load_model(model_path, device=device)
    print(f"  Model loaded on {device}")

    # Step 2: Get challenge
    print("\n[2/5] Fetching challenge...")
    response = requests.get(f"{host}/challenge")
    response.raise_for_status()
    challenge = response.json()

    print(f"  Original class: {challenge['original_class']} ({challenge['original_class_name']})")
    print(f"  L2 threshold: {challenge['l2_threshold']}")
    print(f"  Num classes hint: {challenge['num_classes_hint']}")
    print(f"  Overshoot hint: {challenge['overshoot_hint']}")

    # Step 3: Decode image
    print("\n[3/5] Decoding challenge image...")
    image = base64_to_tensor(challenge['image'])
    mean = challenge['normalization']['mean']
    std = challenge['normalization']['std']
    print(f"  Image shape: {image.shape}")
    print(f"  Normalization: mean={mean}, std={std}")

    # Verify original prediction
    with torch.no_grad():
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        img_norm = (image - mean_t) / std_t
        orig_pred = model(img_norm.unsqueeze(0).to(device)).argmax(dim=1).item()
    print(f"  Verified original prediction: {orig_pred}")

    # Optional: Run robustness and anti-cheat tests
    if run_tests:
        print("\n" + "="*60)
        print("RUNNING SECURITY TESTS")
        print("="*60)

        robustness_passed = test_api_robustness(host, image)
        anticheat_passed = test_anti_cheat(
            host, model, image, challenge['l2_threshold'],
            mean, std, device
        )

        if robustness_passed and anticheat_passed:
            print(f"\n{'='*60}")
            print("[+] ALL SECURITY TESTS PASSED")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("[-] SOME SECURITY TESTS FAILED")
            print(f"{'='*60}\n")

        print("Proceeding with legitimate attack...\n")

    # Step 4: Run DeepFool attack
    print("\n[4/5] Running DeepFool attack...")
    adv_image, perturbation, iters, final_class = deepfool_attack(
        model=model,
        image=image,
        mean=mean,
        std=std,
        num_classes=challenge['num_classes_hint'],
        overshoot=challenge['overshoot_hint'],
        max_iter=challenge['max_iterations_hint'],
        device=device
    )

    print(f"  Iterations: {iters}")
    print(f"  Final class: {final_class}")

    # Verify L2 norm in normalized space
    with torch.no_grad():
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        orig_norm = (image - mean_t) / std_t
        adv_norm = (adv_image - mean_t) / std_t
        delta_norm = adv_norm - orig_norm
        l2_norm = torch.norm(delta_norm).item()
    print(f"  L2 norm (normalized): {l2_norm:.4f}")
    print(f"  Threshold: {challenge['l2_threshold']}")
    print(f"  Within threshold: {l2_norm <= challenge['l2_threshold']}")

    # Step 5: Submit
    print("\n[5/5] Submitting adversarial example...")
    adv_base64 = tensor_to_base64(adv_image)

    response = requests.post(
        f"{host}/submit",
        json={"image": adv_base64}
    )
    response.raise_for_status()
    result = response.json()

    # Display results
    print(f"\n{'='*60}")
    print("Validation Results")
    print(f"{'='*60}")

    validation = result['validation']
    print(f"L2 norm: {validation['l2_norm']:.4f} / {validation['l2_threshold']}")
    print(f"L2 constraint satisfied: {validation['l2_satisfied']}")
    print(f"Valid range [0,1]: {validation['valid_range']}")
    print(f"Original prediction: {validation['original_class']}")
    print(f"Adversarial prediction: {validation['adversarial_class']}")
    print(f"Misclassification: {validation['misclassification']}")
    print(f"\nSuccess: {result['success']}")

    if result['success']:
        print(f"\n{'='*60}")
        print(f"[+] FLAG: {result['flag']}")
        print(f"{'='*60}\n")
    else:
        print(f"\n[-] {result['message']}")
        if 'hint' in result:
            print(f"[!] {result['hint']}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Working solver for Skills Assessment 2'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='http://localhost:8001',
        help='Challenge server host URL (default: http://localhost:8001)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run attack on (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip API robustness and anti-cheat tests (tests run by default)'
    )
    args = parser.parse_args()

    try:
        solve_challenge(args.host, args.device, not args.skip_tests)
    except requests.exceptions.ConnectionError:
        print(f"\n[-] Error: Could not connect to {args.host}")
        print("Make sure the challenge server is running (./run.sh)\n")
    except Exception as e:
        print(f"\n[-] Error: {str(e)}\n")
        raise


if __name__ == "__main__":
    main()
