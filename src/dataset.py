import os
import sys
import cv2
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_sample_lesion_image(class_code, size=(224, 224)):
    """
    Generates a realistic synthetic skin lesion image for a given class code.
    Useful for system demonstration, pipeline verification, and standalone testing.
    """
    width, height = size
    # Base skin tone generator (Fitzpatrick scale sample tones: light to deep brown)
    skin_tones = [
        (235, 210, 195), # Light
        (220, 185, 160), # Medium Light
        (190, 150, 120), # Medium
        (140, 95, 65),   # Dark Medium
        (85, 55, 35)     # Deep Brown
    ]
    base_skin = random.choice(skin_tones)
    
    # Create canvas with subtle Gaussian noise (skin pore texture)
    image = np.ones((height, width, 3), dtype=np.uint8)
    image[:, :] = base_skin
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random center position and lesion dimensions
    cx = random.randint(int(width * 0.35), int(width * 0.65))
    cy = random.randint(int(height * 0.35), int(height * 0.65))
    rx = random.randint(25, 55)
    ry = random.randint(25, 55)

    # Render characteristic lesion patterns per class
    if class_code == "AKIEC":
        # Scaly red/pink scabbing with irregular border
        color = (random.randint(180, 230), random.randint(80, 130), random.randint(100, 150))
        cv2.ellipse(image, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, color, -1)
        # Scaly texture overlay
        scale_noise = np.random.randint(0, 60, (height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx + 5, ry + 5), 0, 0, 360, 255, -1)
        image[mask > 0] = cv2.addWeighted(image[mask > 0], 0.7, scale_noise[mask > 0], 0.3, 0)

    elif class_code == "BCC":
        # Pearly translucent pink bump with central depression
        pink_color = (random.randint(200, 240), random.randint(140, 180), random.randint(160, 200))
        cv2.circle(image, (cx, cy), rx, pink_color, -1)
        cv2.circle(image, (cx, cy), int(rx * 0.5), (210, 120, 140), -1)

    elif class_code == "BKL":
        # Waxy brown/tan stuck-on papule with crisp border
        brown_color = (random.randint(60, 110), random.randint(40, 80), random.randint(20, 50))
        cv2.ellipse(image, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, brown_color, -1)

    elif class_code == "DF":
        # Firm reddish-brown nodule with paler center
        red_brown = (random.randint(140, 180), random.randint(70, 100), random.randint(60, 90))
        cv2.circle(image, (cx, cy), rx, red_brown, -1)
        cv2.circle(image, (cx, cy), int(rx * 0.4), (200, 170, 150), -1)

    elif class_code == "MEL":
        # Asymmetric, multi-colored dark brown/black spot with jagged border
        black_brown = (random.randint(20, 60), random.randint(15, 45), random.randint(10, 35))
        pts = np.array([
            [cx - rx, cy - ry + random.randint(-10, 10)],
            [cx + rx + random.randint(-10, 10), cy - ry],
            [cx + rx + 10, cy + ry],
            [cx - rx + random.randint(-15, 5), cy + ry + 15]
        ], np.int32)
        cv2.fillPoly(image, [pts], black_brown)
        cv2.circle(image, (cx + random.randint(-10, 10), cy), int(rx * 0.4), (120, 40, 40), -1)

    elif class_code == "NV":
        # Smooth, uniform brown mole
        mole_color = (random.randint(80, 120), random.randint(50, 80), random.randint(30, 60))
        cv2.circle(image, (cx, cy), rx, mole_color, -1)

    elif class_code == "VASC":
        # Deep red / cherry vascular papule
        cherry_red = (random.randint(180, 240), random.randint(20, 50), random.randint(40, 70))
        cv2.circle(image, (cx, cy), rx, cherry_red, -1)

    elif class_code == "SCC":
        # Raised crusted red lesion with yellow-brown scaling
        scc_red = (random.randint(160, 210), random.randint(50, 90), random.randint(60, 100))
        cv2.ellipse(image, (cx, cy), (rx, ry), random.randint(0, 90), 0, 360, scc_red, -1)
        cv2.circle(image, (cx + 5, cy - 5), int(rx * 0.3), (220, 190, 120), -1)

    # Blur border for realistic transition into skin
    image = cv2.GaussianBlur(image, (5, 5), 1)
    # Convert BGR to RGB for standard image saving
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def generate_sample_dataset(samples_per_class=40, base_dir=config.DATASET_DIR):
    """
    Generates a sample dataset with train, val, and test split subdirectories
    for all 8 skin condition classes.
    """
    classes = sorted(list(config.SKIN_CONDITIONS.keys()))
    splits = {
        "train": int(samples_per_class * 0.70),
        "val": int(samples_per_class * 0.15),
        "test": int(samples_per_class * 0.15)
    }

    print("Generating synthetic sample dataset across 8 classes...")
    total_generated = 0
    for split, count in splits.items():
        for class_code in classes:
            class_dir = os.path.join(base_dir, split, class_code)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(count):
                img_array = generate_sample_lesion_image(class_code)
                filename = f"{class_code}_{split}_{i+1:03d}.jpg"
                filepath = os.path.join(class_dir, filename)
                # Convert back to BGR for cv2.imwrite
                cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                total_generated += 1

    print(f"Sample dataset generation complete! Generated {total_generated} total images in '{base_dir}'.")
    return base_dir
