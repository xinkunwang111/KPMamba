
# KPMamba: A Token-Level Semantic Alignment Framework for Category-Agnostic Pose Estimation

## Overview

KPMamba is a lightweight, category-agnostic pose estimation framework built upon the Vision Mamba architecture. By introducing token-level semantic alignment, KPMamba delivers state-of-the-art performance for pose estimation tasks, even in scenarios where category-specific training data is unavailable. This framework leverages a novel Mamba-based visual backbone, external pretrained encoders, and advanced multi-attention feature fusion mechanisms to enhance performance, improve convergence, and boost few-shot learning capabilities.

---

## Key Features

### 1. Lightweight Mamba-Based Visual Backbone
- **Enhanced Feature Extraction**: Replaces the initial ESCAPE model backbone with a more precise and contextualized visual backbone.
- **Improved Performance**: Significantly boosts the pose estimation accuracy through high-quality feature representations.

### 2. Token-Level Semantic Alignment
- **High-Quality Representations**: Aligns Vision Mamba U-Net’s hidden states with image representations from pretrained visual encoders (e.g., CLIP).
- **Accelerated Convergence**: Reduces training time by incorporating external semantic knowledge.
- **Category-Agnostic Few-Shot Learning**: Effectively estimates poses without relying on category-specific datasets.

### 3. Multi-Attention Feature Fusion Module
- **Self-Attention**: Captures global dependencies for accurate pose estimation.
- **Linear Attention**: Efficiently processes linear feature relationships.
- **Gate Attention**: Dynamically filters and prioritizes critical features.
- **Adaptive Fusion**: Combines Vision Mamba U-Net’s linear features with attention-extracted semantic features for optimal pose estimation.

---

## Architecture

### KPMamba Framework
- **Vision Mamba Backbone**: Lightweight backbone for efficient and accurate feature extraction.
- **Token-Level Semantic Alignment**: Aligns intermediate hidden states with pretrained visual encoder outputs for rich semantic understanding.
- **Multi-Attention Fusion**: Leverages various attention mechanisms for robust feature integration.

---

## Experiments & Results

### Systematic Ablation Studies
- **Objective**: To determine the best configuration of feature fusion strategies and architecture design.
- **Key Findings**: Results confirm that the multi-attention feature fusion significantly improves pose estimation performance.

### Few-Shot Pose Estimation
- Demonstrates superior few-shot learning performance by leveraging external pretrained visual encoders.
- **Category-Agnostic Success**: The framework generalizes well across unseen categories.

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+

### Setup
```bash
# Clone the repository
git clone https://github.com/your_username/kpmamba.git

# Navigate to the project directory
cd kpmamba

# Install dependencies
pip install -r requirements.txt

