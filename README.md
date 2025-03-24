# Controlling Inverted Pendulum via Visual RL

![Rendering_screenshot_24 03 2025](https://github.com/user-attachments/assets/9689f8e5-1e51-4546-b9b8-83c62670e18e)

# VisualRL - Pendulum

A reinforcement learning project for visual-based control using RGB observations.

---

## 🛠️ Setup

### ✅ Create Conda Virtual Environment (Python 3.10 recommended)

```bash
conda create -n vis_rl python=3.10
conda activate vis_rl
pip install -r requirements.txt

## 🚀 Training
python main_rgb.py

## 🎯 Evaluation
python main_rgb.py --eval True
