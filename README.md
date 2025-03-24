# VisualRL - Pendulum

![Rendering_screenshot_24 03 2025](https://github.com/user-attachments/assets/9689f8e5-1e51-4546-b9b8-83c62670e18e)

A reinforcement learning project for visual-based control using RGB observations.

---

## 🛠️ Setup
```bash
conda create -n vis_rl python=3.10
conda activate vis_rl

git clone https://github.com/sanghyunryoo/VisualRL_POSTECH_Pendulum.git
cd VisualRL_POSTECH_Pendulum/

pip install -r requirements.txt

## 🚀 Training
```bash
python main_rgb.py

## 🎯 Evaluation
```bash
python main_rgb.py --eval True
