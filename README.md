# Postech Pendulum for controlling Inverted Pendulum via Visual RL

![Rendering_screenshot_24 03 2025](https://github.com/user-attachments/assets/9689f8e5-1e51-4546-b9b8-83c62670e18e)


Create a Conda Virtual Environment (Python 3.10 recommended):
  conda create -n vis_rl python=3.10
  conda activate vis_rl
Install Dependencies:
  pip install -r requirements.txt

Train:
python main_rgb.py

Eval:
python main_rgb.py --eval True
