#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y \
  libegl1 \
  libgles2 \
  libgl1 \
  libglx-mesa0 \
  libgl1-mesa-dri \
  libosmesa6 \
  libglib2.0-0

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Run with:"
echo "source .venv/bin/activate"
echo "export MUJOCO_GL=osmesa"
echo "python3 -m Simulation.Cem.dream_cem {with argument}