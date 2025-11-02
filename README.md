# Snake-AI with VQC
This repository is a fork of **patrickloeber/snake-ai-pytorch**.
We replace the original PyTorch MLP policy network (`Linear_QNet`) with a **PennyLane-based Variational Quantum Circuit (VQC)** while keeping the original **training loop, replay buffer, rewards, and game logic** intact.

## Whats changed
- Replaced the original `Linear_QNet` with a Variational Quantum Circuit (VQC) backbone (`model_vqc.py`)
- Added render option - for the purpose of reducing delay in VQC
- Exposed knobs: n_qubits, layers, device_name.
- Recommended start: n_qubits=4\~6, layers=1\~2.

**Circuit sketch (per layer):**
- Input preprocessing: `LayerNorm -> Linear(in_dim -> n_qubits) -> Tanh`.
- Data re-uploading: for each qubit `w`, apply `RX(x_w * W_in[l,w,0])` and `RY(x_w * W_in[l,w,1])`.
- Trainable single-qubit rotations: `RX/RY/RZ(W_rot[l,w,*])`.
- Measurement: <Z> on each qubit -> feature dim = `n_qubits`.

## Quick Start
pip install torch numpy pygame matplotlib ipython pennylane pennylane-lightning
python agent.py


# Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame

In this Python Reinforcement Learning Tutorial series we teach an AI to play Snake! We build everything from scratch using Pygame and PyTorch. The tutorial consists of 4 parts:

You can find all tutorials on my channel: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)

- Part 1: I'll show you the project and teach you some basics about Reinforcement Learning and Deep Q Learning.
- Part 2: Learn how to setup the environment and implement the Snake game.
- Part 3: Implement the agent that controls the game.
- Part 4: Implement the neural network to predict the moves and train it.
