# **Super Mario Bros Reinforcement Learning Agent**

This project trains a **Reinforcement Learning (RL) agent** to play **Super Mario Bros** using a **Double Deep Q-Network (DDQN)**. The agent processes pixel-based observations, selects actions using an epsilon-greedy strategy, and optimizes policies with experience replay.

## Demo
https://github.com/onionsp/Super-Mario-RL/blob/main/mario_demo.mov
---

## **Techniques Used**

### **1. Double Deep Q-Networks (DDQN)**
- Uses two networks: **online network** for action selection and **target network** for evaluation.
- Mitigates **Q-value overestimation** present in standard DQNs.

### **2. Convolutional Neural Network (CNN) for Feature Extraction**
- CNN processes raw game frames to extract spatial features.
- **Architecture**:
  - **Conv Layers**: Extract spatial information from game frames.
  - **Fully Connected Layers**: Compute Q-values for actions.
- Optimized using **Adam optimizer** and **Mean Squared Error (MSE) Loss**.

### **3. Hardware Acceleration (M1 Pro GPU with Metal Backend)**
- **PyTorch MPS backend** utilized for faster training on Apple Silicon.
- Significantly speeds up computation compared to CPU-only training.

### **4. Frame-Skipping for Efficiency**
- Skips **4 frames per action** to reduce computation and improve training speed.
- Maintains essential motion while discarding redundant frames.

### **5. Experience Replay**
- Stores past transitions in a **Replay Buffer** to break temporal correlations.
- Enables efficient off-policy learning and prevents catastrophic forgetting.

### **6. Exploration-Exploitation Tradeoff**
- **Epsilon-greedy policy** balances exploration (random actions) and exploitation (choosing best-known action).
- **Epsilon decay** ensures initial exploration and later convergence to optimal policies.

### **7. Checkpointing & Model Persistence**
- Saves model weights periodically for resuming training or evaluation.
- Allows tracking progress and preventing data loss.

---

## **Setup & Training**

### **Installation**
```bash
git clone https://github.com/onionsp/Super-Mario-RL.git
cd Super-Mario-Bros-RL
conda create --name smbrl python=3.10.12
conda activate smbrl
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
