# model_vqc.py
import os
# import time
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


class _VQCBackbone(nn.Module):
    def __init__(self, in_dim: int, n_qubits: int = 16, layers: int = 2, device_name: str = "lightning.qubit"):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        self.layers = layers

        self.dev = qml.device(device_name, wires=n_qubits)

        # parameter shapes
        weight_shapes = {
            "W_in":  (layers, n_qubits, 2),
            "W_rot": (layers, n_qubits, 3),
        }

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint", cache=False)
        def circuit(inputs, W_in, W_rot):
            # inputs: (n_qubits,)
            for l in range(self.layers):
                # data re-uploading
                for w in range(self.n_qubits):
                    qml.RX(inputs[w] * W_in[l, w, 0], wires=w)
                    qml.RY(inputs[w] * W_in[l, w, 1], wires=w)
                # trainable single-qubit rotations
                for w in range(self.n_qubits):
                    qml.RX(W_rot[l, w, 0], wires=w)
                    qml.RY(W_rot[l, w, 1], wires=w)
                    qml.RZ(W_rot[l, w, 2], wires=w)
                for w in range(self.n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_qubits)] + \
                [qml.expval(qml.PauliX(w)) for w in range(self.n_qubits)]

        # TorchLayer
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.pre = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, n_qubits),
            nn.Tanh(),
        )
        
        for name, p in self.q_layer.named_parameters():
            if p.requires_grad:
                nn.init.normal_(p, mean=0.0, std=0.01)


    def forward(self, x: torch.Tensor):
        x_small = self.pre(x).to(dtype=torch.float32)

        is_infer = not self.training
        if x_small.dim() == 1:
            if is_infer:
                with torch.no_grad():
                    z = self.q_layer(x_small)
            else:
                z = self.q_layer(x_small)
            return z
        else:
            outs = []
            if is_infer:
                with torch.no_grad():
                    for i in range(x_small.shape[0]):
                        outs.append(self.q_layer(x_small[i]))
            else:
                for i in range(x_small.shape[0]):
                    outs.append(self.q_layer(x_small[i]))
            return torch.stack(outs, dim=0)


class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                    n_qubits: int = 12, layers: int = 2, device_name: str = "lightning.qubit"):
        super().__init__()
        self.vqc = _VQCBackbone(input_size, n_qubits=n_qubits, layers=layers, device_name=device_name)

        self.head = nn.Sequential(
            nn.Linear(2*n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.vqc(x)
        q = self.head(z)
        return q

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # state = torch.tensor(state, dtype=torch.float)
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32))
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

