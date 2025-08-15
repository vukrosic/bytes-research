import torch
import torch.nn as nn

# Generate 10x training data as UTF-8 bytes
X, y = [], []
for _ in range(10):
    for a in range(1, 41):
        for b in range(1, 41):
            bytes_input = [ord(c) for c in f"{a:02d}{b:02d}"]  # 4 UTF-8 bytes input
            bytes_output = [ord(c) for c in f"{a+b:02d}"]      # 2 UTF-8 bytes output
            X.append(bytes_input)
            y.append(bytes_output)

X = torch.tensor(X, dtype=torch.float32) / 255  # Normalize input bytes
y = torch.tensor(y, dtype=torch.float32) / 255  # Normalize output bytes

# Larger neural network with more layers
net = nn.Sequential(
    nn.Linear(4, 64), nn.ReLU(),
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 2)
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Train
print(f"Training on {len(X)} examples...")
for epoch in range(10000):
    pred = net(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Training complete!")

# Test function
def test_sum(a, b):
    bytes_input = [ord(c) for c in f"{a:02d}{b:02d}"]
    x = torch.tensor([bytes_input], dtype=torch.float32) / 255
    pred_bytes = (net(x)[0] * 255).int().tolist()
    pred_chars = ''.join([chr(b) for b in pred_bytes])
    print(f"Input: {a}+{b} -> UTF-8 bytes: {bytes_input}")
    print(f"Output bytes: {pred_bytes} -> chars: '{pred_chars}'")
    return f"{a:02d}{b:02d}{pred_chars}"

# Examples
print(test_sum(3, 25))   # 032528
print(test_sum(15, 20))  # 152035
print(test_sum(7, 8))    # 070815