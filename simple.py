# Authors:
# Awni Altabaa provided the pytorch source code
# Ping Luo and John Lafferty added the code for the CPL module.
import torch
import torch.nn as nn
import torch.optim as optim
from cpl import CPL  # <====================


# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

# Data loader generator that produces random data
def data_loader(batch_size):
    while True:
        x = torch.randn(batch_size, 1)
        y = 3 * x + 2 + torch.randn(batch_size, 1) * 0.1  # y = 3x + 2 + noise
        yield x, y

# Set device to use bf16 mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, and optimizer
model = LinearRegressionModel().to(device)
model = torch.compile(model, mode='default')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create data loader
batch_size = 32
data_gen = data_loader(batch_size)

cpl = CPL()   # <====================

# Train the model
step_counter = 0
model.train()
for x, y in data_loader(batch_size):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    #with torch.cuda.amp.autocast(device, dtype=torch.bfloat16):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(x)
        loss = criterion(outputs, y)
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    step_counter += 1
    if step_counter % 1000 == 0:
        print(f"Step: {step_counter}; Loss: {loss.item():,.4f}")

    if cpl.check():  # <====================
        print()
        print('='*100)
        print('detected preemption flag inside training loop')
        print('exiting gracefully (saving model checkpoint, etc.) ...')
        torch.save(model.state_dict(), 'model_checkpoint.pt')
        # wandb.finish()
        # etc...
        print('exiting now')
        print('='*100)
        break

print('Done')
