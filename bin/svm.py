import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_clusters_per_class=1,
    class_sep=0.7
)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# SVM model class
class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

# Custom SVM hinge loss
class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
    
    def forward(self, output, target):
        # Convert target to {-1, 1}
        target = 2 * target - 1
        # Compute hinge loss
        loss = torch.mean(torch.clamp(1 - target * output, min=0))
        # Add regularization term
        reg = torch.mean(torch.pow(model.linear.weight, 2))
        return loss + 0.01 * reg

# Initialize model, loss, and optimizer
input_dim = X_train.shape[1]
model = SVM(input_dim)
criterion = HingeLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 1000
batch_size = 32
n_batches = len(X_train) // batch_size
train_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    # Shuffle data
    indices = torch.randperm(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get batch
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Record average loss for the epoch
    avg_epoch_loss = epoch_loss / n_batches
    train_losses.append(avg_epoch_loss)
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predicted = (test_outputs.squeeze() >= 0).float()
    accuracy = (predicted == y_test).float().mean()
    print(f'\nTest Accuracy: {accuracy:.4f}')

# Visualization of decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z.detach().numpy(), alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary with training data
plot_decision_boundary(X_train.numpy(), y_train.numpy(), model)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
