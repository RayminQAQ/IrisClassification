# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# For binary classification, let's consider classifying whether the species is Setosa or not
# Convert the problem to binary classification
y = (y == 0).astype(int)  # Setosa vs. Non-Setosa

# Split the dataset into training and testing sets
X_train, X_test, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train_np.astype(np.float32)).unsqueeze(1)
y_test = torch.from_numpy(y_test_np.astype(np.float32)).unsqueeze(1)

# Define a simple feedforward neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer for binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Instantiate the model
input_size = X_train.shape[1]
hidden_size = 10
model = NeuralNet(input_size, hidden_size)
# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss (every 10 epochs)
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Switch model to evaluation mode
model.eval()

# Disable gradient computation
with torch.no_grad():
    # Predict on the test set
    y_pred_probs = model(X_test)
    y_pred = (y_pred_probs >= 0.5).float()
    
    # Convert tensors to numpy arrays
    y_test_np = y_test.numpy()
    y_pred_np = y_pred.numpy()
    y_pred_probs_np = y_pred_probs.numpy()
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision = precision_score(y_test_np, y_pred_np)
    recall = recall_score(y_test_np, y_pred_np)
    auc = roc_auc_score(y_test_np, y_pred_probs_np)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'AUC: {auc:.4f}')
