import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import numpy as np
from utils import plot_epoch_data 

# Define the function to train and evaluate a neural network model on the Iris dataset
def train_and_evaluate_iris_model(num_epochs=100, hidden_size=10, learning_rate=0.01):
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels (0, 1, 2 for 3 classes)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert numpy arrays to PyTorch tensors
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train_np.astype(np.int64))  # Use np.int64 for multi-class labels
    y_test = torch.from_numpy(y_test_np.astype(np.int64))

    # Define the neural network class for multi-class classification
    class NeuralNetMultiClass(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNetMultiClass, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer with 'num_classes' units

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)  # CrossEntropyLoss will apply softmax, no need to do it here
            return out

    # Instantiate the model
    input_size = X_train.shape[1]  # 4 features in the Iris dataset
    num_classes = 3  # There are 3 classes in the Iris dataset
    model = NeuralNetMultiClass(input_size, hidden_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Switch to evaluation mode
    model.eval()

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Predict on the test set
        outputs = model(X_test)
        probabilities = torch.softmax(outputs, dim=1)  # Get probabilities for each class
        probabilities_np = probabilities.numpy()

        # Get the predicted class labels
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        predicted_np = predicted.numpy()
        y_test_np = y_test.numpy()

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test_np, predicted_np)
        precision = precision_score(y_test_np, predicted_np, average='weighted')
        recall = recall_score(y_test_np, predicted_np, average='weighted')
        f1 = f1_score(y_test_np, predicted_np, average='weighted')
        auc = roc_auc_score(y_test_np, probabilities_np, multi_class='ovr', average='weighted')

        # Print evaluation metrics
        print("=== Overall result ===")
        print(f'Total epoch: {num_epochs}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC: {auc:.4f}')

    return accuracy, precision, recall, f1, auc

# Call the function to train and evaluate the model
if __name__ == '__main__':
    state = {
        "epoch": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "AUC": [],
    }

    for epoch in state["epoch"]:
        accuracy, precision, recall, f1, auc = train_and_evaluate_iris_model(
            num_epochs=epoch, hidden_size=10, learning_rate=0.01
        )
        state["Accuracy"].append(accuracy)
        state["Precision"].append(precision)
        state["Recall"].append(recall)
        state["F1 Score"].append(f1)
        state["AUC"].append(auc)

    for data in state.keys():
        if data == "epoch":
            continue
        plot_epoch_data(
            state["epoch"],
            state[data],
            save_path="Result",
            filename=f'{data}-plot.png',
            title=f'Epoch vs {data}',
            xlabel="Epoch",
            ylabel=data
        )
