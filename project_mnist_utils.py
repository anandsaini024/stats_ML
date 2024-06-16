import numpy as np
import torch
from torch import nn
from torchvision import datasets

import matplotlib.pyplot as plt


def load_data():
    
    train_set = datasets.MNIST(root="data/", train=True, download=True)
    train_data = train_set.data.view(train_set.data.shape[0], -1).float()
    train_targets = convert_to_one_hot_labels(train_set.targets)
    
    test_set = datasets.MNIST(root="data/", train=False, download=True)
    test_data = test_set.data.view(test_set.data.shape[0], -1).float()
    test_targets = convert_to_one_hot_labels(test_set.targets)

    # Normalise inplace.
    mu, std = train_data.mean(), train_data.std()
    train_data.sub_(mu).div_(std)
    test_data.sub_(mu).div_(std)
    
    return train_data, train_targets, test_data, test_targets


def convert_to_one_hot_labels(target):
    # Same rows as target, as many columns as the number of classes in target.
    tmp = torch.zeros(target.size(0), target.max() + 1)
    # Puts 1.0 in dimension 1 (columns) in the specified indexs in target.
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


def get_shallow_model(nb_hidden=100):
    model = nn.Sequential(
        nn.Linear(784, nb_hidden),
        nn.ReLU(),
        nn.Linear(nb_hidden, 10)
    )
    return model

def get_linear_model(input_size, num_classes):
    model = nn.Sequential(
        nn.Linear(input_size, num_classes)
    )
    return model

def get_deep_model(input_size, hidden_size1, hidden_size2, num_classes):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size1),
        nn.ReLU(),
        nn.Linear(hidden_size1, hidden_size2),
        nn.ReLU(),
        nn.Linear(hidden_size2, num_classes)
    )
    return model

def l1_penalty(model, lambda_l1):
    l1_loss =torch.tensor(0.).to(model[0].weight.device) 
    for param in model.parameters():
        l1_loss += torch.norm(param, 1)
    return lambda_l1 * l1_loss


def train_model_l1(model,nb_epochs=10, lr=1e-1, batch_size=100, lambda_l1=0):
    # Load all data.
    train_data, train_targets, test_data, test_targets = load_data()
    
    if train_targets.ndim > 1:
        train_targets = torch.argmax(train_targets, dim=1)
    if test_targets.ndim > 1:
        test_targets = torch.argmax(test_targets, dim=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(nb_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, targets in zip(train_data.split(batch_size), train_targets.split(batch_size)):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)

            # Add L1 regularization
            loss += l1_penalty(model, lambda_l1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_data)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation phase
        test_loss, test_accuracy = eval_model(model, criterion, test_data, test_targets)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{nb_epochs} - Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    return model, train_losses, train_accuracies, test_losses, test_accuracies


    return model, train_losses, train_accuracies, test_losses, test_accuracies

def train_model(model, lr = 1e-1, batch_size = 100, nb_epochs=50, lambda_l2=0):
    
    # Load all data.
    train_data, train_targets, test_data, test_targets = load_data()
    
    if train_targets.ndim > 1:
        train_targets = torch.argmax(train_targets, dim=1)
    if test_targets.ndim > 1:
        test_targets = torch.argmax(test_targets, dim=1)
    # Learning rate and batch size.
    lr, batch_size = lr, batch_size
    # Cross entropy loss and stochastic gradient descent.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Initialise list of outputs.
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for e in range(nb_epochs):
        # Training
        model.train()
        running_loss =0
        correct = 0
        total = 0
        # Iterate over train data in batches.
        for data, targets in zip(train_data.split(batch_size), train_targets.split(batch_size)):
            
            # Pass data through the model and compute loss.
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            
            # Apply regularisation.
            if lambda_l2 > 0:
                for p in model.parameters():
                    loss += lambda_l2 * p.pow(2).sum()
                    
            loss.backward()
            optimizer.step()
            running_loss += loss
            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_data)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation phase
        test_loss, test_accuracy = eval_model(model, criterion, test_data, test_targets)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {e + 1}/{nb_epochs} - Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    return model, train_losses, train_accuracies, test_losses, test_accuracies


def eval_model(model, criterion, test_data, test_targets):
    
    # Switch the model to eval mode (if you had any dropout or batchnorm, they are turned off).
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in zip(test_data.split(100), test_targets.split(100)):
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss  # Assuming loss is already a float
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    test_loss = total_loss / len(test_data)
    test_accuracy = correct / total
    # Switch back to training mode.
    model.train()
    
    return test_loss, test_accuracy

def visualize_weights(model):
    model.eval()
    with torch.no_grad():
        weights = model[0].weight  
        num_rows = weights.shape[0]

        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, ax in enumerate(axes.flatten()):
            if i >= num_rows:
                break
            weight_image = weights[i].view(28, 28).cpu().numpy()
            ax.imshow(weight_image, cmap='gray')
            ax.set_title(f'Class {i}')
            ax.axis('off')
        plt.show()

def plot_losses(model_losses):
    plt.figure(figsize=(10, 6))
    for model_name, losses in model_losses.items():
        plt.plot(losses, label=model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.title('Test Loss over Epochs for Different Models')
    plt.show()


def plot_accuracies(model_accuracies):
    plt.figure(figsize=(10, 6))
    for model_name, accuracy in model_accuracies.items():
        plt.plot(accuracy, label=model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.title('Test Accuracies over Epochs for Different Models')
    plt.show()
    
def get_prediction(model, test_data, test_targets):
    model.eval()  # Set the model to evaluation mode

    # Convert to PyTorch tensors if they are numpy arrays
    if isinstance(test_data, np.ndarray):
        test_data = torch.from_numpy(test_data).float()
    if isinstance(test_targets, np.ndarray):
        test_targets = torch.from_numpy(test_targets).long()

    with torch.no_grad():
        
        # Forward pass: Compute predicted labels by passing all test data to the model
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        _,test_labels=torch.max(test_targets,1)
        
        # Convert predicted and true labels to numpy arrays for compatibility with the confusion matrix function
        predicted_np = predicted.cpu().numpy()
        test_targets_np = test_labels.cpu().numpy()
        
        # Calculate accuracy
        correct = (predicted == test_labels).sum().item()  # Ensure it's an integer
        total = test_labels.size(0)  # Correctly get the total number of samples
        accuracy = correct / total
    return test_targets_np, predicted_np, accuracy

def confusion_matrix(true_labels, predictions, num_classes):
    # Initialize the confusion matrix with zeros
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(true_labels, predictions):
        conf_matrix[true][pred] += 1

    return conf_matrix

def deep_dropout_model(dropout_rate=0.5):
    model = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(64, 10)
    )
    return model


# if test_targets.ndim > 1:
#     test_targets = torch.argmax(test_targets, dim=1)

# if test_targets.dtype == torch.float32 or test_targets.dtype == torch.float64:
#     test_targets = test_targets.to(torch.int64)

# # Count the occurrences of each label
# label_counts = torch.bincount(test_targets)

# # To access the count of a specific label, say label '0'
# count_label_2 = label_counts[4]

# print(f"Number of samples with label 2: {count_label_2}")