import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

df = pd.read_csv(".\extensionPhishing\datasset2.csv")
X= df.drop(columns=['phishing'])
y = df['phishing']
X_tensor = torch.tensor(X.values, dtype=torch.long)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Define the sizes for training, validation, and testing sets
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

X_np = X_tensor.numpy()
y_np = y_tensor.numpy()

# Split the dataset into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Convert features and labels into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Create DataLoader for training, validation, and testing sets
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Create DataLoader for training, validation, and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_filters, filter_sizes, num_classes, dropout_prob=0.5):
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        conv_outputs = [torch.relu(conv(x)) for conv in self.convs]
        pooled_outputs = [torch.max(conv_out, dim=2)[0] for conv_out in conv_outputs]
        cat_output = torch.cat(pooled_outputs, dim=1)
        cat_output = self.dropout(cat_output)
        logits = self.fc(cat_output)
        return logits

# Define hyperparameters
input_size = X_train.shape[1]  # Number of features
num_filters = 128
filter_sizes = [3, 4, 5]
num_classes = len(y.unique())  # Number of unique classes
dropout_prob = 0.5
learning_rate = 0.001
num_epochs = 10

# Initialize the model, loss function, and optimizer
model = CNNClassifier(input_size, num_filters, filter_sizes, num_classes, dropout_prob)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            _, predicted = torch.max(output, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    val_accuracy = correct / total
    
    # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        output = model(batch_X)
        _, predicted = torch.max(output, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy:.4f}')

example_string = "https://www.youtube.com"

# Tokenization
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(example_string)

# Build vocabulary
counter = Counter()
counter.update(tokens)
vocab = build_vocab_from_iterator([tokens], specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Numerical representation
numerical_tokens = [vocab[token] for token in tokens]

# Padding (if necessary)
max_seq_length = 100
padded_tokens = numerical_tokens[:max_seq_length] + [vocab["<pad>"]] * (max_seq_length - len(numerical_tokens))
input_tensor = torch.tensor(padded_tokens, dtype=torch.long)

# Ensure the tensor has the correct shape
input_tensor = input_tensor.to(torch.float).unsqueeze(0)

# Pass the input tensor through the model
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

# Interpret the prediction
predicted_class = predicted.item()
print("Predicted Class:", predicted_class)