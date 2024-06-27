# myapp/utils.py
def run_python_code(string):
    # Your Python code goes here
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader, random_split
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from collections import Counter
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    df = pd.read_csv("./URLChekcer/datasset.csv")
    X= df.drop(columns=['phishing'])
    y = df['phishing']
    X_tensor = torch.tensor(X.values, dtype=torch.long)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    

    urllen = len(string)
    dots = string.count(".")
    hypens = string.count("-")
    underline = string.count("_")
    slash = string.count("/")
    question = string.count("?")
    equal = string.count("=")
    at = string.count("@")
    ands = string.count("&")
    exclamation = string.count("!")
    space = string.count(" ")
    tilde = string.count("~")
    comma = string.count(",")
    plus = string.count("+")
    asterisk = string.count("*")
    hashtag = string.count("#")
    dollar = string.count("$")
    percent = string.count("%")


    http=string.count("http")
    http-=1
    redirecting = http + string.count("url=") + string.count("redirect") + string.count("next=") + string.count("out=") + string.count("view")

    elements = [urllen, dots, hypens, underline, slash, question, equal, at, ands, exclamation, space, tilde, comma, plus, asterisk, hashtag, dollar,percent, redirecting]
    testing = torch.tensor(elements, dtype=torch.float)


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
    batch_size = 19
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # Define hyperparameters
    input_size = 19
    num_classes = 2
    learning_rate = 0.001
    num_epochs = 10



    class CNNClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(CNNClassifier, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
            self.flattened_size = 32 * 1 * (input_size // 4+2)
            
            self.fc = nn.Linear(self.flattened_size, num_classes)

        def forward(self, x):
            # Apply first convolutional layer followed by ReLU activation and max pooling
            x = x.unsqueeze(1).unsqueeze(2)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            
            # Apply second convolutional layer followed by ReLU activation and max pooling
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            
            # Flatten the output for fully connected layer
            x = x.view(x.size(0), -1)
            
            # Apply fully connected layer
            logits = self.fc(x)
            return logits

    # Initialize the model, loss function, and optimizer
    model = CNNClassifier( input_size,num_classes)
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

    value = 0
    # Evaluate the model on your own input data
    model.eval()
    with torch.no_grad():
        output = model(testing.unsqueeze(0))  # Unsqueeze to add a batch dimension
        _, predicted = torch.max(output, 1)
        value = output


    output_sum = output.sum()

    if output_sum > 0.5:
        result = "Phishing URL"
    else:
        result = "Legit URL"
        
    return result, test_accuracy
