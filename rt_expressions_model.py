import time
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from rt_expression_dataset import ExpressionDataset, SyntheticExpressionDataset
from rt_expressions_plot import plot_training

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 128
num_epochs = 250 # <- 200
learning_rate = 0.001

train_set = ExpressionDataset(True)
train_supplement_set = SyntheticExpressionDataset(True)

train_loader = DataLoader(ConcatDataset([train_set, train_supplement_set]), batch_size=batch_size, shuffle=True)
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # No Concat

test_set = ExpressionDataset(False)
test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False)

l = []
a = []

def train_model(model):
    print("Training Model...")
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.squeeze()).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_loss /= len(train_loader)
        
        l.append(train_loss)
        a.append(train_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%")

    stop_time = time.time()
    print('Finished Training in ' + str(stop_time-start_time) + 's')

def evaluate_model(model):
    print("Evaluating Model...")

    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    # Calculate evaluation metrics
    accuracy = 100 * correct / total
    average_loss = total_loss / len(test_loader)

    print(f"Model Evaluation:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {average_loss:.4f}")
    
    return accuracy

# CNN Model
class FacialExpressionNet(nn.Module):
    def __init__(self):
        super(FacialExpressionNet, self).__init__()
        # Define layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),  # Adjusted based on the output shape after conv layers
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 7)  # Output layer for 7 classes
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc_layers(x)
        return x

net = FacialExpressionNet().to(device)

train_model(net)

# Export the model to ONNX format
print ("Saving Model...")
example_input = torch.randn(1, 1, 48, 48).cuda()  # Assuming input size is (batch_size, channels, height, width)
onnx_program = torch.onnx.export(net,
                  example_input,
                  input_names=["input"],
                  output_names=["output"],
                  f="expr_net_s_synth_final.onnx")

ac = evaluate_model(net)

plot_training(l, a, batch_size, num_epochs, learning_rate, ac, "9")