import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Define device
mps_device = torch.device("mps")

# Define your model
class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # 4 classes instead of 3

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Draw contours on the original image
def draw_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = cv2.imread(image_path)
        image_with_contours = draw_contours(image)
        if self.transform:
            image_with_contours = self.transform(image_with_contours)
        return image_with_contours, label  # Return a tuple (image, label)

    def load_data(self):
        data = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.txt'):
                    annotation_file = os.path.join(root, file)
                    image_file = os.path.join(root, file.replace('.txt', '.jpg'))
                    with open(annotation_file, 'r') as f:
                        annotation = f.read().strip()
                        if annotation in ['centre', 'left', 'right', 'unsure'] and os.path.exists(image_file):
                            # Encode labels as integers (0, 1, 2, 3)
                            label = ['centre', 'left', 'right', 'unsure'].index(annotation)
                            data.append((image_file, label))
                        else:
                            print(f"Invalid annotation '{annotation}' or missing image file: {image_file}")
        return data

# Set directory and transform
directory = '/Users/rahulwadia/Downloads/imageframes'
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CustomDataset(directory, transform=transform)

# Split dataset into training and test sets
train_size = 0.8
train_data, test_data = train_test_split(dataset, train_size=train_size, shuffle=True)

# Define dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Instantiate model, loss function, and optimizer
model = ImprovedModel().to(device=mps_device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
train_loss_history = []
train_acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.to(mps_device))
        loss = criterion(outputs, labels.to(mps_device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels = labels.to(mps_device)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader.dataset)

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    train_loss_history.append(train_loss)
    train_acc_history.append(train_accuracy)

# Testing loop
model.eval()
test_correct = 0
test_total = 0
predictions = []
true_labels = []
outputs_list = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(mps_device)
        labels = labels.to(mps_device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        outputs_list.extend(outputs.cpu().numpy())

# Calculate metrics
test_accuracy = 100 * test_correct / test_total
conf_matrix = confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3])
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['centre', 'left', 'right', 'unsure'], yticklabels=['centre', 'left', 'right', 'unsure'])
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['centre', 'left', 'right', 'unsure'], yticklabels=['centre', 'left', 'right', 'unsure'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot training loss and accuracy versus epochs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Training Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy vs Epochs')
plt.legend()

plt.show()

# Precision-Recall Curve and AUC for each class
precision = dict()
recall = dict()
pr_auc = dict()
# Define true_labels_bin before the loop
true_labels_bin = label_binarize(true_labels, classes=[0, 1, 2, 3])

# Determine the number of classes
n_classes = true_labels_bin.shape[1]

# Initialize arrays to store precision, recall, and AUC
precision = [None] * n_classes
recall = [None] * n_classes
pr_auc = [None] * n_classes

# Loop over classes
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(true_labels_bin[:, i], [output[i] for output in outputs_list])
    pr_auc[i] = auc(recall[i], precision[i])
# Plot Precision-Recall curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# Define colors
colors = ['red', 'blue', 'green', 'orange']  # Add more colors if needed

# Now you can use the colors list in your code

for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR curve of class {i} (area = {pr_auc[i]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Epochs: {num_epochs})')
plt.legend(loc="lower left")

# ROC Curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], [output[i] for output in outputs_list])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.subplot(1, 2, 2)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic Curve (Epochs: {num_epochs})')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
