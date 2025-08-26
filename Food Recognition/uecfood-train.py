import torch  
import torch.nn as nn  
import torch.optim as optim  
import torchvision  
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader  
from torchvision.datasets import ImageFolder   
import matplotlib.pyplot as plt  
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  

# 检查是否有可用的GPU  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
print(f'Using device: {device}')  # 打印使用的设备  

# 数据预处理和增强  
transform = transforms.Compose([  
    transforms.Resize((320, 320)),  # 调整图像大小  
    transforms.RandomHorizontalFlip(),  # 随机水平翻转  
    transforms.RandomVerticalFlip(),  # 随机垂直翻转  
    transforms.ToTensor(),  # 转换为Tensor  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化处理  
])  

# 加载数据集  
train_dataset = ImageFolder(root='路径/UECFOOD100/train', transform=transform)  
test_dataset = ImageFolder(root='路径/UECFOOD100/test', transform=transform)  

# 创建数据加载器  
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)  

print("训练集数量:", len(train_dataset))  # 打印训练集数量  
print("测试集数量:", len(test_dataset))  # 打印测试集数量  

# 定义自定义ResNet-50模型  
class ResNet50(nn.Module):  
    def __init__(self):  
        super(ResNet50, self).__init__()  
        self.resnet50 = torchvision.models.resnet50(pretrained=True)  # 加载预训练的ResNet-50模型  
        # 修改第一个卷积层以接受3通道输入（如有必要，更改输入通道数）  
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        # 更改最后一层以输出正确的类别数  
        num_ftrs = self.resnet50.fc.in_features  
        self.resnet50.fc = nn.Linear(num_ftrs, 100)  

    def forward(self, x):  
        return self.resnet50(x)  

# 初始化模型、损失函数和优化器  
model = ResNet50().to(device)  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  

# 训练过程并跟踪损失  
epochs = 50 
train_loss_history = []  

for epoch in range(epochs):  
    running_loss = 0.0  
    for inputs, labels in trainloader:  
        inputs, labels = inputs.to(device), labels.to(device)  

        optimizer.zero_grad()  # 参数梯度清零  
        outputs = model(inputs)  # 前向传播  
        loss = criterion(outputs, labels)   
        loss.backward()   
        optimizer.step()  # 更新权重  

        running_loss += loss.item() * inputs.size(0)  
    
    # 计算每个epoch的平均损失  
    epoch_loss = running_loss / len(trainloader.dataset)  
    train_loss_history.append(epoch_loss)  
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')  

# 绘制训练损失曲线  
plt.figure(figsize=(10, 5))  
plt.plot(range(1, epochs + 1), train_loss_history, marker='o')  
plt.title('Training Loss Curve')  
plt.xlabel('Epochs')  
plt.ylabel('Loss')  
plt.grid()  
plt.xlim(1, epochs)  
plt.ylim(0, max(train_loss_history) * 1.1) 
plt.show()

plot_filename = os.path.join('./', 'training_loss_curve.png') 
plt.savefig(plot_filename) 
print('Finished Training')  # 训练完成  

# 保存训练好的模型  
PATH = './UECFood_resnet50.pth'  
torch.save(model.state_dict(), PATH)  

# 加载训练好的模型进行评估  
model.eval()  # 将模型设置为评估模式  
model.load_state_dict(torch.load(PATH))  

# 评估模型  
correct = 0  
total = 0  
y_true, y_pred = [], []  
with torch.no_grad():  
    for inputs, labels in testloader:  
        inputs, labels = inputs.to(device), labels.to(device)  
        outputs = model(inputs)  
        _, predicted = torch.max(outputs.data, 1)  

        y_true.extend(labels.cpu().numpy())  
        y_pred.extend(predicted.cpu().numpy())  

# 计算评估指标  
precision = precision_score(y_true, y_pred, average='weighted')    
recall = recall_score(y_true, y_pred, average='weighted')  
f1 = f1_score(y_true, y_pred, average='weighted')  
accuracy = accuracy_score(y_true, y_pred)  

# 打印评估指标  
print(f'Precision: {precision:.4f}')  
print(f'Recall: {recall:.4f}')  
print(f'F1 Score: {f1:.4f}')  
print(f'Accuracy: {accuracy:.4f}')