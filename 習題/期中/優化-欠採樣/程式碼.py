import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/kaggle/input/classifyleaves/train.csv')
test = pd.read_csv('/kaggle/input/classifyleaves/sample_submission.csv')
image_path = '/kaggle/input/classifyleaves/'
 # 指定预训练权重文件的路径
pretrained_weights_path = '/kaggle/input/resnet/resnet50-0676ba61.pth' 

# 创建类别映射
class_to_num = {label: i for i, label in enumerate(train['label'].unique())}
num_to_class = {v: k for k, v in class_to_num.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def resnet_model(num_classes, pretrained_weights=None):
    model_ft = resnet50(pretrained=False)
    if pretrained_weights is not None:
        model_ft.load_state_dict(torch.load(pretrained_weights))
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, num_classes)
    model_ft.bn = nn.BatchNorm1d(num_features)
    model_ft.dropout = nn.Dropout(0.5)
    return model_ft

num_classes = len(class_to_num)
model = resnet_model(num_classes, pretrained_weights_path)
model.to(device)

class LeavesDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.data.iloc[index]['image'])
        image = Image.open(image_path).convert('RGB')
        
        label = self.data.iloc[index]['label']
        label_num = class_to_num[label]
        
        if self.transform is not None:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']
        
        return image, label_num
    
    def __len__(self):
        return len(self.data)

def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.RandomRotate90(),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            albumentations.RandomBrightnessContrast(),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0),
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0),
        ]
    )

# 使用欠采样处理不平衡数据
def undersample_data(data, ratio):
    classes = data['label'].unique()
    samples_per_class = int(len(data) * ratio / len(classes))
    undersampled_data = pd.DataFrame()

    for class_label in classes:
        class_data = data[data['label'] == class_label]
        if len(class_data) < samples_per_class:
            undersampled_class_data = class_data
        else:
            undersampled_class_data = class_data.sample(samples_per_class, random_state=42)
        undersampled_data = pd.concat([undersampled_data, undersampled_class_data], axis=0)

    undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return undersampled_data, undersampled_data.index

train_data, train_indices = undersample_data(train, ratio=0.5)

# 輸出欠擷取後的數據分佈
resampled_data_counts = train_data['label'].value_counts().to_dict()
for key, value in resampled_data_counts.items():
    print(f'{key}: {value}')

# 设置随机种子以确保结果的可重复性
torch.manual_seed(42)
# 将图像转换为张量形式
data_transforms = ToTensor()

# k折交叉验证的折数
k_folds = 3
# 训练的总迭代次数
num_epochs = 5
# 训练批次的样本数量
batch_size = 48
# 学习率
learning_rate = 0.0001

# 损失函数，我使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam 优化器，它根据梯度自动调整学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
kfold = KFold(n_splits=k_folds, random_state=42, shuffle=True)

# 創建字典來記錄損失和準確度
losses_acc = {
    'train_acc': [],  
    'val_acc': []
}
# 創建字典來記錄損失和準確度
losses = {
    'train_loss': [], 
    'val_loss': [], 
}

val_labels_all = np.array([])
val_predictions_all = np.array([])
confusion_matrices = []
last_fold_val_labels = []
last_fold_val_predictions = []


for fold, (train_indices, val_indices) in enumerate(kfold.split(train_data.index)):
    print(f'--------------- Fold {fold + 1} ---------------')
    
    train_indices = [idx for idx in train_indices if idx in train_data.index]
    val_indices = [idx for idx in val_indices if idx in train_data.index]

    train_data_fold = train_data.loc[train_indices]
    val_data_fold = train_data.loc[val_indices]

    train_dataset = LeavesDataset(train_data_fold, image_path, transform=get_train_transforms())
    val_dataset = LeavesDataset(val_data_fold, image_path, transform=get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in tqdm(train_loader, desc=f'train {epoch+1}/{num_epochs}'):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_indices)
        losses['train_loss'].append(train_loss)
        losses_acc['train_acc'].append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        val_labels = []
        val_predictions = []

        for images, labels in tqdm(val_loader, desc=f'val {epoch+1}/{num_epochs}'):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

                val_loss += loss.item() * images.size(0)

                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_indices)
        losses['val_loss'].append(val_loss)
        losses_acc['val_acc'].append(val_acc)

        val_labels_all = np.concatenate((val_labels_all, np.array(val_labels)))
        val_predictions_all = np.concatenate((val_predictions_all, np.array(val_predictions)))
        
        if fold == kfold.get_n_splits() - 1:
            last_fold_val_labels.extend(val_labels)
            last_fold_val_predictions.extend(val_predictions)


        print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')
    
    # 仅记录最后一次折的val_labels和val_predictions
    if fold == kfold.get_n_splits() - 1:
        # 计算当前折的混淆矩阵
        confusion_matrices.append(confusion_matrix(val_labels, val_predictions))
    
# 计算最后一次折的准确率、F1 分数和召回率
accuracy = accuracy_score(last_fold_val_labels, last_fold_val_predictions)
f1 = f1_score(last_fold_val_labels, last_fold_val_predictions, average='weighted')
recall = recall_score(last_fold_val_labels, last_fold_val_predictions, average='weighted')
val_precision = precision_score(last_fold_val_labels, last_fold_val_predictions, average='weighted')

print(f'Precision: {val_precision:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')
print("finish!")
