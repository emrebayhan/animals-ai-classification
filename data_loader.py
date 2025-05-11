import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import shutil

def split_dataset(base_dir='data/raw-img', target_base_dir='data/animals-10-split', train_ratio=0.8):
    if os.path.exists(target_base_dir):
        print(f"{target_base_dir} zaten mevcut, ayırma işlemi atlanıyor.")
        return

    os.makedirs(target_base_dir, exist_ok=True)
    train_dir = os.path.join(target_base_dir, 'train')
    val_dir = os.path.join(target_base_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=42)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
    print(f"Veri seti {target_base_dir} altına train/val olarak ayrıldı.")


RAW_DATA_DIR = 'C:\\Users\\bayha\\Documents\\dataset\\raw-img'
SPLIT_DATA_DIR = 'C:\\Users\\bayha\\Documents\\split'

split_dataset(base_dir=RAW_DATA_DIR, target_base_dir=SPLIT_DATA_DIR)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_data_path = os.path.join(SPLIT_DATA_DIR, 'train')
val_data_path = os.path.join(SPLIT_DATA_DIR, 'val')

image_datasets = {
    'train': datasets.ImageFolder(train_data_path, data_transforms['train']),
    'val': datasets.ImageFolder(val_data_path, data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"Sınıflar: {class_names}")
print(f"Sınıf sayısı: {num_classes}")
print(f"Eğitim veri seti boyutu: {dataset_sizes['train']}")
print(f"Doğrulama veri seti boyutu: {dataset_sizes['val']}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")
