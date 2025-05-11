import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    from data_loader import dataloaders, device, class_names, num_classes, SPLIT_DATA_DIR
    from model import get_model
    import os

    if not os.path.exists(SPLIT_DATA_DIR) or not os.path.exists(os.path.join(SPLIT_DATA_DIR, 'val')):
        print(f"Lütfen önce `data_loader.py` içindeki `split_dataset()` fonksiyonunu çalıştırarak")
        print(f"veri setini '{SPLIT_DATA_DIR}' altına ayırın ve sonra bu scripti çalıştırın.")
        exit()

    MODEL_PATH = 'animals10_best_model.pth'
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} bulunamadı. Lütfen önce modeli eğitin.")
        exit()

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    if 'val' not in dataloaders:
        print("Doğrulama veri yükleyicisi (dataloaders['val']) bulunamadı.")
        exit()

    print("Model değerlendiriliyor...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['val'], desc="Değerlendirme"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nSınıflandırma Raporu:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    print("\nKarışıklık Matrisi:")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Karışıklık Matrisi')
    plt.savefig('confusion_matrix.png')
    print("Karışıklık matrisi 'confusion_matrix.png' olarak kaydedildi.")wa