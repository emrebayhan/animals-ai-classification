import torch
import torch.optim as optim
import torch.nn as nn
import time
import copy
from tqdm import tqdm

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/inputs.size(0))


            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'animals10_best_model.pth')
                print(f"Yeni en iyi model kaydedildi: val_acc = {best_acc:.4f}")


    time_elapsed = time.time() - since
    print(f'Eğitim {time_elapsed // 60:.0f}d {time_elapsed % 60:.0f}s sürdü')
    print(f'En iyi val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history


if __name__ == '__main__':
    from data_loader import dataloaders, dataset_sizes, device, num_classes, class_names, SPLIT_DATA_DIR
    from model import get_model
    import os

    if not os.path.exists(SPLIT_DATA_DIR) or not os.path.exists(os.path.join(SPLIT_DATA_DIR, 'train')):
        print(f"Lütfen önce `data_loader.py` içindeki `split_dataset()` fonksiyonunu çalıştırarak")
        print(f"veri setini '{SPLIT_DATA_DIR}' altına ayırın ve sonra bu scripti çalıştırın.")
        exit()

    print("Model yükleniyor...")
    model_ft = get_model(num_classes=num_classes, pretrained=True)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("Eğitim başlıyor...")
    trained_model, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                         dataloaders, dataset_sizes, device, num_epochs=15)

    torch.save(trained_model.state_dict(), 'animals10_final_model.pth')
    print("Eğitilmiş model 'animals10_final_model.pth' olarak kaydedildi.")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('training_history.png')
    print("Eğitim grafikleri 'training_history.png' olarak kaydedildi.")
    
    print("Sınıf isimleri 'class_names.txt' dosyasına kaydediliyor...")
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Sınıf isimleri kaydedildi.") 