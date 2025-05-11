import os
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

MODEL_PATH = 'animals10_best_model.pth'

try:
    with open("class_names.txt", "r") as f:
        class_names_from_file = [line.strip() for line in f.readlines()]
    CLASS_NAMES = class_names_from_file
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Sınıf isimleri dosyadan yüklendi: {CLASS_NAMES}")
except FileNotFoundError:
    print("class_names.txt bulunamadı. Lütfen eğitim sırasında oluşturulan sınıf listesini kullanın.")
    CLASS_NAMES = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
    NUM_CLASSES = 10
    print(f"UYARI: Varsayılan sınıf isimleri kullanılıyor: {CLASS_NAMES}")


from data_loader import device
from model import get_model

model_inf = get_model(num_classes=NUM_CLASSES, pretrained=False)
model_inf.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_inf = model_inf.to(device)
model_inf.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_pil):
    if image_pil is None:
        return "Lütfen bir resim yükleyin."

    if image_pil.mode == 'RGBA':
        image_pil = image_pil.convert('RGB')
    
    img_tensor = preprocess(image_pil)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model_inf(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    
    return confidences

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Hayvan Resmi Yükle"),
    outputs=gr.Label(num_top_classes=3, label="Tahminler"),
    title="Hayvan Sınıflandırıcı (Animals-10)",
    description="Animals-10 veri seti ile eğitilmiş bir model kullanarak hayvan resimlerini sınıflandırın. Bir resim yükleyin ve 'Submit' butonuna tıklayın.",
)

if __name__ == '__main__':
    os.makedirs("examples", exist_ok=True)
    print("Gradio arayüzü başlatılıyor... Tarayıcınızda http://127.0.0.1:7860 adresini açın.")
    iface.launch()