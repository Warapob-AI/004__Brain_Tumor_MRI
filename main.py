import numpy as np
import gdown
import sys
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
sys.path.append(os.path.abspath('004__Brain_Tumor_MRI/Scripts'))
from Scripts.train_model import train_model

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] 

path_model = "004__Brain_Tumor_MRI/Model/Best_Model.h5"

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def download_model(): 
    file_id = "1Ov2uovKm2WYu6qkOM3HqLaVJE3koYZmS" 

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    gdown.download(url, path_model, quiet=False)

if os.path.exists(path_model):
    model_path = path_model
else: 
    download_model()
    model_path = path_model

if __name__ == "__main__":
    if os.path.exists(model_path):
        print("📦 เจอโมเดลแล้ว กำลังโหลด...")
        model = load_model(model_path) 
        img_path = '004__Brain_Tumor_MRI/test_image.jpg' 
        img_array = preprocess_image(img_path)

        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]

        print("ภาพนี้น่าจะเป็น:", predicted_class)

    else:
        train_model() # ใช้เวลารันนานมาก ประมาณครึ่งวัน
