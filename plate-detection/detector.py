import cv2
import numpy as np
import re
from ultralytics import YOLO
import easyocr
import os

# 1. LOAD MODEL YOLO
model = YOLO("models/best.pt")

# 2. LOAD OCR
reader = easyocr.Reader(['en'])

def clean_plate_text(text):
    """Membersihkan teks dan menstandarisasi karakter yang sering salah baca."""
    text = text.upper()
    # Hilangkan karakter aneh
    text = re.sub(r'[^A-Z0-9 ]', '', text)
    return text.strip()

def detect_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Gambar tidak terbaca", None

    # --- DETEKSI LOKASI PLAT ---
    results = model(img, imgsz=1024, conf=0.25) 
    plate_img = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h_orig, w_orig, _ = img.shape
            pad = 15 
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w_orig, x2 + pad); y2 = min(h_orig, y2 + pad)
            plate_img = img[y1:y2, x1:x2]
            break 

    if plate_img is None:
        return "Plat tidak terdeteksi", None

    # --- PENAJAMAN GAMBAR ---
    plate_img = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_plate = cv2.filter2D(gray_plate, -1, kernel)

    # --- PROSES OCR CERDAS ---
    ocr_results = reader.readtext(sharpened_plate)

    if ocr_results:
        # Hitung tinggi maksimal untuk filter teks kecil (pajak)
        heights = [res[0][2][1] - res[0][0][1] for res in ocr_results]
        max_height = max(heights) if heights else 0
        h_crop, _ = sharpened_plate.shape
        
        valid_elements = []
        for res in ocr_results:
            box, current_text, prob = res
            center_y = (box[0][1] + box[2][1]) / 2
            current_height = box[2][1] - box[0][1]
            
            # FILTER: 
            # 1. Harus di atas 60% tinggi plat (buang baris pajak)
            # 2. Tinggi harus minimal 45% dari teks terbesar (buang noise)
            if center_y < (h_crop * 0.6) and current_height > (max_height * 0.45):
                valid_elements.append(res)

        # Urutkan dari kiri ke kanan
        valid_elements.sort(key=lambda x: x[0][0][0])
        
        # LOGIKA SPASI CERDAS:
        # Jika jarak horizontal antar kotak teks > tinggi teks, tambahkan spasi
        final_string = ""
        for i in range(len(valid_elements)):
            final_string += valid_elements[i][1]
            if i < len(valid_elements) - 1:
                curr_box_right = valid_elements[i][0][1][0]
                next_box_left = valid_elements[i+1][0][0][0]
                gap = next_box_left - curr_box_right
                avg_height = (valid_elements[i][0][2][1] - valid_elements[i][0][0][1])
                
                if gap > (avg_height * 0.3): # Jika ada celah signifikan
                    final_string += " "

        final_text = clean_plate_text(final_string)
    else:
        final_text = "Tidak terbaca"

    result_path = "static/result/cropped.jpg"
    cv2.imwrite(result_path, sharpened_plate)

    return final_text, result_path