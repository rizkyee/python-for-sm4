# 📘 Sistem Pengenalan Gestur Tangan

## 📌 Deskripsi
Aplikasi Python untuk mendeteksi gestur tangan secara real-time menggunakan kamera.

Menggunakan:
- OpenCV (cv2)
- MediaPipe
- NumPy

Fitur:
- Deteksi jumlah jari
- Deteksi gesture (BUKA, TINJU, dll)
- Tampilan realtime (FPS, jumlah tangan, dll)
- Screenshot
- Kontrol keyboard

---

## ⚠️ Persyaratan

WAJIB:
- Python 3.10

JANGAN gunakan:
- Python 3.11 ke atas (terutama 3.14)

---

## 🛠️ Instalasi

### 1. Masuk ke folder project
```bash
cd D:\project-devlopments\python\deteksi-10-jari
```

### 2. Buat virtual environment
```bash
c:\laragon\bin\python\python-3.10\python.exe -m venv venv
```

### 3. Aktifkan venv (Windows)
```bash
venv\Scripts\activate
```

Jika berhasil:
```
(venv) D:\project-devlopments\python\deteksi-10-jari>
```

### 4. Install dependency
```bash
pip install opencv-python mediapipe numpy
```

### 5. Test install
```bash
python -c "import cv2, mediapipe; print('OK')"
```

Jika muncul:
```
OK
```

---

## ▶️ Cara Menjalankan

```bash
python main.py
```

---

## ⚙️ Setting VS Code

1. Tekan: `Ctrl + Shift + P`
2. Pilih: `Python: Select Interpreter`
3. Pilih:
```
venv\Scripts\python.exe
```

---

## 🎮 Kontrol

| Tombol | Fungsi |
|--------|--------|
| ESC    | Keluar |
| S      | Screenshot |
| L      | Toggle landmark |
| B      | Toggle box |
| F      | Toggle bar jari |
| R      | Reset |

---

## 📂 Struktur Project

```
deteksi-10-jari/
│
├── main.py
├── venv/
└── screenshots/
```

---

## ❌ Error Umum

### 1. cv2 tidak ditemukan
```
ModuleNotFoundError: No module named 'cv2'
```
Penyebab:
- Python beda dengan tempat install library

---

### 2. mediapipe error
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```
Penyebab:
- Environment salah
- Install rusak
- Bentrok nama file

---

## 🔍 Debug

Tambahkan di kode:
```python
import mediapipe as mp
print(mp)
print(mp.__file__)
```

Harus mengarah ke:
```
site-packages/mediapipe/
```

---

## 🚀 Cara Cepat

```bash
cd D:\project-devlopments\python\deteksi-10-jari

c:\laragon\bin\python\python-3.10\python.exe -m venv venv

venv\Scripts\activate

pip install opencv-python mediapipe numpy

python main.py
```

---

## 💡 Tips

- Gunakan 1 versi Python saja
- Selalu pakai venv
- Jangan langsung klik RUN di VS Code sebelum set interpreter