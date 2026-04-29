from flask import Flask, render_template, request
import os
import time
from detector import detect_plate

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder statis dan upload tersedia
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/result', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    text = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', text="Gagal unggah file")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', text="Nama file kosong")

        if file:
            # Gunakan timestamp agar nama file selalu unik
            filename = f"{int(time.time())}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Proses deteksi plat
            text, result_img = detect_plate(filepath)

            if result_img:
                # Tambahkan parameter ?t= untuk menghindari cache browser
                image_path = f"/{result_img}?t={int(time.time())}"

    return render_template('index.html', text=text, image=image_path)

if __name__ == '__main__':
    app.run(debug=True)