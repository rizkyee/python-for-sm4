#!/bin/bash

echo "==============================="
echo "SETUP PRAKTIKUM PCD PYTHON"
echo "==============================="

echo "[1] Membuat Virtual Environment..."
python3 -m venv venv

echo "[2] Aktivasi..."
source venv/bin/activate

echo "[3] Upgrade pip..."
pip install --upgrade pip

echo "[4] Install Library..."
pip install opencv-python matplotlib numpy scikit-image jupyter ipykernel

echo "[5] Registrasi Kernel..."
python -m ipykernel install --user --name=pcd-env --display-name "PCD Python"

echo "==============================="
echo "SETUP SELESAI"
echo "==============================="
