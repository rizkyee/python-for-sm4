@echo off
echo ===============================
echo SETUP PRAKTIKUM PCD PYTHON
echo ===============================

echo.
echo [1] Membuat Virtual Environment...
python -m venv venv

echo.
echo [2] Mengaktifkan Environment...
call venv\Scripts\activate

echo.
echo [3] Upgrade pip...
python -m pip install --upgrade pip

echo.
echo [4] Install Library Praktikum...
pip install opencv-python matplotlib numpy scikit-image jupyter ipykernel

echo.
echo [5] Registrasi Kernel Jupyter...
python -m ipykernel install --user --name=pcd-env --display-name "PCD Python"

echo.
echo ===============================
echo SETUP SELESAI
echo ===============================
pause
