import pandas as pd
import matplotlib.pyplot as plt

# ================= IMPORT DATA =================
dataset = pd.read_csv('frekuensi_olahraga_vs_denyut_jantung.csv')

print(dataset.head())

# ================= VISUALISASI =================
plt.scatter(
    dataset['Frekuensi Olahraga (kali/minggu)'],
    dataset['Denyut Jantung Istirahat (bpm)']
)

plt.xlabel('Frekuensi Olahraga (kali/minggu)')
plt.ylabel('Denyut Jantung Istirahat (bpm)')
plt.title('Frekuensi Olahraga vs Denyut Jantung Istirahat')
plt.show()

# ================= PEMISAHAN DATA =================
X = dataset[['Frekuensi Olahraga (kali/minggu)']].values
y = dataset['Denyut Jantung Istirahat (bpm)'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# ================= REGRESI LINEAR =================
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Prediksi:", y_pred)
print("Data Asli:", y_test)

# ================= VISUALISASI MODEL =================
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train), color='blue')

plt.title('Frekuensi Olahraga vs Denyut Jantung (Training Set)')
plt.xlabel('Frekuensi Olahraga (kali/minggu)')
plt.ylabel('Denyut Jantung Istirahat (bpm)')
plt.show()

# ================= PREDIKSI BARU =================
frekuensi_olahraga = 3  # kali per minggu
prediksi_denyut = model.predict([[frekuensi_olahraga]])

print('Prediksi denyut jantung istirahat:', prediksi_denyut)