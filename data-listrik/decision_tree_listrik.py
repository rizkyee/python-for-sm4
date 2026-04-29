import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =========================
# 1. Import Data dari CSV
# =========================
try:
    df = pd.read_csv('data_listrik.csv')
    print("Data berhasil diimpor!")
    print(df.head())
except FileNotFoundError:
    print("File 'data_listrik.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
    exit()

# =========================
# 2. Preprocessing
# =========================
# Hemat = 1, Boros = 0
df['Label_Num'] = df['Label'].map({'Hemat': 1, 'Boros': 0})

# Fitur dan target
X = df[['Daya_Listrik', 'Pemakaian_kWh']]
y = df['Label_Num']

# =========================
# 3. Model Decision Tree (CART)
# =========================
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# =========================
# 4. Visualisasi
# =========================
plt.figure(figsize=(15, 6))

# -------------------------
# Subplot 1: Decision Boundary
# -------------------------
plt.subplot(1, 2, 1)

x_min, x_max = X['Daya_Listrik'].min() - 100, X['Daya_Listrik'].max() + 100
y_min, y_max = X['Pemakaian_kWh'].min() - 20, X['Pemakaian_kWh'].max() + 20

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 10),
    np.arange(y_min, y_max, 5)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')

# Plot data
plt.scatter(
    df[df['Label_Num'] == 1]['Daya_Listrik'],
    df[df['Label_Num'] == 1]['Pemakaian_kWh'],
    color='green',
    label='Hemat',
    edgecolors='k'
)

plt.scatter(
    df[df['Label_Num'] == 0]['Daya_Listrik'],
    df[df['Label_Num'] == 0]['Pemakaian_kWh'],
    color='red',
    label='Boros',
    edgecolors='k'
)

plt.title("Decision Boundary (Listrik Rumah)")
plt.xlabel("Daya Listrik (VA)")
plt.ylabel("Pemakaian (kWh)")
plt.legend()

# -------------------------
# Subplot 2: Pohon Keputusan
# -------------------------
plt.subplot(1, 2, 2)

plot_tree(
    model,
    feature_names=['Daya_Listrik', 'Pemakaian_kWh'],
    class_names=['Boros', 'Hemat'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Struktur Pohon Keputusan (Rules)")

plt.tight_layout()
plt.show()

# =========================
# 5. Testing Prediksi Manual
# =========================
print("\n=== TEST PREDIKSI ===")

# Contoh input baru
contoh = pd.DataFrame({
    'Daya_Listrik': [1300],
    'Pemakaian_kWh': [150]
})

prediksi = model.predict(contoh)

if prediksi[0] == 1:
    print("Prediksi: Hemat")
else:
    print("Prediksi: Boros")