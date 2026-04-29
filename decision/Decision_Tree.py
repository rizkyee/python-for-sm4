import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Import Data dari CSV
try:
    df = pd.read_csv('data_postur.csv')
    print("Data berhasil diimpor!")
except FileNotFoundError:
    print("File 'data_postur.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")

# 2. Preprocessing: Ubah Label teks ke angka (Label Encoding)
# Ideal = 1, Tidak Ideal = 0
df['Label_Num'] = df['Label'].map({'Ideal': 1, 'Tidak Ideal': 0})

X = df[['Berat', 'Tinggi']]
y = df['Label_Num']

# 3. Inisialisasi Model Decision Tree (CART)
# Menggunakan Gini Impurity sebagai kriteria utama CART
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model.fit(X, y)

# 4. Visualisasi
plt.figure(figsize=(15, 6))

# Subplot 1: Decision Boundary (Area Keputusan)
plt.subplot(1, 2, 1)
x_min, x_max = X['Berat'].min() - 5, X['Berat'].max() + 5
y_min, y_max = X['Tinggi'].min() - 5, X['Tinggi'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
plt.scatter(df[df['Label_Num']==1]['Berat'], df[df['Label_Num']==1]['Tinggi'], color='green', label='Ideal', edgecolors='k')
plt.scatter(df[df['Label_Num']==0]['Berat'], df[df['Label_Num']==0]['Tinggi'], color='red', label='Tidak Ideal', edgecolors='k')

plt.title("Decision Boundary (CART dari CSV)")
plt.xlabel("Berat Badan (kg)")
plt.ylabel("Tinggi Badan (cm)")
plt.legend()

# Subplot 2: Visualisasi Pohon Keputusan
plt.subplot(1, 2, 2)
plot_tree(model,
          feature_names=['Berat', 'Tinggi'],
          class_names=['Tidak Ideal', 'Ideal'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Struktur Pohon Keputusan (Rules)")

plt.tight_layout()
plt.show()