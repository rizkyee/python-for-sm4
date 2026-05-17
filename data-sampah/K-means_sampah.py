import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. MEMBACA DATA
# Pastikan file data_sampah.csv ada di folder yang sama
try:
    df = pd.read_csv('data_sampah.csv')
    print("Berhasil memuat data.")
except FileNotFoundError:
    print("Error: File 'data_sampah.csv' tidak ditemukan!")
    exit()

# 2. MENYIAPKAN FITUR
# Kita pakai kolom Kadar-Air dan Kekerasan
X = df[['Kadar-Air', 'Kekerasan']].values

# 3. PROSES K-MEANS
# Membagi data menjadi 2 kelompok
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 4. MENGAMBIL KOORDINAT CENTROID
# Inilah koordinat pusat dari masing-masing cluster
centroids = kmeans.cluster_centers_

# 5. MEMBERI LABEL OTOMATIS (Organik/Anorganik)
avg_kadar_air = df.groupby('Cluster')['Kadar-Air'].mean()
if avg_kadar_air[0] > avg_kadar_air[1]:
    mapping = {0: 'Organik', 1: 'Anorganik'}
else:
    mapping = {1: 'Organik', 0: 'Anorganik'}
df['Kategori'] = df['Cluster'].map(mapping)

# 6. VISUALISASI
plt.figure(figsize=(12, 7))

# Plot titik-titik sampah
colors = {'Organik': '#2ecc71', 'Anorganik': '#e74c3c'}
for kategori, warna in colors.items():
    subset = df[df['Kategori'] == kategori]
    plt.scatter(subset['Kadar-Air'], subset['Kekerasan'],
                c=warna, label=kategori, s=100, edgecolors='black', alpha=0.8)

# --- LETAK CENTROIDNYA ---
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=300, linewidths=5,
            color='black', label='Centroids', zorder=10)
# ----------------------------

# Tambahkan nama benda di tiap titik (opsional)
for i, txt in enumerate(df['Benda']):
    plt.annotate(txt, (df['Kadar-Air'][i], df['Kekerasan'][i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.6)

plt.title('Hasil Clustering Sampah')
plt.xlabel('Kadar Air (%)')
plt.ylabel('Tingkat Kekerasan (1-10)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Menampilkan angka koordinat centroid di terminal
print("\nTitik Tengah (Centroid):")
for i, pt in enumerate(centroids):
    print(f"Centroid {mapping[i]}: Kadar Air {pt[0]:.2f}%, Kekerasan {pt[1]:.2f}")