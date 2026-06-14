import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ======================================================
# PROGRAM K-MEANS CLUSTERING 100 DATA PROVINSI INDONESIA
# Data berbentuk Provinsi-Tahun.
#
# Analisis menggunakan 3 variabel:
# 1. IPM
# 2. Kemiskinan
# 3. TPT
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_DATA = os.path.join(BASE_DIR, "data_provinsi.csv")
FILE_HASIL = os.path.join(BASE_DIR, "hasil_clustering_provinsi.csv")
FILE_GAMBAR = os.path.join(BASE_DIR, "hasil_clustering_provinsi_2D.png")


# ======================================================
# 1. MEMBACA DATA
# ======================================================

try:
    df = pd.read_csv(FILE_DATA)
    print("Berhasil memuat data_provinsi.csv.")
    print("Lokasi file data:", FILE_DATA)
except FileNotFoundError:
    print("Error: File data_provinsi.csv tidak ditemukan.")
    print("Jalankan terlebih dahulu file buat_data_provinsi.py")
    exit()


# ======================================================
# 2. CEK DATA
# ======================================================

print("\nJumlah data:", len(df))
print("\nData yang digunakan:")
print(df)

if len(df) != 100:
    print("\nPERINGATAN:")
    print("Jumlah data belum 100.")
    print("Jalankan ulang file buat_data_provinsi.py terlebih dahulu.")


# ======================================================
# 3. MENYIAPKAN VARIABEL UNTUK CLUSTERING
# ======================================================

X = df[["IPM", "Kemiskinan", "TPT"]].values


# ======================================================
# 4. STANDARISASI DATA
# ======================================================
# Standarisasi dilakukan agar IPM, Kemiskinan, dan TPT seimbang.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ======================================================
# 5. PROSES K-MEANS
# ======================================================

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)


# ======================================================
# 6. MENGAMBIL TITIK CENTROID
# ======================================================

centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)


# ======================================================
# 7. MEMBERI LABEL CLUSTER
# ======================================================
# IPM tinggi = lebih baik
# Kemiskinan rendah = lebih baik
# TPT rendah = lebih baik

df_scaled = pd.DataFrame(
    X_scaled,
    columns=["IPM_Z", "Kemiskinan_Z", "TPT_Z"]
)

df_scaled["Cluster"] = df["Cluster"]

rata_cluster_scaled = df_scaled.groupby("Cluster")[["IPM_Z", "Kemiskinan_Z", "TPT_Z"]].mean()

rata_cluster_scaled["Skor_Kesejahteraan"] = (
    rata_cluster_scaled["IPM_Z"]
    - rata_cluster_scaled["Kemiskinan_Z"]
    - rata_cluster_scaled["TPT_Z"]
)

urutan_cluster = rata_cluster_scaled["Skor_Kesejahteraan"].sort_values(ascending=False).index.tolist()

label_cluster = {
    urutan_cluster[0]: "Kesejahteraan Relatif Tinggi",
    urutan_cluster[1]: "Kesejahteraan Relatif Sedang",
    urutan_cluster[2]: "Kesejahteraan Relatif Rendah"
}

df["Kategori"] = df["Cluster"].map(label_cluster)


# ======================================================
# 8. MENGHITUNG SILHOUETTE SCORE
# ======================================================

silhouette = silhouette_score(X_scaled, df["Cluster"])


# ======================================================
# 9. VISUALISASI 2D
# ======================================================

plt.figure(figsize=(18, 10))

colors = {
    "Kesejahteraan Relatif Tinggi": "#2ecc71",
    "Kesejahteraan Relatif Sedang": "#f1c40f",
    "Kesejahteraan Relatif Rendah": "#e74c3c"
}

for kategori, warna in colors.items():
    subset = df[df["Kategori"] == kategori]

    plt.scatter(
        subset["IPM"],
        subset["Kemiskinan"],
        c=warna,
        label=kategori,
        s=90,
        edgecolors="black",
        alpha=0.80
    )


# Menampilkan centroid
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=400,
    linewidths=5,
    color="black",
    label="Centroids",
    zorder=10
)


# Label titik tidak ditampilkan semua agar grafik tidak terlalu penuh.
# Jika ingin menampilkan semua label, ubah False menjadi True.
TAMPILKAN_LABEL_SEMUA_TITIK = False

if TAMPILKAN_LABEL_SEMUA_TITIK:
    for i, row in df.iterrows():
        plt.annotate(
            f"{row['Provinsi']} {int(row['Tahun'])}",
            (row["IPM"], row["Kemiskinan"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5,
            alpha=0.75
        )


# Menambahkan label centroid
for i, pt in enumerate(centroids):
    plt.annotate(
        f"Centroid {i}\nTPT: {pt[2]:.2f}%",
        (pt[0], pt[1]),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="black"
    )


plt.title("Hasil Clustering 100 Data Provinsi-Tahun di Indonesia Menggunakan K-Means")
plt.xlabel("IPM")
plt.ylabel("Kemiskinan (%)")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)


# Sumbu dibuat 0 sampai 100
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.xticks(range(0, 101, 10))
plt.yticks(range(0, 101, 10))


plt.tight_layout()
plt.savefig(FILE_GAMBAR, dpi=300)
plt.show()


# ======================================================
# 10. OUTPUT DI TERMINAL
# ======================================================

print("\n==============================")
print("HASIL CLUSTERING K-MEANS")
print("==============================")

print("\nJumlah Data:", len(df))
print("Jumlah Cluster: 3")
print(f"Silhouette Score: {silhouette:.3f}")

print("\nTitik Tengah Centroid:")
for i, pt in enumerate(centroids):
    print(
        f"Centroid Cluster {i}: "
        f"IPM = {pt[0]:.2f}, "
        f"Kemiskinan = {pt[1]:.2f}%, "
        f"TPT = {pt[2]:.2f}%"
    )

print("\nRata-rata Setiap Cluster:")
print(df.groupby("Kategori")[["IPM", "Kemiskinan", "TPT"]].mean())

print("\nJumlah Data per Cluster:")
print(df["Kategori"].value_counts())

print("\nJumlah Data per Tahun:")
print(df["Tahun"].value_counts().sort_index())

print("\nHasil Clustering Lengkap:")
print(df[["Provinsi", "Tahun", "IPM", "Kemiskinan", "TPT", "Cluster", "Kategori"]])


# ======================================================
# 11. MENYIMPAN HASIL
# ======================================================

df.to_csv(FILE_HASIL, index=False)

cek_hasil = pd.read_csv(FILE_HASIL)

print("\nFile hasil_clustering_provinsi.csv berhasil dibuat.")
print("Lokasi file hasil:", FILE_HASIL)
print("Jumlah data pada file hasil:", len(cek_hasil))

print("File gambar hasil_clustering_provinsi_2D.png berhasil dibuat.")
print("Lokasi gambar:", FILE_GAMBAR)