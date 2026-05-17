import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ======================================================
# PROGRAM K-MEANS CLUSTERING PROVINSI INDONESIA
# Analisis menggunakan 3 variabel:
# 1. IPM
# 2. Kemiskinan
# 3. TPT
#
# Visualisasi dibuat 2D agar mirip contoh dosen.
# ======================================================


# 1. MEMBACA DATA
try:
    df = pd.read_csv("data_provinsi.csv")
    print("Berhasil memuat data_provinsi.csv.")
except FileNotFoundError:
    print("Error: File data_provinsi.csv tidak ditemukan.")
    print("Jalankan terlebih dahulu file buat_data_provinsi.py")
    exit()


# 2. CEK DATA
print("\nJumlah data:", len(df))
print("\nData yang digunakan:")
print(df)

if len(df) < 30:
    print("\nPERINGATAN:")
    print("Jumlah data terlihat belum lengkap.")
    print("Jika memakai data tahun 2022, seharusnya mendekati 34 provinsi.")


# 3. MENYIAPKAN 3 VARIABEL UNTUK CLUSTERING
X = df[["IPM", "Kemiskinan", "TPT"]].values


# 4. STANDARISASI DATA
# Standarisasi dilakukan agar skala IPM, Kemiskinan, dan TPT seimbang.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. PROSES K-MEANS
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)


# 6. MENGAMBIL TITIK CENTROID
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)


# 7. MEMBERI LABEL CLUSTER
# IPM tinggi dianggap lebih baik.
# Kemiskinan rendah dianggap lebih baik.
# TPT rendah dianggap lebih baik.

rata_cluster = df.groupby("Cluster")[["IPM", "Kemiskinan", "TPT"]].mean()

rata_cluster["Skor_Kesejahteraan"] = (
    rata_cluster["IPM"]
    - rata_cluster["Kemiskinan"]
    - rata_cluster["TPT"]
)

urutan_cluster = rata_cluster["Skor_Kesejahteraan"].sort_values(ascending=False).index.tolist()

label_cluster = {
    urutan_cluster[0]: "Kesejahteraan Relatif Tinggi",
    urutan_cluster[1]: "Kesejahteraan Relatif Sedang",
    urutan_cluster[2]: "Kesejahteraan Relatif Rendah"
}

df["Kategori"] = df["Cluster"].map(label_cluster)


# 8. MENGHITUNG SILHOUETTE SCORE
silhouette = silhouette_score(X_scaled, df["Cluster"])


# 9. VISUALISASI 2D SEPERTI CONTOH DOSEN
plt.figure(figsize=(14, 8))

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
        s=120,
        edgecolors="black",
        alpha=0.85
    )

# Menampilkan centroid
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=350,
    linewidths=5,
    color="black",
    label="Centroids",
    zorder=10
)

# Menambahkan nama provinsi pada setiap titik
for i, row in df.iterrows():
    plt.annotate(
        row["Provinsi"],
        (row["IPM"], row["Kemiskinan"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        alpha=0.75
    )

# Menambahkan label centroid
for i, pt in enumerate(centroids):
    plt.annotate(
        f"Centroid {i}\nTPT: {pt[2]:.2f}%",
        (pt[0], pt[1]),
        xytext=(10, -25),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="black"
    )

plt.title("Hasil Clustering Beberapa Provinsi di Indonesia Menggunakan K-Means")
plt.xlabel("IPM")
plt.ylabel("Kemiskinan (%)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("hasil_clustering_provinsi_2D.png", dpi=300)
plt.show()


# 10. OUTPUT DI TERMINAL
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

print("\nJumlah Provinsi per Cluster:")
print(df["Kategori"].value_counts())

print("\nHasil Clustering Lengkap:")
print(df[["Provinsi", "IPM", "Kemiskinan", "TPT", "Cluster", "Kategori"]])


# 11. MENYIMPAN HASIL
df.to_csv("hasil_clustering_provinsi.csv", index=False)

print("\nFile hasil_clustering_provinsi.csv berhasil dibuat.")
print("File gambar hasil_clustering_provinsi_2D.png berhasil dibuat.")