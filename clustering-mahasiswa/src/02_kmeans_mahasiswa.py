from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ======================================================
# K-MEANS CLUSTERING DATA MAHASISWA
#
# Program ini akan:
# 1. Membaca data mahasiswa
# 2. Melakukan clustering K-Means
# 3. Menyimpan hasil ke folder output
# 4. Menampilkan grafik langsung di layar
# ======================================================


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

FILE_DATA = DATA_RAW_DIR / "data_mahasiswa.csv"
FILE_HASIL = OUTPUT_DIR / "hasil_clustering_mahasiswa.csv"
FILE_RINGKASAN = OUTPUT_DIR / "ringkasan_cluster.csv"
FILE_CENTROID = OUTPUT_DIR / "centroid_cluster.csv"
FILE_GRAFIK_IPK_NILAI = OUTPUT_DIR / "grafik_ipk_nilai.png"
FILE_GRAFIK_PCA = OUTPUT_DIR / "grafik_pca_2d.png"

JUMLAH_CLUSTER = 3
RANDOM_STATE = 42
TAMPILKAN_GRAFIK = True

FITUR = [
    "IPK",
    "Kehadiran",
    "Nilai_Akhir",
    "Jam_Belajar_Mingguan",
    "Jumlah_Terlambat"
]

URUTAN_KATEGORI = [
    "Prestasi Akademik Relatif Tinggi",
    "Prestasi Akademik Relatif Sedang",
    "Prestasi Akademik Relatif Rendah"
]

WARNA_KATEGORI = {
    "Prestasi Akademik Relatif Tinggi": "#2ecc71",
    "Prestasi Akademik Relatif Sedang": "#f1c40f",
    "Prestasi Akademik Relatif Rendah": "#e74c3c"
}


def baca_data():
    if not FILE_DATA.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {FILE_DATA}\n"
            "Jalankan dulu: python src/01_buat_data_mahasiswa.py"
        )

    df = pd.read_csv(FILE_DATA)

    kolom_wajib = [
        "No",
        "NIM",
        "Nama",
        "Program_Studi",
        "Semester",
        "Tahun_Masuk",
        "IPK",
        "Kehadiran",
        "Nilai_Tugas",
        "Nilai_UTS",
        "Nilai_UAS",
        "Nilai_Akhir",
        "Jam_Belajar_Mingguan",
        "Jumlah_Terlambat"
    ]

    kolom_hilang = [kolom for kolom in kolom_wajib if kolom not in df.columns]

    if kolom_hilang:
        raise ValueError(f"Kolom wajib belum ada: {kolom_hilang}")

    for kolom in FITUR:
        df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

    df = df.dropna(subset=FITUR).reset_index(drop=True)

    if len(df) < JUMLAH_CLUSTER:
        raise ValueError("Jumlah data terlalu sedikit untuk K-Means.")

    return df


def beri_label_cluster(df, model):
    centroid_scaled = pd.DataFrame(
        model.cluster_centers_,
        columns=[
            "IPK_Z",
            "Kehadiran_Z",
            "Nilai_Akhir_Z",
            "Jam_Belajar_Mingguan_Z",
            "Jumlah_Terlambat_Z"
        ]
    )

    centroid_scaled["Cluster"] = range(JUMLAH_CLUSTER)

    # IPK tinggi lebih baik.
    # Kehadiran tinggi lebih baik.
    # Nilai akhir tinggi lebih baik.
    # Jam belajar tinggi lebih baik.
    # Jumlah terlambat rendah lebih baik.
    centroid_scaled["Skor_Prestasi"] = (
        centroid_scaled["IPK_Z"]
        + centroid_scaled["Kehadiran_Z"]
        + centroid_scaled["Nilai_Akhir_Z"]
        + centroid_scaled["Jam_Belajar_Mingguan_Z"]
        - centroid_scaled["Jumlah_Terlambat_Z"]
    )

    urutan_cluster = (
        centroid_scaled
        .sort_values("Skor_Prestasi", ascending=False)["Cluster"]
        .tolist()
    )

    label_cluster = {
        urutan_cluster[0]: "Prestasi Akademik Relatif Tinggi",
        urutan_cluster[1]: "Prestasi Akademik Relatif Sedang",
        urutan_cluster[2]: "Prestasi Akademik Relatif Rendah"
    }

    df["Kategori"] = df["Cluster"].map(label_cluster)

    return df, label_cluster


def hitung_batas_sumbu(seri, padding_minimum):
    nilai_min = seri.min()
    nilai_max = seri.max()
    rentang = nilai_max - nilai_min
    padding = max(rentang * 0.12, padding_minimum)

    return nilai_min - padding, nilai_max + padding


def buat_grafik_ipk_nilai(df, centroid_asli):
    plt.figure(figsize=(18, 11))

    for kategori in URUTAN_KATEGORI:
        subset = df[df["Kategori"] == kategori]

        if subset.empty:
            continue

        plt.scatter(
            subset["IPK"],
            subset["Nilai_Akhir"],
            c=WARNA_KATEGORI[kategori],
            label=f"{kategori} ({len(subset)} data)",
            s=120,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.85
        )

        # Angka kecil pada titik adalah kolom No di CSV.
        # Jadi setiap titik bisa dilacak ke data asli.
        for _, row in subset.iterrows():
            plt.annotate(
                str(int(row["No"])),
                (row["IPK"], row["Nilai_Akhir"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9
            )

    for _, row in centroid_asli.iterrows():
        plt.scatter(
            row["IPK"],
            row["Nilai_Akhir"],
            marker="X",
            s=420,
            color="black",
            edgecolors="white",
            linewidths=1.5,
            zorder=10
        )

        plt.annotate(
            f"Centroid {int(row['Cluster'])}\nTerlambat {row['Jumlah_Terlambat']:.1f}",
            (row["IPK"], row["Nilai_Akhir"]),
            xytext=(10, -35),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    plt.xlim(hitung_batas_sumbu(df["IPK"], 0.15))
    plt.ylim(hitung_batas_sumbu(df["Nilai_Akhir"], 5))

    plt.title("Hasil Clustering 100 Data Mahasiswa Menggunakan K-Means")
    plt.xlabel("IPK")
    plt.ylabel("Nilai Akhir")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(FILE_GRAFIK_IPK_NILAI, dpi=300)


def buat_grafik_pca(df, x_scaled, model):
    pca = PCA(n_components=2)
    titik_pca = pca.fit_transform(x_scaled)
    centroid_pca = pca.transform(model.cluster_centers_)

    df_pca = df.copy()
    df_pca["PC1"] = titik_pca[:, 0]
    df_pca["PC2"] = titik_pca[:, 1]

    plt.figure(figsize=(18, 11))

    for kategori in URUTAN_KATEGORI:
        subset = df_pca[df_pca["Kategori"] == kategori]

        if subset.empty:
            continue

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            c=WARNA_KATEGORI[kategori],
            label=f"{kategori} ({len(subset)} data)",
            s=120,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.85
        )

        for _, row in subset.iterrows():
            plt.annotate(
                str(int(row["No"])),
                (row["PC1"], row["PC2"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9
            )

    for i in range(JUMLAH_CLUSTER):
        plt.scatter(
            centroid_pca[i, 0],
            centroid_pca[i, 1],
            marker="X",
            s=420,
            color="black",
            edgecolors="white",
            linewidths=1.5,
            zorder=10
        )

        plt.annotate(
            f"Centroid {i}",
            (centroid_pca[i, 0], centroid_pca[i, 1]),
            xytext=(10, -25),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

    variasi_pc1 = pca.explained_variance_ratio_[0] * 100
    variasi_pc2 = pca.explained_variance_ratio_[1] * 100

    plt.title("Visualisasi PCA 2D Clustering 100 Data Mahasiswa")
    plt.xlabel(f"PC1 ({variasi_pc1:.1f}% variasi data)")
    plt.ylabel(f"PC2 ({variasi_pc2:.1f}% variasi data)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(FILE_GRAFIK_PCA, dpi=300)


def main():
    df = baca_data()

    print("Data mahasiswa berhasil dibaca.")
    print(f"Jumlah data: {len(df)}")
    print("")
    print(df.head(10))

    x = df[FITUR].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(
        n_clusters=JUMLAH_CLUSTER,
        random_state=RANDOM_STATE,
        n_init=20
    )

    df["Cluster"] = model.fit_predict(x_scaled)

    df, label_cluster = beri_label_cluster(df, model)

    centroid_asli = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=FITUR
    )

    centroid_asli["Cluster"] = range(JUMLAH_CLUSTER)
    centroid_asli["Kategori"] = centroid_asli["Cluster"].map(label_cluster)

    silhouette = silhouette_score(x_scaled, df["Cluster"])

    ringkasan = (
        df
        .groupby("Kategori")
        .agg(
            Jumlah_Data=("Kategori", "size"),
            Rata_Rata_IPK=("IPK", "mean"),
            Rata_Rata_Kehadiran=("Kehadiran", "mean"),
            Rata_Rata_Nilai_Akhir=("Nilai_Akhir", "mean"),
            Rata_Rata_Jam_Belajar=("Jam_Belajar_Mingguan", "mean"),
            Rata_Rata_Terlambat=("Jumlah_Terlambat", "mean")
        )
        .reset_index()
    )

    urutan_ringkasan = {kategori: i for i, kategori in enumerate(URUTAN_KATEGORI)}
    ringkasan["Urutan"] = ringkasan["Kategori"].map(urutan_ringkasan)
    ringkasan = ringkasan.sort_values("Urutan").drop(columns=["Urutan"])

    kolom_output = [
        "No",
        "NIM",
        "Nama",
        "Program_Studi",
        "Semester",
        "Tahun_Masuk",
        "IPK",
        "Kehadiran",
        "Nilai_Tugas",
        "Nilai_UTS",
        "Nilai_UAS",
        "Nilai_Akhir",
        "Jam_Belajar_Mingguan",
        "Jumlah_Terlambat",
        "Cluster",
        "Kategori"
    ]

    if "Kelompok_Simulasi" in df.columns:
        kolom_output.append("Kelompok_Simulasi")

    df_hasil = df[kolom_output]

    df_hasil.to_csv(FILE_HASIL, index=False)
    ringkasan.to_csv(FILE_RINGKASAN, index=False)
    centroid_asli.to_csv(FILE_CENTROID, index=False)

    buat_grafik_ipk_nilai(df_hasil, centroid_asli)
    buat_grafik_pca(df_hasil, x_scaled, model)

    print("")
    print("=" * 70)
    print("HASIL CLUSTERING K-MEANS MAHASISWA")
    print("=" * 70)

    print("")
    print(f"Jumlah data: {len(df_hasil)}")
    print(f"Jumlah cluster: {JUMLAH_CLUSTER}")
    print(f"Silhouette Score: {silhouette:.3f}")

    print("")
    print("Centroid cluster dalam skala asli:")
    print(centroid_asli)

    print("")
    print("Ringkasan cluster:")
    print(ringkasan)

    print("")
    print("Jumlah data per kategori:")
    print(df_hasil["Kategori"].value_counts())

    print("")
    print("Contoh hasil lengkap:")
    print(df_hasil.head(20))

    print("")
    print("File berhasil dibuat:")
    print(f"1. {FILE_HASIL}")
    print(f"2. {FILE_RINGKASAN}")
    print(f"3. {FILE_CENTROID}")
    print(f"4. {FILE_GRAFIK_IPK_NILAI}")
    print(f"5. {FILE_GRAFIK_PCA}")

    # Bagian ini membuat grafik tampil langsung seperti Figure 1.
    if TAMPILKAN_GRAFIK:
        print("")
        print("Menampilkan grafik. Tutup window grafik untuk mengakhiri program.")
        plt.show()


if __name__ == "__main__":
    main()