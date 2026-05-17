import os
import re
import pandas as pd

# ======================================================
# PROGRAM MEMBUAT DATA PROVINSI UNTUK CLUSTERING K-MEANS
# Variabel:
# 1. IPM
# 2. Kemiskinan
# 3. TPT
# ======================================================

TAHUN_ANALISIS = 2022

FOLDER_RAW = "data_raw"
os.makedirs(FOLDER_RAW, exist_ok=True)

# Sumber data IPM dari OpenData Aceh
URL_IPM = "https://data.acehprov.go.id/dataset/17461ac8-50af-475e-93cf-f7af600e9cc8/resource/de0f6d31-3d61-4bb7-8d96-445234c7758f/download/indeks-pembangunan-manusia-menurut-provinsi-di-indonesia.csv"

# Sumber data persentase penduduk miskin dari OpenData Aceh
URL_KEMISKINAN = "https://data.acehprov.go.id/dataset/ba0a7bd9-c4b5-4348-ab46-049ad48c62d1/resource/6f31957a-c4ed-4257-8e40-b6f20446114f/download/persentase-penduduk-miskin-menurut-provinsi-di-indonesia.csv"


# ======================================================
# FUNGSI BANTUAN
# ======================================================

def baca_csv_sumber(sumber):
    """
    Membaca CSV dari URL atau file lokal.
    sep=None digunakan agar pandas mendeteksi pemisah koma atau titik koma secara otomatis.
    """
    return pd.read_csv(sumber, sep=None, engine="python")


def normalisasi_nama_kolom(df):
    """
    Mengubah nama kolom agar seragam.
    Contoh:
    BPS Nama Provinsi -> bps_nama_provinsi
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def bersihkan_angka(nilai):
    """
    Mengubah angka menjadi float.
    Aman untuk format angka Indonesia yang memakai koma.
    """
    if pd.isna(nilai):
        return None

    nilai = str(nilai).strip()
    nilai = re.sub(r"[^0-9,.-]", "", nilai)

    if "," in nilai and "." not in nilai:
        nilai = nilai.replace(",", ".")
    elif "," in nilai and "." in nilai:
        nilai = nilai.replace(".", "").replace(",", ".")

    try:
        return float(nilai)
    except ValueError:
        return None


def bersihkan_nama_provinsi(nama):
    """
    Menyamakan penulisan nama provinsi agar bisa digabung.
    """
    nama = str(nama).upper().strip()
    nama = nama.replace("PROVINSI ", "")
    nama = nama.replace("DKI JAKARTA", "DKI JAKARTA")
    nama = nama.replace("DI YOGYAKARTA", "DI YOGYAKARTA")
    nama = nama.replace("D I YOGYAKARTA", "DI YOGYAKARTA")
    nama = nama.replace("KEP. BANGKA BELITUNG", "KEPULAUAN BANGKA BELITUNG")
    nama = nama.replace("KEP. RIAU", "KEPULAUAN RIAU")
    return nama


def cari_kolom(df, daftar_kemungkinan):
    """
    Mencari nama kolom berdasarkan beberapa kemungkinan nama.
    """
    for kolom in daftar_kemungkinan:
        if kolom in df.columns:
            return kolom

    raise ValueError(f"Kolom tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")


# ======================================================
# 1. MEMBUAT FILE ipm.csv
# ======================================================

print("Mengunduh dan membaca data IPM...")

ipm_raw = baca_csv_sumber(URL_IPM)
ipm_raw = normalisasi_nama_kolom(ipm_raw)

ipm_raw.to_csv(f"{FOLDER_RAW}/ipm.csv", index=False)

kolom_provinsi_ipm = cari_kolom(ipm_raw, ["bps_nama_provinsi", "nama_provinsi", "provinsi"])
kolom_tahun_ipm = cari_kolom(ipm_raw, ["tahun"])
kolom_nilai_ipm = cari_kolom(ipm_raw, ["indeks_pembangunan_manusia", "ipm"])

ipm_raw[kolom_tahun_ipm] = pd.to_numeric(ipm_raw[kolom_tahun_ipm], errors="coerce")
ipm_raw[kolom_nilai_ipm] = ipm_raw[kolom_nilai_ipm].apply(bersihkan_angka)

ipm = ipm_raw[ipm_raw[kolom_tahun_ipm] == TAHUN_ANALISIS].copy()
ipm["Provinsi"] = ipm[kolom_provinsi_ipm].apply(bersihkan_nama_provinsi)

ipm = ipm[["Provinsi", kolom_nilai_ipm]]
ipm = ipm.rename(columns={kolom_nilai_ipm: "IPM"})


# ======================================================
# 2. MEMBUAT FILE kemiskinan.csv
# ======================================================

print("Mengunduh dan membaca data kemiskinan...")

kemiskinan_raw = baca_csv_sumber(URL_KEMISKINAN)
kemiskinan_raw = normalisasi_nama_kolom(kemiskinan_raw)

kemiskinan_raw.to_csv(f"{FOLDER_RAW}/kemiskinan.csv", index=False)

kolom_provinsi_kemiskinan = cari_kolom(
    kemiskinan_raw,
    ["bps_nama_provinsi", "nama_provinsi", "provinsi"]
)

kolom_tahun_kemiskinan = cari_kolom(kemiskinan_raw, ["tahun"])

kolom_nilai_kemiskinan = cari_kolom(
    kemiskinan_raw,
    ["persentase", "persentase_penduduk_miskin", "kemiskinan"]
)

kemiskinan_raw[kolom_tahun_kemiskinan] = pd.to_numeric(
    kemiskinan_raw[kolom_tahun_kemiskinan],
    errors="coerce"
)

kemiskinan_raw[kolom_nilai_kemiskinan] = kemiskinan_raw[kolom_nilai_kemiskinan].apply(bersihkan_angka)

kemiskinan = kemiskinan_raw[kemiskinan_raw[kolom_tahun_kemiskinan] == TAHUN_ANALISIS].copy()
kemiskinan["Provinsi"] = kemiskinan[kolom_provinsi_kemiskinan].apply(bersihkan_nama_provinsi)

kemiskinan = kemiskinan[["Provinsi", kolom_nilai_kemiskinan]]
kemiskinan = kemiskinan.rename(columns={kolom_nilai_kemiskinan: "Kemiskinan"})


# ======================================================
# 3. MEMBUAT FILE tpt.csv
# Data TPT 2022 dibuat manual berdasarkan tabel Bappenas/SIMREG
# ======================================================

print("Membuat data TPT...")

data_tpt_2022 = [
    ["ACEH", 6.17],
    ["SUMATERA UTARA", 6.16],
    ["SUMATERA BARAT", 6.28],
    ["RIAU", 4.37],
    ["JAMBI", 4.59],
    ["SUMATERA SELATAN", 4.63],
    ["BENGKULU", 3.59],
    ["LAMPUNG", 4.52],
    ["KEPULAUAN BANGKA BELITUNG", 4.77],
    ["KEPULAUAN RIAU", 8.23],
    ["DKI JAKARTA", 7.18],
    ["JAWA BARAT", 8.31],
    ["JAWA TENGAH", 5.57],
    ["DI YOGYAKARTA", 4.06],
    ["JAWA TIMUR", 5.49],
    ["BANTEN", 8.09],
    ["BALI", 4.80],
    ["NUSA TENGGARA BARAT", 2.89],
    ["NUSA TENGGARA TIMUR", 3.54],
    ["KALIMANTAN BARAT", 5.11],
    ["KALIMANTAN TENGAH", 4.26],
    ["KALIMANTAN SELATAN", 4.74],
    ["KALIMANTAN TIMUR", 5.71],
    ["KALIMANTAN UTARA", 4.33],
    ["SULAWESI UTARA", 6.61],
    ["SULAWESI TENGAH", 3.00],
    ["SULAWESI SELATAN", 4.51],
    ["SULAWESI TENGGARA", 3.36],
    ["GORONTALO", 2.58],
    ["SULAWESI BARAT", 2.34],
    ["MALUKU", 6.88],
    ["MALUKU UTARA", 3.98],
    ["PAPUA BARAT", 5.37],
    ["PAPUA", 2.83],
]

tpt = pd.DataFrame(data_tpt_2022, columns=["Provinsi", "TPT"])
tpt.to_csv(f"{FOLDER_RAW}/tpt.csv", index=False)


# ======================================================
# 4. MENGGABUNGKAN DATA
# ======================================================

print("Menggabungkan data IPM, Kemiskinan, dan TPT...")

df = ipm.merge(kemiskinan, on="Provinsi", how="inner")
df = df.merge(tpt, on="Provinsi", how="inner")

df = df.dropna()
df = df.sort_values("Provinsi").reset_index(drop=True)

df.to_csv("data_provinsi.csv", index=False)

print("\nFile data_provinsi.csv berhasil dibuat.")
print("Jumlah data:", len(df))
print("\nContoh data:")
print(df.head())

print("\nDaftar kolom:")
print(df.columns.tolist())