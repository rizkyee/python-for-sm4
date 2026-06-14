import os
import re
import pandas as pd

# ======================================================
# PROGRAM MEMBUAT 100 DATA PROVINSI UNTUK CLUSTERING K-MEANS
# Data dibuat dalam bentuk Provinsi-Tahun.
#
# Variabel:
# 1. IPM
# 2. Kemiskinan
# 3. TPT
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

JUMLAH_DATA_DIINGINKAN = 100

FOLDER_RAW = os.path.join(BASE_DIR, "data_raw")
os.makedirs(FOLDER_RAW, exist_ok=True)

FILE_IPM = os.path.join(FOLDER_RAW, "ipm.csv")
FILE_KEMISKINAN = os.path.join(FOLDER_RAW, "kemiskinan.csv")
FILE_TPT = os.path.join(FOLDER_RAW, "tpt.csv")
FILE_OUTPUT = os.path.join(BASE_DIR, "data_provinsi.csv")

URL_IPM = "https://data.acehprov.go.id/dataset/17461ac8-50af-475e-93cf-f7af600e9cc8/resource/de0f6d31-3d61-4bb7-8d96-445234c7758f/download/indeks-pembangunan-manusia-menurut-provinsi-di-indonesia.csv"

URL_KEMISKINAN = "https://data.acehprov.go.id/dataset/ba0a7bd9-c4b5-4348-ab46-049ad48c62d1/resource/6f31957a-c4ed-4257-8e40-b6f20446114f/download/persentase-penduduk-miskin-menurut-provinsi-di-indonesia.csv"


# ======================================================
# FUNGSI BANTUAN
# ======================================================

def baca_csv_sumber(file_lokal, url):
    """
    Membaca CSV dari file lokal jika sudah ada.
    Jika file lokal belum ada, maka membaca dari URL.
    """
    if os.path.exists(file_lokal):
        print(f"Membaca file lokal: {file_lokal}")
        return pd.read_csv(file_lokal, sep=None, engine="python")

    print(f"Membaca data dari URL: {url}")
    df = pd.read_csv(url, sep=None, engine="python")
    df.to_csv(file_lokal, index=False)
    return df


def normalisasi_nama_kolom(df):
    """
    Menyamakan format nama kolom.
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
    Mengubah nilai menjadi float.
    Aman untuk angka dengan format koma atau titik.
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
    Menyamakan nama provinsi agar data IPM, Kemiskinan, dan TPT bisa digabung.
    """
    nama = str(nama).strip()

    # Mengubah bentuk seperti SumatraUtara menjadi Sumatra Utara
    nama = re.sub(r"([a-z])([A-Z])", r"\1 \2", nama)

    nama = nama.upper()
    nama = nama.replace("PROVINSI ", "")
    nama = nama.replace(".", "")
    nama = nama.replace("-", " ")
    nama = nama.replace("_", " ")
    nama = re.sub(r"\s+", " ", nama).strip()

    # Menyamakan Sumatra menjadi Sumatera
    nama = nama.replace("SUMATRA ", "SUMATERA ")

    mapping = {
        "DKIJAKARTA": "DKI JAKARTA",
        "DKI JAKARTA": "DKI JAKARTA",

        "DIYOGYAKARTA": "DI YOGYAKARTA",
        "DI YOGYAKARTA": "DI YOGYAKARTA",
        "D I YOGYAKARTA": "DI YOGYAKARTA",

        "KEP BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
        "KEPULAUAN BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",

        "KEP RIAU": "KEPULAUAN RIAU",
        "KEPULAUAN RIAU": "KEPULAUAN RIAU",
    }

    nama_tanpa_spasi = nama.replace(" ", "")

    if nama in mapping:
        return mapping[nama]

    if nama_tanpa_spasi in mapping:
        return mapping[nama_tanpa_spasi]

    return nama


def cari_kolom(df, daftar_kemungkinan):
    """
    Mencari kolom berdasarkan beberapa kemungkinan nama.
    """
    for kolom in daftar_kemungkinan:
        if kolom in df.columns:
            return kolom

    raise ValueError(f"Kolom tidak ditemukan. Kolom tersedia: {df.columns.tolist()}")


# ======================================================
# 1. MEMBACA DATA IPM
# ======================================================

print("Membaca data IPM...")

ipm_raw = baca_csv_sumber(FILE_IPM, URL_IPM)
ipm_raw = normalisasi_nama_kolom(ipm_raw)

kolom_provinsi_ipm = cari_kolom(ipm_raw, ["bps_nama_provinsi", "nama_provinsi", "provinsi"])
kolom_tahun_ipm = cari_kolom(ipm_raw, ["tahun"])
kolom_nilai_ipm = cari_kolom(ipm_raw, ["indeks_pembangunan_manusia", "ipm"])

ipm_raw[kolom_tahun_ipm] = pd.to_numeric(ipm_raw[kolom_tahun_ipm], errors="coerce")
ipm_raw[kolom_nilai_ipm] = ipm_raw[kolom_nilai_ipm].apply(bersihkan_angka)

ipm = ipm_raw.copy()
ipm["Provinsi"] = ipm[kolom_provinsi_ipm].apply(bersihkan_nama_provinsi)
ipm["Tahun"] = ipm[kolom_tahun_ipm]

ipm = ipm[["Provinsi", "Tahun", kolom_nilai_ipm]]
ipm = ipm.rename(columns={kolom_nilai_ipm: "IPM"})


# ======================================================
# 2. MEMBACA DATA KEMISKINAN
# ======================================================

print("Membaca data kemiskinan...")

kemiskinan_raw = baca_csv_sumber(FILE_KEMISKINAN, URL_KEMISKINAN)
kemiskinan_raw = normalisasi_nama_kolom(kemiskinan_raw)

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

kemiskinan = kemiskinan_raw.copy()
kemiskinan["Provinsi"] = kemiskinan[kolom_provinsi_kemiskinan].apply(bersihkan_nama_provinsi)
kemiskinan["Tahun"] = kemiskinan[kolom_tahun_kemiskinan]

kemiskinan = kemiskinan[["Provinsi", "Tahun", kolom_nilai_kemiskinan]]
kemiskinan = kemiskinan.rename(columns={kolom_nilai_kemiskinan: "Kemiskinan"})


# ======================================================
# 3. MEMBUAT DATA TPT
# ======================================================
# Catatan:
# Data TPT yang tersedia pada proyek kamu hanya tahun 2022.
# Nilai TPT ini dipakai sebagai acuan untuk setiap provinsi.
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
tpt["Provinsi"] = tpt["Provinsi"].apply(bersihkan_nama_provinsi)
tpt.to_csv(FILE_TPT, index=False)


# ======================================================
# 4. MENGGABUNGKAN DATA
# ======================================================

print("Menggabungkan data IPM, Kemiskinan, dan TPT...")

df = ipm.merge(kemiskinan, on=["Provinsi", "Tahun"], how="inner")
df = df.merge(tpt, on="Provinsi", how="inner")

df = df.dropna()
df = df.drop_duplicates(subset=["Provinsi", "Tahun"])

# Mengambil data terbaru terlebih dahulu
df = df.sort_values(["Tahun", "Provinsi"], ascending=[False, True]).reset_index(drop=True)

# Membatasi agar jumlah data tepat 100
if len(df) >= JUMLAH_DATA_DIINGINKAN:
    df = df.head(JUMLAH_DATA_DIINGINKAN)
else:
    print("\nPERINGATAN:")
    print("Jumlah data gabungan kurang dari 100.")
    print("Data yang berhasil dibuat hanya:", len(df))

# Mengurutkan kembali agar rapi
df = df.sort_values(["Tahun", "Provinsi"]).reset_index(drop=True)

df.to_csv(FILE_OUTPUT, index=False)

print("\nFile data_provinsi.csv berhasil dibuat.")
print("Lokasi file:", FILE_OUTPUT)
print("Jumlah data:", len(df))

print("\nData akhir:")
print(df)

print("\nJumlah data per tahun:")
print(df["Tahun"].value_counts().sort_index())

print("\nDaftar kolom:")
print(df.columns.tolist())