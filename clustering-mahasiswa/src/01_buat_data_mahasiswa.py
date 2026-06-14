from pathlib import Path
import numpy as np
import pandas as pd


# ======================================================
# MEMBUAT DATA MAHASISWA SIMULASI UNTUK CLUSTERING
#
# Data ini fiktif, tetapi dibuat realistis dan konsisten.
# Setiap baris punya identitas mahasiswa yang jelas.
#
# Variabel utama:
# 1. IPK
# 2. Kehadiran
# 3. Nilai Tugas
# 4. Nilai UTS
# 5. Nilai UAS
# 6. Jam Belajar Mingguan
# 7. Jumlah Terlambat
# ======================================================


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "output"

DATA_RAW_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_OUTPUT = DATA_RAW_DIR / "data_mahasiswa.csv"

JUMLAH_DATA = 100
RANDOM_SEED = 42


NAMA_DEPAN = [
    "Ahmad", "Budi", "Citra", "Dewi", "Eka", "Fajar", "Gita", "Hadi", "Intan", "Joko",
    "Kirana", "Lukman", "Maya", "Nanda", "Oki", "Putri", "Rafi", "Salsa", "Tegar", "Ulfa",
    "Vina", "Wahyu", "Yusuf", "Zahra", "Rizky", "Dimas", "Nabila", "Aulia", "Farhan", "Sinta",
    "Rama", "Anisa", "Ilham", "Mega", "Bayu", "Fitri", "Galih", "Nia", "Reza", "Tania",
    "Rendi", "Dinda", "Ari", "Laras", "Bagus", "Tiara", "Yoga", "Niken", "Arif", "Mila"
]

NAMA_BELAKANG = [
    "Pratama", "Saputra", "Lestari", "Wulandari", "Santoso", "Permana", "Ramadhan", "Maulana",
    "Nugroho", "Anggraini", "Kusuma", "Wijaya", "Hidayat", "Febriani", "Firmansyah", "Sari",
    "Setiawan", "Purnama", "Hakim", "Safitri", "Kurniawan", "Utami", "Fauzi", "Amalia",
    "Syahputra", "Rahmawati", "Gunawan", "Aprilia", "Saputri", "Hermawan"
]

PROGRAM_STUDI = {
    "Teknik Informatika": "TI",
    "Sistem Informasi": "SI",
    "Manajemen": "MJ",
    "Akuntansi": "AK",
    "Ilmu Komunikasi": "IK"
}


def batasi(nilai, minimum, maksimum):
    return float(np.clip(nilai, minimum, maksimum))


def buat_nim(tahun_masuk, kode_prodi, nomor_urut):
    return f"{tahun_masuk}{kode_prodi}{nomor_urut:04d}"


def pilih_kelompok_awal(rng):
    peluang = rng.random()

    if peluang < 0.35:
        return "Tinggi"

    if peluang < 0.80:
        return "Sedang"

    return "Rendah"


def buat_nilai_berdasarkan_kelompok(rng, kelompok):
    if kelompok == "Tinggi":
        ipk = batasi(rng.normal(3.55, 0.18), 3.05, 4.00)
        kehadiran = batasi(rng.normal(91, 5), 78, 100)
        nilai_tugas = batasi(rng.normal(86, 6), 72, 100)
        nilai_uts = batasi(rng.normal(84, 7), 70, 100)
        nilai_uas = batasi(rng.normal(86, 7), 72, 100)
        jam_belajar = batasi(rng.normal(16, 4), 8, 28)
        terlambat = int(round(batasi(rng.normal(1.5, 1.2), 0, 5)))

    elif kelompok == "Sedang":
        ipk = batasi(rng.normal(2.85, 0.25), 2.20, 3.35)
        kehadiran = batasi(rng.normal(78, 7), 60, 92)
        nilai_tugas = batasi(rng.normal(73, 7), 55, 88)
        nilai_uts = batasi(rng.normal(71, 8), 52, 88)
        nilai_uas = batasi(rng.normal(72, 8), 52, 90)
        jam_belajar = batasi(rng.normal(9, 3), 3, 16)
        terlambat = int(round(batasi(rng.normal(5, 2), 1, 10)))

    else:
        ipk = batasi(rng.normal(2.15, 0.30), 1.20, 2.75)
        kehadiran = batasi(rng.normal(63, 9), 40, 80)
        nilai_tugas = batasi(rng.normal(58, 8), 35, 75)
        nilai_uts = batasi(rng.normal(55, 9), 30, 74)
        nilai_uas = batasi(rng.normal(57, 9), 30, 76)
        jam_belajar = batasi(rng.normal(5, 2.5), 1, 11)
        terlambat = int(round(batasi(rng.normal(10, 3), 4, 18)))

    nilai_akhir = (
        0.25 * nilai_tugas
        + 0.30 * nilai_uts
        + 0.35 * nilai_uas
        + 0.10 * kehadiran
    )

    return {
        "IPK": round(ipk, 2),
        "Kehadiran": round(kehadiran, 2),
        "Nilai_Tugas": round(nilai_tugas, 2),
        "Nilai_UTS": round(nilai_uts, 2),
        "Nilai_UAS": round(nilai_uas, 2),
        "Nilai_Akhir": round(nilai_akhir, 2),
        "Jam_Belajar_Mingguan": round(jam_belajar, 2),
        "Jumlah_Terlambat": terlambat
    }


def buat_data_mahasiswa():
    rng = np.random.default_rng(RANDOM_SEED)

    data = []
    kombinasi_nama = []

    for nama_depan in NAMA_DEPAN:
        for nama_belakang in NAMA_BELAKANG:
            kombinasi_nama.append(f"{nama_depan} {nama_belakang}")

    rng.shuffle(kombinasi_nama)

    daftar_prodi = list(PROGRAM_STUDI.keys())

    for i in range(1, JUMLAH_DATA + 1):
        prodi = rng.choice(daftar_prodi)
        kode_prodi = PROGRAM_STUDI[prodi]

        tahun_masuk = int(rng.choice([2021, 2022, 2023, 2024]))
        semester = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8]))

        nim = buat_nim(tahun_masuk, kode_prodi, i)
        nama = kombinasi_nama[i - 1]

        kelompok_awal = pilih_kelompok_awal(rng)
        nilai = buat_nilai_berdasarkan_kelompok(rng, kelompok_awal)

        data.append({
            "No": i,
            "NIM": nim,
            "Nama": nama,
            "Program_Studi": prodi,
            "Semester": semester,
            "Tahun_Masuk": tahun_masuk,
            **nilai,
            "Kelompok_Simulasi": kelompok_awal
        })

    df = pd.DataFrame(data)

    urutan_kolom = [
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
        "Kelompok_Simulasi"
    ]

    df = df[urutan_kolom]

    return df


def validasi_data(df):
    if len(df) != JUMLAH_DATA:
        raise ValueError(f"Jumlah data harus {JUMLAH_DATA}, tetapi data sekarang {len(df)}.")

    if df["NIM"].duplicated().any():
        raise ValueError("Ada NIM yang duplikat.")

    kolom_numerik = [
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

    if df[kolom_numerik].isna().any().any():
        raise ValueError("Ada data numerik yang kosong.")

    if not df["IPK"].between(0, 4).all():
        raise ValueError("Ada IPK di luar rentang 0 sampai 4.")

    for kolom in ["Kehadiran", "Nilai_Tugas", "Nilai_UTS", "Nilai_UAS", "Nilai_Akhir"]:
        if not df[kolom].between(0, 100).all():
            raise ValueError(f"Ada nilai {kolom} di luar rentang 0 sampai 100.")


def main():
    df = buat_data_mahasiswa()
    validasi_data(df)

    df.to_csv(FILE_OUTPUT, index=False)

    print("Data mahasiswa berhasil dibuat.")
    print(f"Lokasi file: {FILE_OUTPUT}")
    print(f"Jumlah data: {len(df)}")

    print("")
    print("Contoh 10 data pertama:")
    print(df.head(10))

    print("")
    print("Jumlah data berdasarkan kelompok simulasi:")
    print(df["Kelompok_Simulasi"].value_counts())


if __name__ == "__main__":
    main()