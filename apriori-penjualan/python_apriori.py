import pandas as pd
import warnings
from mlxtend.frequent_patterns import apriori, association_rules

# Mematikan tampilan warning agar output terminal lebih bersih
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# KONFIGURASI PARAMETER
# ==========================================
NAMA_FILE_CSV = "data_penjualan_100.csv"
MIN_SUPPORT = 0.20
MIN_CONFIDENCE = 0.60

print(f"## MEMULAI PROSES APRIORI UNIVERSAL UNTUK FILE: {NAMA_FILE_CSV}")
print("=" * 80)

# ==========================================
# 1. MEMBACA FILE CSV
# ==========================================
try:
    df_raw = pd.read_csv(NAMA_FILE_CSV)
except FileNotFoundError:
    print(f"Error: File '{NAMA_FILE_CSV}' tidak ditemukan.")
    print("Pastikan file CSV berada satu folder dengan file python_apriori.py.")
    exit()

print(f"Min Support    : {MIN_SUPPORT * 100}%")
print(f"Min Confidence : {MIN_CONFIDENCE * 100}%\n")

# Mendeteksi nama kolom secara otomatis
kolom_id = df_raw.columns[0]
kolom_item = df_raw.columns[1]

print(f"Detected Kolom ID   : {kolom_id}")
print(f"Detected Kolom Item : {kolom_item}\n")

# ==========================================
# 2. TRANSFORMASI DATA MENJADI MATRIKS BINER
# ==========================================
transactions = df_raw[kolom_item].apply(
    lambda x: [item.strip() for item in str(x).split(",")]
)

all_items = sorted(list(set([
    item
    for sublist in transactions
    for item in sublist
])))

binary_data = {}

for item in all_items:
    binary_data[item] = df_raw[kolom_item].apply(
        lambda x: item in [i.strip() for i in str(x).split(",")]
    )

df_biner = pd.DataFrame(binary_data)
df_biner.index = df_raw[kolom_id]

print("### [TAHAPAN 1] MATRIKS DATA TRANSAKSI")
print("-" * 80)

df_display = df_biner.astype(int).copy()
df_display.loc["Total Kemunculan"] = df_display.sum()

print(df_display)
print("\n" + "=" * 80 + "\n")

# ==========================================
# 3. PROSES ALGORITMA APRIORI
# ==========================================
frequent_itemsets = apriori(
    df_biner,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

total_transaksi = len(df_biner)
min_muncul_kali = int(total_transaksi * MIN_SUPPORT)

frequent_itemsets["Jumlah Transaksi"] = (
    frequent_itemsets["support"] * total_transaksi
).round().astype(int)

frequent_itemsets["% Support"] = (
    frequent_itemsets["support"] * 100
).round(2).astype(str) + "%"

frequent_itemsets["Jumlah Itemset"] = frequent_itemsets["itemsets"].apply(
    lambda x: len(x)
)

print("### [TAHAPAN 2] DAFTAR KOMBINASI ITEM YANG LOLOS MINIMUM SUPPORT")
print(
    f"Batas Minimal Support: {MIN_SUPPORT * 100}% "
    f"atau minimal muncul {min_muncul_kali} kali dari {total_transaksi} transaksi"
)
print("-" * 80)

if frequent_itemsets.empty:
    print("Tidak ada kombinasi item yang memenuhi batas minimum support.")
else:
    max_item_length = frequent_itemsets["Jumlah Itemset"].max()

    for length in range(1, max_item_length + 1):
        subset = frequent_itemsets[
            frequent_itemsets["Jumlah Itemset"] == length
        ]

        if not subset.empty:
            print(f"\n-> Kombinasi {length}-Itemset:")

            display_subset = subset.copy()
            display_subset["Daftar Produk"] = display_subset["itemsets"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )

            print(
                display_subset[
                    ["Daftar Produk", "Jumlah Transaksi", "% Support"]
                ].to_string(index=False)
            )

print("\n" + "=" * 80 + "\n")

# ==========================================
# 4. PEMBENTUKAN ATURAN ASOSIASI
# ==========================================
print("### [TAHAPAN 3] ATURAN ASOSIASI ATAU STRONG RULES")
print(f"Batas Minimal Confidence: {MIN_CONFIDENCE * 100}%")
print("-" * 100)

if not frequent_itemsets.empty:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if not rules.empty:
        format_rules = []

        for idx, row in rules.iterrows():
            antecedents = ", ".join(sorted(list(row["antecedents"])))
            consequents = ", ".join(sorted(list(row["consequents"])))

            rule_text = (
                f"Jika membeli [{antecedents}] "
                f"maka akan membeli [{consequents}]"
            )

            format_rules.append({
                "Aturan Asosiasi": rule_text,
                "Support": f"{row['support'] * 100:.1f}%",
                "Confidence": f"{row['confidence'] * 100:.1f}%",
                "Lift Ratio": f"{row['lift']:.2f}",
                "raw_confidence": row["confidence"],
                "raw_lift": row["lift"]
            })

        df_rules_final = pd.DataFrame(format_rules)

        print(
            df_rules_final[
                ["Aturan Asosiasi", "Support", "Confidence", "Lift Ratio"]
            ].to_string(index=False)
        )

        print("\nCatatan:")
        print("Lift Ratio > 1 menunjukkan hubungan antar item bersifat positif.")
        print("Semakin tinggi confidence, semakin kuat peluang pembelian lanjutan.")

        # ==========================================
        # 5. KESIMPULAN STRATEGIS OTOMATIS
        # ==========================================
        print("\n" + "=" * 80)
        print("### KESIMPULAN STRATEGIS")
        print("=" * 80)

        # A. Aturan dengan confidence 100%
        pasti_rules = df_rules_final[
            df_rules_final["raw_confidence"] == 1.0
        ]

        print("A. Aturan Mutlak dengan Confidence 100%:")

        if not pasti_rules.empty:
            for i, rule in enumerate(pasti_rules["Aturan Asosiasi"], 1):
                print(f"  {i}. {rule}")

            print(
                "  Arti: Konsumen yang membeli item pertama selalu membeli "
                "item pasangannya."
            )
        else:
            print("  Tidak ditemukan aturan dengan confidence 100%.")

        # B. Rekomendasi berdasarkan lift ratio tertinggi
        print("\nB. Rekomendasi Penempatan atau Bundling Barang:")

        valid_lift_rules = df_rules_final[
            df_rules_final["raw_lift"] > 1.0
        ].sort_values(
            by="raw_lift",
            ascending=False
        )

        if not valid_lift_rules.empty:
            top_rule = valid_lift_rules.iloc[0]

            print(f"  Rekomendasi Utama: {top_rule['Aturan Asosiasi']}")
            print(f"  Nilai Lift Ratio : {top_rule['Lift Ratio']}")
            print(
                "  Strategi: Produk tersebut dapat diletakkan berdekatan "
                "atau dibuat dalam paket promosi."
            )
        else:
            print("  Tidak ditemukan hubungan item dengan lift ratio lebih dari 1.")

    else:
        print(
            "Tidak ada aturan asosiasi yang terbentuk karena tidak memenuhi "
            "minimum confidence."
        )
else:
    print(
        "Aturan asosiasi tidak dapat dihitung karena frequent itemset kosong."
    )