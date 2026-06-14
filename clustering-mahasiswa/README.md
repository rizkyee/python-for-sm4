# Clustering Prestasi Akademik Mahasiswa Menggunakan K-Means

Project ini mengelompokkan 100 data mahasiswa berdasarkan performa akademik.

Data yang digunakan adalah data simulasi realistis. Setiap mahasiswa memiliki NIM, nama, program studi, semester, IPK, kehadiran, nilai tugas, nilai UTS, nilai UAS, jam belajar mingguan, dan jumlah terlambat.

## Fitur Clustering

1. IPK
2. Kehadiran
3. Nilai Akhir
4. Jam Belajar Mingguan
5. Jumlah Terlambat

## Kategori Hasil

1. Prestasi Akademik Relatif Tinggi
2. Prestasi Akademik Relatif Sedang
3. Prestasi Akademik Relatif Rendah

## Cara Menjalankan

```bash
pip install -r requirements.txt
python src/01_buat_data_mahasiswa.py
python src/02_kmeans_mahasiswa.py

## tentang project
Judul:

Implementasi Algoritma K-Means untuk Clustering Prestasi Akademik Mahasiswa Berdasarkan Data Akademik Simulasi

Project ini menerapkan algoritma K-Means untuk mengelompokkan 100 data mahasiswa ke dalam tiga kategori prestasi akademik, yaitu Prestasi Akademik Relatif Tinggi, Prestasi Akademik Relatif Sedang, dan Prestasi Akademik Relatif Rendah. Data yang digunakan merupakan data simulasi realistis yang dibuat menggunakan Python. Data tersebut memuat identitas mahasiswa seperti NIM, nama, program studi, semester, dan tahun masuk, serta data akademik seperti IPK, kehadiran, nilai tugas, nilai UTS, nilai UAS, nilai akhir, jam belajar mingguan, dan jumlah keterlambatan.

Clustering dilakukan berdasarkan lima variabel utama, yaitu IPK, kehadiran, nilai akhir, jam belajar mingguan, dan jumlah terlambat. Sebelum proses clustering, data numerik distandarisasi menggunakan StandardScaler agar setiap variabel memiliki skala yang seimbang. Setelah itu, algoritma K-Means membagi data mahasiswa menjadi tiga cluster berdasarkan kemiripan pola akademik.

Hasil clustering menunjukkan bahwa mahasiswa dapat dikelompokkan berdasarkan tingkat performa akademik. Pada hasil yang tampil, terdapat 37 mahasiswa dalam kategori Prestasi Akademik Relatif Tinggi, 45 mahasiswa dalam kategori Prestasi Akademik Relatif Sedang, dan 18 mahasiswa dalam kategori Prestasi Akademik Relatif Rendah. Kelompok prestasi tinggi cenderung memiliki IPK dan nilai akhir tinggi, kehadiran baik, jam belajar lebih tinggi, serta jumlah keterlambatan lebih rendah. Kelompok prestasi rendah cenderung memiliki IPK dan nilai akhir lebih rendah, serta jumlah keterlambatan lebih tinggi.

Deskripsi Figure 1:

Grafik ini menampilkan hasil clustering mahasiswa berdasarkan IPK dan Nilai Akhir. Sumbu X menunjukkan IPK, sedangkan sumbu Y menunjukkan Nilai Akhir. Setiap titik mewakili satu mahasiswa, dan angka di dekat titik menunjukkan nomor data mahasiswa pada file CSV. Warna hijau menunjukkan mahasiswa dengan prestasi akademik relatif tinggi, warna kuning menunjukkan prestasi akademik relatif sedang, dan warna merah menunjukkan prestasi akademik relatif rendah. Tanda X hitam menunjukkan centroid, yaitu titik pusat dari masing-masing cluster.



Deskripsi Figure 2:

Grafik ini menampilkan visualisasi hasil clustering menggunakan metode PCA 2D. PCA digunakan untuk mereduksi lima variabel clustering menjadi dua komponen utama, yaitu PC1 dan PC2. Grafik ini membantu melihat pemisahan cluster berdasarkan gabungan seluruh variabel akademik, bukan hanya IPK dan Nilai Akhir. Pada grafik, cluster hijau berada di sisi kanan, cluster kuning berada di bagian tengah, dan cluster merah berada di sisi kiri. Pola tersebut menunjukkan bahwa data mahasiswa berhasil dipisahkan menjadi tiga kelompok performa akademik yang berbeda.

Kalimat singkat untuk laporan:

Penelitian ini bertujuan untuk mengelompokkan mahasiswa berdasarkan performa akademik menggunakan algoritma K-Means. Data yang digunakan merupakan data simulasi realistis sebanyak 100 mahasiswa dengan variabel IPK, kehadiran, nilai akhir, jam belajar mingguan, dan jumlah keterlambatan. Hasil clustering membagi mahasiswa ke dalam tiga kategori, yaitu prestasi akademik relatif tinggi, sedang, dan rendah. Hasil ini dapat digunakan sebagai contoh sistem analisis awal untuk membantu mengidentifikasi mahasiswa yang membutuhkan pendampingan akademik.