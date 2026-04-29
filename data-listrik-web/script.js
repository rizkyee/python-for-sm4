// script.js

let dayaListrik = 1000;
let pemakaianKWh = 100;

// Fungsi untuk menampilkan halaman prediksi
function goToPredictionPage() {
  // Sembunyikan Halaman Utama
  document.querySelector("#landingPage").classList.add("hidden");
  // Tampilkan Halaman Prediksi
  document.querySelector("#predictionPage").classList.remove("hidden");
}

// Fungsi untuk memperbarui Daya Listrik
function updateDayaListrik(value) {
  dayaListrik = value;
  document.querySelector("#daya-listrik-value").textContent = `${value} VA`;
}

// Fungsi untuk memperbarui Pemakaian kWh
function updatePemakaian(value) {
  pemakaianKWh = value;
  document.querySelector("#pemakaian-value").textContent = `${value} kWh`;
}

// Fungsi Prediksi
function predict() {
  let resultMessage = '';
  if (dayaListrik < 1500 && pemakaianKWh < 150) {
    resultMessage = 'Selamat, Anda Hemat Energi!';
    document.querySelector("#result-message").classList.remove('bg-red-600');
    document.querySelector("#result-message").classList.add('bg-green-600');
  } else {
    resultMessage = 'Cobalah Menghemat Energi!';
    document.querySelector("#result-message").classList.remove('bg-green-600');
    document.querySelector("#result-message").classList.add('bg-red-600');
  }

  document.querySelector("#result-message").textContent = resultMessage;
  document.querySelector("#result").classList.remove("hidden");
}