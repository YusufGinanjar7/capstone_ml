# Flood-Sight: Model Prediksi Banjir Multimodal

Repositori ini berisi kode dan sumber daya untuk proyek **Flood-Sight**, sebuah model pembelajaran mesin inovatif yang dirancang untuk memprediksi kejadian banjir dengan mengintegrasikan data tabular dan citra satelit. Pendekatan multimodal ini bertujuan untuk memanfaatkan beragam sumber data demi peramalan banjir yang lebih akurat dan kuat.

## Daftar Isi
- [Tentang Proyek](#tentang-proyek)
- [Fitur](#fitur)
- [Dataset](#dataset)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Arsitektur Model](#arsitektur-model)
- [Hasil](#hasil)
- [Kontribusi](#kontribusi)

## Tentang Proyek

Flood-Sight menjawab kebutuhan kritis akan prediksi banjir yang efektif, sebuah tantangan utama dalam manajemen bencana. Dengan menggabungkan data meteorologi dan geografis tradisional dengan informasi visual dari citra satelit, model ini berupaya untuk mendapatkan pemahaman yang lebih komprehensif tentang kerentanan banjir. Inti dari proyek ini adalah model deep learning multimodal yang memproses tipe data heterogen ini untuk memprediksi kejadian banjir di masa mendatang.

## Fitur

- **Integrasi Data Multimodal:** Memanfaatkan data tabular terstruktur dan citra satelit tidak terstruktur untuk prediksi.
- **Pra-pemrosesan Data Komprehensif:**
  - **Penanganan Outlier:** Menggunakan Winsorization untuk mengurangi dampak nilai ekstrem pada fitur numerik.
  - **Penskalaan Fitur:** Menstandardisasi fitur numerik menggunakan `StandardScaler` untuk mengoptimalkan kinerja model.
  - **Encoding Kategorikal:** Menggunakan One-Hot Encoding untuk `landcover_class` untuk mengubah variabel kategorikal ke format numerik yang sesuai untuk model pembelajaran mesin.
- **Arsitektur Model Tingkat Lanjut:** Mengimplementasikan model deep learning hibrida yang menggabungkan:
  - **Convolutional Neural Network (CNN)** untuk mengekstraksi fitur spasial dari citra satelit.
  - **Multilayer Perceptron (MLP)** untuk memproses fitur data tabular.
  - **Lapisan Konkatenasi** untuk menggabungkan keluaran dari cabang CNN dan MLP, memungkinkan model belajar dari informasi gabungan.
- **Prediksi Runtun Waktu:** Model dirancang untuk memprediksi kejadian banjir untuk bulan *berikutnya* dengan menggeser kolom `banjir`, memungkinkan peramalan proaktif.
- **Pelatihan & Evaluasi yang Kuat:**
  - **Callbacks:** Mengintegrasikan `EarlyStopping` untuk mencegah overfitting, `ReduceLROnPlateau` untuk penyesuaian laju pembelajaran adaptif, dan `ModelCheckpoint` untuk menyimpan model berkinerja terbaik selama pelatihan.
  - **Metrik Kinerja:** Dievaluasi menggunakan metrik klasifikasi standar termasuk akurasi, presisi, recall, F1-score, dan matriks kebingungan untuk memberikan pemahaman komprehensif tentang kinerja model.

## Dataset

Proyek ini menggunakan dataset kustom yang terdiri dari dua komponen utama:

### 1. Data Tabular
Tersimpan dalam `data_banjir_combine_final.csv`, file ini berisi berbagai fitur lingkungan dan geografis:
- `NAME_2`: Nama Kabupaten
- `NAME_3`: Nama Kecamatan
- `avg_rainfall`: Rata-rata curah hujan
- `max_rainfall`: Curah hujan maksimum
- `avg_temperature`: Rata-rata suhu
- `elevation`: Data ketinggian
- `landcover_class`: Klasifikasi tutupan lahan (misalnya, Tutupan Pohon, Daerah Terbangun)
- `ndvi`: Normalized Difference Vegetation Index
- `slope`: Kemiringan medan
- `soil_moisture`: Kandungan kelembaban tanah
- `year`: Tahun pengumpulan data
- `month`: Bulan pengumpulan data
- `banjir`: Variabel target yang menunjukkan kejadian banjir (1 untuk banjir, 0 untuk tidak banjir)
- `lat`, `long`: Koordinat Lintang dan Bujur
- `map_image`: Nama file citra satelit yang sesuai

### 2. Citra Satelit
Terletak di folder `citra_satelit`, ini adalah file `.tif` (RGB) yang merepresentasikan citra satelit untuk lokasi dan waktu tertentu, yang dihubungkan dengan data tabular melalui kolom `map_image`.

Dataset ini diproses untuk menyeimbangkan kelas target (`banjir`) menggunakan undersampling untuk memastikan pembelajaran yang adil di antara kasus banjir dan non-banjir.

## Instalasi

Untuk menyiapkan proyek secara lokal, ikuti langkah-langkah berikut:

### 1. Kloning repositori
```bash
git clone https://github.com/yusufginanjar7/capstone_ml.git
cd capstone_ml
```

### 2. Buat virtual environment (direkomendasikan)
```bash
python -m venv venv
source venv/bin/activate  # Di Windows, gunakan `venv\Scripts\activate`
```

### 3. Instal dependensi yang diperlukan
Proyek ini bergantung pada beberapa pustaka Python. Anda dapat menginstalnya menggunakan `pip`:
```bash
pip install tensorflow pandas matplotlib numpy seaborn rasterio scikit-image Pillow scikit-learn
```
*Catatan: File `requirements.txt` direkomendasikan untuk instalasi yang lebih mudah: `pip install -r requirements.txt`*

### 4. Tempatkan dataset
Pastikan folder `dataset` Anda, yang berisi `citra_satelit` dan `tabular/data_banjir_combine_final.csv`, terletak di direktori root repositori yang Anda kloning, seperti yang ditentukan oleh `BASE_DIR` dalam notebook.

## Penggunaan

Untuk menjalankan model, memproses data, dan melihat evaluasi:

### 1. Buka Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```

### 2. Jalankan sel
Jalankan semua sel di `notebook.ipynb` secara berurutan. Notebook ini akan memandu Anda melalui:
- Mengimpor pustaka yang diperlukan
- Memuat dan eksplorasi awal dataset
- Langkah-langkah pra-pemrosesan data (penanganan outlier, encoding, penskalaan)
- Membagi data menjadi set pelatihan dan pengujian
- Memuat dan pra-pemrosesan citra satelit
- Membangun dan mengkompilasi model deep learning multimodal
- Melatih model
- Mengevaluasi kinerja model
- Menyimpan model yang telah dilatih dan preprocessor

### 3. Akses model yang disimpan
Setelah eksekusi, model yang telah dilatih (`flood-sight.keras`) dan preprocessor (`preprocessor.pkl`) akan disimpan di direktori `saved_model/`.

## Arsitektur Model

Model multimodal terdiri dari dua cabang utama:

### Cabang Citra (CNN)
- Tiga blok `Conv2D` dengan 32, 64, dan 128 filter, diikuti oleh `BatchNormalization` dan `MaxPooling2D` untuk downsampling dan ekstraksi fitur
- `GlobalAveragePooling2D` untuk meratakan keluaran CNN
- Lapisan `Dense` (64 unit) dengan aktivasi `relu` dan `Dropout` untuk regularisasi

### Cabang Tabular (MLP)
- Lapisan `Input` untuk fitur tabular
- Lapisan `Dense` (64 unit) dengan aktivasi `relu`, `BatchNormalization`, dan `Dropout`

### Keluaran Gabungan
- Keluaran dari kedua cabang di `Concatenate`
- Lapisan `Dense` lainnya (64 unit) dengan aktivasi `relu` dan `Dropout`
- Lapisan `Dense` terakhir (1 unit) dengan aktivasi `sigmoid` untuk klasifikasi biner (prediksi banjir/tidak banjir)

Model dikompilasi dengan optimizer `adam` dan fungsi loss `binary_crossentropy`, melacak `accuracy` sebagai metrik.

## Hasil

Kinerja model dievaluasi pada dataset pengujian (tahun 2024). Hasil awal menunjukkan kemampuan prediksi yang tinggi:

```
Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      1341
           1       0.99      0.96      0.98       409

    accuracy                           0.99      1750
   macro avg       0.99      0.98      0.98      1750
weighted avg       0.99      0.99      0.99      1750
```

Matriks kebingungan lebih lanjut menggambarkan kemampuan model untuk mengklasifikasikan dengan benar baik kejadian banjir maupun non-banjir, dengan sangat sedikit kesalahan klasifikasi.

## Kontribusi

Kontribusi sangat kami sambut! Jangan ragu untuk melakukan fork repositori, membuka *issue*, atau mengirimkan *pull request*.
