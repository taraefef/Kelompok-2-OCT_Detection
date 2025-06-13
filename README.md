#  OCT Retina Disease Classifier

### 📚 Mata Kuliah: Pengolahan Sinyal Medis  
### 👨‍💻 Proyek Kelompok 2  
**Anggota:**
- Jovano Angelo Apriarto – `2206060561`
- Kattya Aulia Faharani – `2206030382`
- Tenaya Shafa Kirana – `2206031555`
- Tara Nur Amalina – `2206032261`

---

## 📝 Deskripsi Proyek

Proyek ini bertujuan untuk membangun sistem klasifikasi citra **OCT (Optical Coherence Tomography)** retina untuk mendeteksi penyakit mata seperti **CNV**, **DME**, **DRUSEN**, dan **NORMAL** menggunakan deep learning dengan arsitektur **EfficientNet-B0**, serta disajikan dalam bentuk aplikasi web interaktif menggunakan **Streamlit**.

---

## 🧪 Metodologi

### A. Data Acquisition
Dataset yang digunakan adalah **Kermany OCT Dataset**, tersedia di:
🔗 [Kaggle - Kermany 2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)  
Dataset ini terdiri dari 4 kelas utama:
- CNV (Choroidal Neovascularization)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

### B. Exploratory Data Analysis (EDA)
Analisis dilakukan untuk memahami distribusi label dan karakteristik visual dari masing-masing kelas:
- Menggunakan **seaborn countplot** untuk mengetahui distribusi jumlah data pada tiap kelas.
- Menampilkan contoh gambar dari tiap kelas untuk memahami kompleksitas intra-kelas dan variasi antar-kelas.

### C. Preprocessing
Citra akan diresize menjadi 224x224 pixel dan dinormalisasi menggunakan mean dan standard deviation dari ImageNet:
```python
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
