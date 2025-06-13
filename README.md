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

## Dataset
Dataset yang digunakan adalah **Kermany OCT Dataset**, tersedia di:
🔗 [Kaggle - Kermany 2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)  
Dataset ini terdiri dari 4 kelas utama:
- CNV (Choroidal Neovascularization)
- DME (Diabetic Macular Edema)
- DRUSEN
- NORMAL

---

## 🔧 Alur Proyek

Notebook `training_oct.ipynb` menjalankan seluruh pipeline pelatihan model klasifikasi, mulai dari eksplorasi data (EDA), preprocessing citra OCT (denoising, konversi warna, CLAHE, resizing), augmentasi data, hingga pelatihan dua model: **baseline** dan **advanced** menggunakan arsitektur **EfficientNet-B0**.

Setelah proses pelatihan selesai, notebook ini menghasilkan tiga file:

- `baseline_model.pth` – model baseline hasil pelatihan awal.
- `advance_model.pth` – model advanced dengan preprocessing dan augmentasi lanjutan.
- `label_encoder.pkl` – encoder label untuk mengubah label kelas menjadi nilai numerik.

Ketiga file tersebut digunakan dalam script `gui_oct.py`, yang merupakan antarmuka pengguna berbasis **Streamlit** untuk melakukan prediksi citra OCT secara real-time.
