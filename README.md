# Klasifikasi Sendok vs Garpu (ResNet-18)
Fine-tuning ResNet-18 (pretrained ImageNet) untuk mengklasifikasi gambar sendok dan garpu, serta pengukuran latency.

## Setup
- Dataset: 347 gambar (train 278, val 35, test 34)
- Input 224x224, Adam (lr 1e-4), 20 epoch
- Model terbaik diambil dari epoch 7 (val loss terendah)
- Perangkat: CPU

## Hasil
- Akurasi test: 94.12% (32 dari 34 benar)
- Precision / recall / F1:
  - garpu: 1.00 / 0.89 / 0.94
  - sendok: 0.89 / 1.00 / 0.94
- Latency CPU (batch 1): 31.74 ms rata-rata, p95 33.59 ms, p99 35.59 ms
- Kecepatan: 31.5 FPS

## Analisis
- Ada 2 kesalahan, keduanya garpu yang terbaca sebagai sendok.
- Akurasi train hampir 100% sedangkan val loss naik-turun, tanda mulai overfitting. Karena itu dipakai model dari epoch 7.
- Data val dan test kecil, jadi akurasi punya ketidakpastian cukup besar.
- Latency stabil, tetapi p99 sudah melewati batas kamera 30 FPS (sekitar 33 ms), jadi untuk real-time masih mepet.

## Menjalankan
```bash
python3 train_resnet18_colab_style.py --data_dir dataset --epochs 20 --patience 20
```

Output: `resnet18_sendok_garpu.pth`, `kurva_loss.png`, `kurva_akurasi.png`, `confusion_matrix.png`, `latency.png`