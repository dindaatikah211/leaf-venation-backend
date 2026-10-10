# VenaLeaf Backend

REST API untuk mengklasifikasikan pola venasi daun ke empat kelas: arcuate, palmate, parallel, dan pinnate. Dipakai oleh frontend VenaLeaf di [vena-leaf.vercel.app](https://vena-leaf.vercel.app).

## Teknologi

Flask, Gunicorn, TensorFlow 2.21.0, Pillow, NumPy. Model: MobileNetV2 dan EfficientNetB0, plus model ringan untuk memeriksa apakah gambar berisi daun.

## Menjalankan Lokal

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Server berjalan di `http://localhost:5000`.

## Endpoint

| Method | Path       | Keterangan                                                        |
| ------ | ---------- | ----------------------------------------------------------------- |
| GET    | `/health`  | Status layanan                                                    |
| POST   | `/predict` | Kirim gambar (maks. 5 MB) sebagai form-data dengan field `file`   |

```bash
curl -F "file=@foto-daun.jpg" http://localhost:5000/predict
```

Respons sukses:

```json
{ "label": "pinnate", "confidence": 0.97, "probabilities": { "arcuate": 0.01, "palmate": 0.01, "parallel": 0.01, "pinnate": 0.97 } }
```

Gambar yang tidak dikenali sebagai daun dibalas status `422`.

## Variabel Lingkungan

| Variabel         | Default                 | Keterangan                                                  |
| ---------------- | ----------------------- | ----------------------------------------------------------- |
| `MODEL_TYPE`     | `efficientnet`          | `efficientnet` atau `mobilenet`                             |
| `LEAF_THRESHOLD` | `0.5`                   | Batas skor daun, turunkan jika foto daun asli ditolak       |
| `CORS_ORIGINS`   | `http://localhost:3000` | Origin yang diizinkan, dipisah koma tanpa spasi             |

## Deploy

Di-deploy ke Railway lewat `Dockerfile`. Isi `CORS_ORIGINS` dengan domain frontend, buat domain publik dengan port `8080`, lalu isi `NEXT_PUBLIC_API_URL` di frontend dengan domain tersebut. File model harus ikut ter-commit (`git add -f model/*.h5`).