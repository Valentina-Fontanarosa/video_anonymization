# video_anonymization
Deep Learning project per anonimizzare parti sensibili nei video

---

## 📁 Download necessari

### 📥 Tokenizer

Scarica i seguenti file da Hugging Face e salvali nella cartella `data/`:

- [`vocab.json`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json)
- [`merges.txt`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt)

> 📌 URL di origine: [https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer)

---

### 📥 Stable Diffusion Weights

Scarica il checkpoint principale da:

- [`v1-5-pruned-emaonly.ckpt`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt)

Salvalo nella cartella `data/`.

> 📌 URL di origine: [https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)

---

### ✅ Modelli fine-tuned testati

Puoi anche utilizzare modelli fine-tuned basati su Stable Diffusion v1.5. Ad esempio:

- [`v1-5-pruned-emaonly-fp16.safetensors`](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors)

Salvalo nella cartella `data/`.

> 📌 URL di origine: [https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/main/v1-5-pruned-emaonly-fp16.safetensors](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/blob/main/v1-5-pruned-emaonly-fp16.safetensors)

---

## 📦 YOLOv8-seg (Segmentazione)

Per usare la segmentazione, è necessario scaricare YOLOv8 con supporto a segmentazione (`yolov8-seg`)
