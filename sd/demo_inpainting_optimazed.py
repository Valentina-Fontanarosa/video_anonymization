import os
import time
import torch
import torch._dynamo  # importa torch._dynamo per gestire errori
torch._dynamo.config.suppress_errors = True
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import cProfile
import pstats
from io import StringIO
from ultralytics import YOLO
from transformers import CLIPTokenizer

import model_loader
import pipeline_optimized as pipeline

# === Config ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"
#USE_FP16 = False
USE_TORCH_COMPILE = DEVICE == "cuda" and hasattr(torch, 'compile')
COMPILE_MODE = "reduce-overhead"
N_INFERENCE_STEPS = 20
BATCH_SIZE_INPAINT = 4
USE_TENSORRT = True

os.chdir('sd')
print(os.getcwd())
print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Using FP16 precision: {USE_FP16}")
print(f"[INFO] Using torch.compile: {USE_TORCH_COMPILE} (mode: {COMPILE_MODE})")
print(f"[INFO] Inpainting steps: {N_INFERENCE_STEPS}, batch size: {BATCH_SIZE_INPAINT}")

# === Paths ===
IMG_PATH = "../images/frame_10500ms.jpg"
OUT_PATH = "../images/output_custom_inpainting_optimized_2.jpg"
CSV_PATH = "../dataFrame/tempi_inpainting_custom_optimized_3.csv"
MODEL_DATA_DIR = "../data"
YOLO_MODEL_NAME = "yolov8x-seg.pt"
models_name = {
                "INPAINTING_MODEL_NAME": "v1-5-pruned-emaonly-fp16.ckpt",
                "TENSORTRT_MODEL_NAME": "diffusion.engine"
              }

REAL_CROP_DIR = "../images/original_crops"
GEN_CROP_DIR = "../images/generated_crops"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

os.makedirs(REAL_CROP_DIR, exist_ok=True)
os.makedirs(GEN_CROP_DIR, exist_ok=True)

# === Profilazione globale ===
pr = cProfile.Profile()
pr.enable()
start_total = time.time()

# === Load models ===
print("[INFO] Loading models...")
tokenizer = CLIPTokenizer(
    os.path.join(MODEL_DATA_DIR, "vocab.json"),
    merges_file=os.path.join(MODEL_DATA_DIR, "merges.txt")
)

models_file = {
                "INPAINTING_MODEL_NAME": os.path.join(MODEL_DATA_DIR, models_name["INPAINTING_MODEL_NAME"]),
                "TENSORTRT_MODEL_NAME": os.path.join(MODEL_DATA_DIR, models_name["TENSORTRT_MODEL_NAME"])
              }

if not os.path.exists(models_file["INPAINTING_MODEL_NAME"]):
    raise FileNotFoundError(f"Model checkpoint not found at {models_file['INPAINTING_MODEL_NAME']}")

if USE_TENSORRT and not os.path.exists(models_file["TENSORTRT_MODEL_NAME"]):
    raise FileNotFoundError(f"Model checkpoint not found at {models_name['TENSORTRT_MODEL_NAME']}")

print(models_file['TENSORTRT_MODEL_NAME'])
models = model_loader.preload_models_from_standard_weights(models_file, DEVICE, USE_TENSORRT)

for model_name, model in models.items():
    if model is None:
        continue

    use_trt_for_this = USE_TENSORRT and model_name == "diffusion"
    if not use_trt_for_this:
        model.to(DEVICE)
        if USE_FP16:
            model = model.half()

        if USE_TORCH_COMPILE and model_name in ["encoder", "decoder", "diffusion"]:
            try:
                print(f"[INFO] Compiling {model_name} with torch.compile...")
                model = torch.compile(model, mode=COMPILE_MODE)
            except Exception as e:
                print(f"[WARNING] Failed to compile {model_name}, using eager mode instead. Error:\n{e}")
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True  # fallback automatico
    else:
        print(f"[INFO] Using TensorRT engine for {model_name}, skipping PyTorch compile.")

    models[model_name] = model


# === Load image ===
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found at {IMG_PATH}")
original_img = Image.open(IMG_PATH).convert("RGB")
img_np = np.array(original_img)
img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
img_height, img_width = img_np.shape[:2]

# === YOLOv8 segmentation ===
print("[INFO] Running YOLOv8 segmentation...")
if not os.path.exists(YOLO_MODEL_NAME):
    raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_NAME}")
yolo_model = YOLO(YOLO_MODEL_NAME)
start_yolo = time.time()
results = yolo_model(img_cv, task="segment", verbose=False)
yolo_time = time.time() - start_yolo

final_img_np = img_np.copy().astype(np.uint8)
crop_counter = 0
inpaint_total_time = 0
all_crops_data = []

# === Crop and Mask Extraction ===
start_crop = time.time()
for result in results:
    if result.masks is None:
        continue

    boxes = result.boxes
    masks_yolo = result.masks

    for i, cls_id in enumerate(boxes.cls):
        if int(cls_id) != 0:
            continue

        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)
        box_w, box_h = x2 - x1, y2 - y1

        if box_w < 30 or box_h < 30:
            continue

        mask_raw = masks_yolo.data[i].cpu().numpy().astype(np.uint8)
        mask_resized_full = cv2.resize(mask_raw, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(mask_resized_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled_full_for_crop = np.zeros_like(mask_resized_full)
        cv2.drawContours(mask_filled_full_for_crop, contours, -1, 255, thickness=cv2.FILLED)

        mask_crop_np = mask_filled_full_for_crop[y1:y2, x1:x2]
        if mask_crop_np.sum() / 255 < 200:
            continue

        crop_img_pil = original_img.crop((x1, y1, x2, y2))

        crop_index = len(all_crops_data)
        real_crop_path = os.path.join(REAL_CROP_DIR, f"real_crop_{crop_index:03}.png")
        crop_img_pil.save(real_crop_path)

        crop_mask_pil = Image.fromarray(mask_crop_np).convert("L")

        resized_img_pil = crop_img_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
        resized_mask_pil = crop_mask_pil.resize((512, 512), resample=Image.Resampling.NEAREST)

        prompt = "realistic person, same pose, same clothes, full body, high quality"
        negative_prompt = "cartoon, anime, distorted, glitch, unrealistic, deformed"

        all_crops_data.append({
            'input_image': resized_img_pil,
            'input_mask': resized_mask_pil,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'coords': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'box_w': box_w, 'box_h': box_h},
            'original_crop_pil': crop_img_pil,
            'blend_mask_pil': Image.fromarray(mask_crop_np).convert("L")
        })
crop_time = time.time() - start_crop

# === Inpainting ===
if all_crops_data:
    num_batches = (len(all_crops_data) + BATCH_SIZE_INPAINT - 1) // BATCH_SIZE_INPAINT
    print(f"[INFO] Processing {len(all_crops_data)} people in {num_batches} batches...")

    for batch_num in range(num_batches):
        batch_start_idx = batch_num * BATCH_SIZE_INPAINT
        batch_end_idx = min((batch_num + 1) * BATCH_SIZE_INPAINT, len(all_crops_data))
        current_batch_data = all_crops_data[batch_start_idx:batch_end_idx]

        batched_input_images = [item['input_image'] for item in current_batch_data]
        batched_input_masks = [item['input_mask'] for item in current_batch_data]
        batched_prompts = [item['prompt'] for item in current_batch_data]
        batched_neg_prompts = [item['negative_prompt'] for item in current_batch_data]

        start_inpaint = time.time()
        generated = pipeline.generate(
            prompt=batched_prompts,
            uncond_prompt=batched_neg_prompts,
            input_image=batched_input_images,
            input_mask=batched_input_masks,
            strength=0.9,
            do_cfg=True,
            cfg_scale=7.5,
            sampler_name="ddpm",
            n_inference_steps=N_INFERENCE_STEPS,
            seed=None,
            models=models,
            device=DEVICE,
            tokenizer=tokenizer,
            use_fp16=USE_FP16,
            use_tensorrt=USE_TENSORRT
        )
        inpaint_total_time += (time.time() - start_inpaint)

        for i, result_np in enumerate(generated):
            item_data = current_batch_data[i]
            result_pil = Image.fromarray(result_np)

            gen_crop_path = os.path.join(GEN_CROP_DIR, f"gen_crop_{batch_start_idx + i:03}.png")
            result_pil.save(gen_crop_path)


            coords = item_data['coords']
            original_crop_pil = item_data['original_crop_pil']
            blend_mask_pil = item_data['blend_mask_pil']

            result_resized = result_pil.resize((coords['box_w'], coords['box_h']), resample=Image.Resampling.LANCZOS)
            result_np = np.array(result_resized).astype(np.float32)
            original_np = np.array(original_crop_pil).astype(np.float32)

            blend_mask = np.array(blend_mask_pil.resize((coords['box_w'], coords['box_h']), resample=Image.Resampling.NEAREST)).astype(np.float32) / 255.0
            blend_mask = np.expand_dims(blend_mask, axis=-1)

            blended = result_np * blend_mask + original_np * (1 - blend_mask)
            final_img_np[coords['y1']:coords['y2'], coords['x1']:coords['x2']] = blended.astype(np.uint8)
            crop_counter += 1
else:
    print("[INFO] No valid people found for inpainting.")

# === Save final output ===
final_img_pil = Image.fromarray(final_img_np)
final_img_pil.save(OUT_PATH)
print(f"[INFO] Output image saved to: {OUT_PATH}")

# === Save performance CSV ===
total_time = time.time() - start_total
if crop_counter > 0:
    df_data = {
        # "Modello": f"YOLOv8 + SD 1.5 (Steps:{N_INFERENCE_STEPS} FP16:{USE_FP16} Compile:{USE_TORCH_COMPILE} Batch:{BATCH_SIZE_INPAINT})",
        "Segmentation model": YOLO_MODEL_NAME.split('.')[0],
        "Inpainting model": models_name["INPAINTING_MODEL_NAME"].split('.')[0],
        "Device": DEVICE,
        "TensorRT": USE_TENSORRT,
        "Inference steps": N_INFERENCE_STEPS,
        "Batch size":BATCH_SIZE_INPAINT,
        "Tempo Segmentazione (s)": round(yolo_time, 2),
        "Tempo Estrazione Crop (s)": round(crop_time, 2),
        "Tempo Inpainting Totale (s)": round(inpaint_total_time, 2),
        "Tempo Totale Script (s)": round(total_time, 2),
        "Persone Elaborate": crop_counter,
        "Tempo Medio Inpainting/Persona (s)": round(inpaint_total_time / crop_counter, 2)
    }
    df = pd.DataFrame([df_data])
    if os.path.exists(CSV_PATH):
        try:
            df_old = pd.read_csv(CSV_PATH)
            df = pd.concat([df_old, df], ignore_index=True)
        except Exception as e:
            print(f"[WARNING] Errore nel CSV esistente: {e}. Sovrascrivo.")
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Performance data saved to: {CSV_PATH}")
    print(df)

# === Profilazione finale ===
pr.disable()
s = StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(20)  # Mostra le 20 funzioni più lente
print(s.getvalue())

print(f"[✅] {crop_counter} persone elaborate.")
print("[INFO] Script completato.")
