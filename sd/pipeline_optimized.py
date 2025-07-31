import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from PIL import Image

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep, device="cpu", dtype=torch.float32):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=dtype) / 160).to(device)
    x = torch.tensor([timestep], dtype=dtype, device=device)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    input_mask=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    tokenizer=None,
    use_fp16=False,
    use_tensorrt=False
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        current_dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32

        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif input_image and isinstance(input_image, list):
            batch_size = len(input_image)
        else:
            batch_size = 1
            if isinstance(prompt, str):
                prompt = [prompt]
            if isinstance(uncond_prompt, str) or uncond_prompt is None:
                uncond_prompt = [uncond_prompt] * batch_size if uncond_prompt is not None else [""] * batch_size
            if input_image and not isinstance(input_image, list):
                input_image = [input_image] * batch_size
            if input_mask and not isinstance(input_mask, list):
                input_mask = [input_mask] * batch_size

        if len(uncond_prompt) == 1 and batch_size > 1:
            uncond_prompt = uncond_prompt * batch_size

        if not (len(prompt) == batch_size and 
                len(uncond_prompt) == batch_size and 
                (not input_image or len(input_image) == batch_size) and
                (not input_mask or len(input_mask) == batch_size)):
            raise ValueError("La lunghezza di prompt, uncond_prompt, input_image e input_mask deve essere la stessa per il batching.")

        clip = models["clip"]
        encoder = models["encoder"]
        diffusion = models["diffusion"]
        decoder = models["decoder"]

        generators = []
        if seed is None:
            for _ in range(batch_size):
                gen = torch.Generator(device=device)
                gen.seed()
                generators.append(gen)
        elif isinstance(seed, int):
            for _ in range(batch_size):
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                generators.append(gen)
        elif isinstance(seed, list) and len(seed) == batch_size:
            for s_val in seed:
                gen = torch.Generator(device=device)
                if s_val is not None:
                    gen.manual_seed(s_val)
                else:
                    gen.seed()
                generators.append(gen)
        else:
            raise ValueError("Seed deve essere None, un intero, o una lista di interi/None di lunghezza batch_size.")

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(uncond_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context]).to(dtype=current_dtype)
        else:
            tokens = tokenizer.batch_encode_plus(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
            context = clip(tokens).to(dtype=current_dtype)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generators[0])
            sampler.set_inference_timesteps(n_inference_steps)
            full_timesteps = sampler.timesteps.clone()
        else:
            raise ValueError(f"Unknown sampler value {sampler_name}.")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        batch_latents_list = []

        if input_image:
            for i in range(batch_size):
                current_img_pil = input_image[i]
                if current_img_pil is None:
                    noise = torch.randn((1, 4, LATENTS_HEIGHT, LATENTS_WIDTH), generator=generators[i], device=device, dtype=current_dtype)
                    batch_latents_list.append(noise)
                    continue

                img_tensor = np.array(current_img_pil.resize((WIDTH, HEIGHT), resample=Image.Resampling.LANCZOS))
                img_tensor = torch.from_numpy(img_tensor).to(device, dtype=current_dtype) / 255.0
                img_tensor = rescale(img_tensor, (0, 1), (-1, 1))
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

                encoder_noise = torch.randn(latents_shape, generator=generators[i], device=device, dtype=current_dtype)
                img_latents = encoder(img_tensor, encoder_noise)

                sampler.timesteps = full_timesteps.clone()
                sampler.set_strength(strength=strength)

                if len(sampler.timesteps) == 0:
                    raise ValueError("Sampler.timesteps is empty after applying strength. Check strength value.")

                img_latents = sampler.add_noise(img_latents, sampler.timesteps[0])
                batch_latents_list.append(img_latents)

            latents = torch.cat(batch_latents_list, dim=0)
        else:
            latents = torch.randn((batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH), generator=generators[0], device=device, dtype=current_dtype)

        timesteps_iterable = tqdm(sampler.timesteps) if batch_size == 1 else sampler.timesteps
        for i, timestep in enumerate(timesteps_iterable):
            time_embedding = get_time_embedding(timestep, device=device, dtype=current_dtype)
            model_input = latents

            if do_cfg:
                model_input = torch.cat([latents, latents], dim=0)

            if use_tensorrt:
                #latent_np = model_input.detach().cpu().numpy()
                #context_np = context.detach().cpu().numpy()
                #time_np = time_embedding.detach().cpu().numpy()
                
                # Inference con TensorRT
                #model_output = diffusion.infer(latent_np, context_np, time_np).to(device).to(dtype=current_dtype)
                # Inference con TensorRT (usiamo tensori su GPU)
                latent_tensor = model_input.to(device=device, dtype=current_dtype)
                context_tensor = context.to(device=device, dtype=current_dtype)
                time_tensor = time_embedding.to(device=device, dtype=current_dtype)

                print(f"[INFER] latent: shape={latent_tensor.shape}, min={latent_tensor.min().item()}, max={latent_tensor.max().item()}, mean={latent_tensor.mean().item()}")
                print(f"[INFER] context: shape={context_tensor.shape}, min={context_tensor.min().item()}, max={context_tensor.max().item()}, mean={context_tensor.mean().item()}")
                print(f"[INFER] timestep: shape={time_tensor.shape}, min={time_tensor.min().item()}, max={time_tensor.max().item()}, mean={time_tensor.mean().item()}")

                #model_output = diffusion.infer(latent_tensor, context_tensor, time_tensor)
                outputs = []
                for i in range(latent_tensor.shape[0]):
                    latent_i = latent_tensor[i:i+1].contiguous()
                    context_i = context_tensor[i:i+1].contiguous()
                    timestep_i = time_tensor  # già (1, 320), va bene per tutti

                    out_i = diffusion.infer(latent_i, context_i, timestep_i)
                    outputs.append(out_i)

                model_output = torch.cat(outputs, dim=0)

            else:
                model_output = diffusion(model_input, context, time_embedding)

            #if do_cfg:
            #    output_cond, output_uncond = model_output.chunk(2)
            #    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            if do_cfg and model_output.shape[0] == 2 * latents.shape[0]:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            else:
                # Salta la fusione CFG se l'output è già uno solo
                pass


            latents = sampler.step(timestep, latents, model_output)

        #images = decoder(latents)
        images = decoder(latents.to(dtype=next(decoder.parameters()).dtype))

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images_np = images.to("cpu", torch.uint8).numpy()

        return [images_np[i] for i in range(images_np.shape[0])]
