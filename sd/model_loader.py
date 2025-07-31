from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
#from diffusion_trt import diffusionTRT
from diffusionTRT import diffusionTRT

import model_converter

def preload_models_from_standard_weights(model_path, device, use_tensorrt):
    state_dict = model_converter.load_from_standard_weights(model_path["INPAINTING_MODEL_NAME"], device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    if use_tensorrt:
        diffusion = diffusionTRT(model_path["TENSORTRT_MODEL_NAME"])
    else:
        diffusion = Diffusion().to(device)
        diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }