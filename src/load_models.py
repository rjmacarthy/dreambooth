import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

def load_pretrained_models(args):
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=torch.float32
    )

    return tokenizer, text_encoder, vae, unet
    