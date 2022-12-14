import torch
from tqdm.auto import tqdm

from torchvision import transforms

def get_image_transforms(size):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def collate_fn(samples, tokenizer):
        input_ids = [sample["instance_prompt_ids"] for sample in samples]
        image_ids = [sample["instance_images"] for sample in samples]

        image_ids = torch.stack(image_ids)
        image_ids = image_ids.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "image_ids": image_ids,
        }
        return batch

def get_batches(
    train_dataloader,
    accelerator,
    vae,
    text_encoder,
    weight_dtype
):
    latents_cache = []
    text_encoder_cache = []
    for batch in tqdm(train_dataloader, desc="Caching latents"):
        with torch.no_grad():
            batch["image_ids"] = batch["image_ids"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
            batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
            latents_cache.append(vae.encode(batch["image_ids"]).latent_dist)
            text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
    return latents_cache, text_encoder_cache