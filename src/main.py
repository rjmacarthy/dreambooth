import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import  DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from pathlib import Path
from functools import partial

from parse_args import parse_args
from save_weights import save_weights
from classes import DreamBoothDataset, LatentsDataset, AverageMeter
from load_models import load_pretrained_models
from utilities import collate_fn, get_batches

logger = get_logger(__name__)

def init():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(args):

    init()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        logging_dir=Path(args.output_dir, "0", args.logging_dir),
    )

    tokenizer, text_encoder, vae, unet = load_pretrained_models(args)
    
    vae.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        instance_prompt=args.instance_prompt,
        instance_dir=args.instance_dir,
        tokenizer=tokenizer,
        size=args.size,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True
    )

    weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)

    text_encoder.to(accelerator.device, dtype=weight_dtype)

    latents_cache, text_encoder_cache = get_batches(train_dataloader, accelerator, vae, text_encoder, weight_dtype)

    train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    del vae

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    args.train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    print("Training...please wait")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    for _ in range(args.train_epochs):
        unet.train()
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                with torch.no_grad():
                    latent_dist = batch[0][0]
                    latents = latent_dist.sample() * 0.18215

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = batch[0][1]
                prediction = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(prediction.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            if not global_step % args.log_interval:
                logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    save_weights(args, accelerator, unet)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
