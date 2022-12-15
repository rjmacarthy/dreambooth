
# dreambooth

This repository contains a refactored and slimmed-down version of the [diffusers](https://github.com/ShivamShrirao/diffusers) project.

## Training StableDiffusion on your own images

To train StableDiffusion using on your own images, follow these steps:

1. Set up a directory structure with the following folders:
    * `pretrained_model`: This should contain the name pretrained model to download from huggingface, `runwayml/stable-diffusion-v1-5`.
    * `instance_dir`: This should contain the images that you want to train StableDiffusion on.
    * `output_dir`: This is where the trained model will be saved.
2. Run the `main.py` script with the following parameters:

```
main.py
--pretrained_model runwayml/stable-diffusion-v1-5
--instance_dir /path/to/instance_dir
--output_dir /path/to/output_dir
--instance_prompt a\ photo\ of\ sks\ person
--size 512
--batch_size 1
--gradient_accumulation_steps 1
--learning_rate 5e-6
--lr_scheduler constant
--lr_warmup_steps 0
--max_train_steps 1000
```

## convert to original stable diffusion ckpt

The `src/convert.py` file script allows you to convert your trained model to the original stable diffusion format.

To use it, run the following command:

```
src/convert.py 
    --model_path /home/person/models/output_dir
    --checkpoint_path /home/person/models/person.ckpt
```

The --model_path option specifies the location of your trained model, and the --checkpoint_path option specifies the location where the converted model will be saved.

Please note that the converted model will be saved in the original stable diffusion format, which may not be compatible with your current model. Therefore, it is recommended to save the converted model to a separate location to avoid overwriting your current model.

For more information on the available parameters and their meanings, please see the [diffusers](https://github.com/huggingface/diffusers) repository.
