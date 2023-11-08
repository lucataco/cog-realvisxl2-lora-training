from cog import BaseModel, Input, Path
from model_train import train

MODEL_NAME = "SG161222/RealVisXL_V2.0"
MODEL_CACHE = "model-cache"

class Predictor(BaseModel):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    def predict(
        self,
        input_images: Path = Input(
            description="A .zip or .tar file containing the image files that will be used for fine-tuning"
        ),
        seed: int = Input(
            description="Random seed for reproducible training. Leave empty to use a random seed",
            default=None,
        ),
        resolution: int = Input(
            description="Square pixel resolution which your images will be resized to for training",
            default=768,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for training",
            default=4,
        ),
        num_train_epochs: int = Input(
            description="Number of epochs to loop through your training dataset",
            default=4000,
        ),
        max_train_steps: int = Input(
            description="Number of individual training steps. Takes precedence over num_train_epochs",
            default=1000,
        ),
        is_lora: bool = Input(
            description="Whether to use LoRA training. If set to False, will use Full fine tuning",
            default=True,
        ),
        unet_learning_rate: float = Input(
            description="Learning rate for the U-Net. We recommend this value to be somewhere between `1e-6` to `1e-5`.",
            default=1e-6,
        ),
        ti_lr: float = Input(
            description="Scaling of learning rate for training textual inversion embeddings. Don't alter unless you know what you're doing.",
            default=3e-4,
        ),
        lora_lr: float = Input(
            description="Scaling of learning rate for training LoRA embeddings. Don't alter unless you know what you're doing.",
            default=1e-4,
        ),
        lora_rank: int = Input(
            description="Rank of LoRA embeddings. Don't alter unless you know what you're doing.",
            default=32,
        ),
        lr_scheduler: str = Input(
            description="Learning rate scheduler to use for training",
            default="constant",
            choices=[
                "constant",
                "linear",
            ],
        ),
        lr_warmup_steps: int = Input(
            description="Number of warmup steps for lr schedulers with warmups.",
            default=100,
        ),
        token_string: str = Input(
            description="A unique string that will be trained to refer to the concept in the input images. Can be anything, but TOK works well",
            default="TOK",
        ),
        caption_prefix: str = Input(
            description="Text which will be used as prefix during automatic captioning. Must contain the `token_string`. For example, if caption text is 'a photo of TOK', automatic captioning will expand to 'a photo of TOK under a bridge', 'a photo of TOK holding a cup', etc.",
            default="a photo of TOK, ",
        ),
        mask_target_prompts: str = Input(
            description="Prompt that describes part of the image that you will find important. For example, if you are fine-tuning your pet, `photo of a dog` will be a good prompt. Prompt-based masking is used to focus the fine-tuning process on the important/salient parts of the image",
            default=None,
        ),
        crop_based_on_salience: bool = Input(
            description="If you want to crop the image to `target_size` based on the important parts of the image, set this to True. If you want to crop the image based on face detection, set this to False",
            default=True,
        ),
        use_face_detection_instead: bool = Input(
            description="If you want to use face detection instead of CLIPSeg for masking. For face applications, we recommend using this option.",
            default=False,
        ),
        clipseg_temperature: float = Input(
            description="How blurry you want the CLIPSeg mask to be. We recommend this value be something between `0.5` to `1.0`. If you want to have more sharp mask (but thus more errorful), you can decrease this value.",
            default=1.0,
        ),
        verbose: bool = Input(description="verbose output", default=True),
        checkpointing_steps: int = Input(
            description="Number of steps between saving checkpoints. Set to very very high number to disable checkpointing, because you don't need one.",
            default=999999,
        ),
        input_images_filetype: str = Input(
            description="Filetype of the input images. Can be either `zip` or `tar`. By default its `infer`, and it will be inferred from the ext of input file.",
            default="infer",
            choices=["zip", "tar", "infer"],
        ),
    ) -> Path:
        
        training_result = train(
            input_images,
            seed,
            resolution,
            train_batch_size,
            num_train_epochs,
            max_train_steps,
            is_lora,
            unet_learning_rate,
            ti_lr,
            lora_lr,
            lora_rank,
            lr_scheduler,
            lr_warmup_steps,
            token_string,
            caption_prefix,
            mask_target_prompts,
            crop_based_on_salience,
            use_face_detection_instead,
            clipseg_temperature,
            verbose,
            checkpointing_steps,
            input_images_filetype,
        )
        return training_result.weights