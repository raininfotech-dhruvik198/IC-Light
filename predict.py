# Prediction interface for Cog ⚙️
# https://cog.run/python

import time
import os
import math
import torch
import tempfile
import mimetypes
import subprocess
import numpy as np
from typing import List
import safetensors.torch as sf
from cog import BasePredictor, Input, Path

from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
# from diffusers import AutoencoderKL # Removed: Flux has its own VAE, BriaRMBG handles its needs.
# from diffusers.models.attention_processor import AttnProcessor2_0 # Removed: Typically handled by pipeline.
# from transformers import CLIPTextModel, CLIPTokenizer # FluxPipeline will handle its own
from briarmbg import BriaRMBG
from enum import Enum

mimetypes.add_type("image/webp", ".webp")

MODEL_CACHE = "models"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE


# Removed hooked_unet_forward function entirely as it's IC-Light specific.

# @torch.inference_mode() # Removed: IC-Light specific prompt encoding
# def encode_prompt_inner(txt: str):
#     max_length = tokenizer.model_max_length
#     chunk_length = tokenizer.model_max_length - 2
#     id_start = tokenizer.bos_token_id
#     id_end = tokenizer.eos_token_id
#     id_pad = id_end
#
#     def pad(x, p, i):
#         return x[:i] if len(x) >= i else x + [p] * (i - len(x))
#
#     tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
#     chunks = [
#         [id_start] + tokens[i : i + chunk_length] + [id_end]
#         for i in range(0, len(tokens), chunk_length)
#     ]
#     chunks = [pad(ck, id_pad, max_length) for ck in chunks]
#
#     token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
#     conds = text_encoder(token_ids).last_hidden_state
#
#     return conds
#
#
# @torch.inference_mode() # Removed: IC-Light specific prompt encoding
# def encode_prompt_pair(positive_prompt, negative_prompt):
#     c = encode_prompt_inner(positive_prompt)
#     uc = encode_prompt_inner(negative_prompt)
#
#     c_len = float(len(c))
#     uc_len = float(len(uc))
#     max_count = max(c_len, uc_len)
#     c_repeat = int(math.ceil(max_count / c_len))
#     uc_repeat = int(math.ceil(max_count / uc_len))
#     max_chunk = max(len(c), len(uc))
#
#     c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
#     uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]
#
#     c = torch.cat([p[None, ...] for p in c], dim=1)
#     uc = torch.cat([p[None, ...] for p in uc], dim=1)
#
#     return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = (
        torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    )  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(
    # input_fg, # Removed: This parameter is no longer used by the Flux implementation
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    # The following parameters are no longer used by this function:
    # highres_scale,
    # highres_denoise,
    # lowres_denoise,
    # bg_source,
):
    # The input_bg generation logic based on bg_source has been removed as it's not used
    # with the current Flux T2I implementation.
    # If specific background colors/gradients are needed, they could be achieved via prompt engineering
    # or by creating an initial image and using img2img if that feature is added.

    rng = torch.Generator(device=device).manual_seed(int(seed))

    # This function generates images using the FLUX text-to-image pipeline.
    # Parameters like lowres_denoise, highres_scale, highres_denoise were removed
    # as they are not directly used in this simplified T2I Flux implementation.
    # Future enhancements could add support for img2img based refinement.

    # rng = torch.Generator(device=device).manual_seed(int(seed)) # Redundant rng creation

    full_prompt = prompt + ", " + a_prompt # 'prompt' here is the background_prompt

    # Generate images using FluxPipeline (flux_pipe)
    images_pil = flux_pipe(
        prompt=full_prompt,
        negative_prompt=n_prompt,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        guidance_scale=cfg,
        output_type="pil",
    ).images

    result_numpy_images = []
    for img_pil in images_pil:
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        np_img = np.array(img_pil)
        result_numpy_images.append(np_img)

    return result_numpy_images


@torch.inference_mode()
def process_relight(
    original_subject_image, # Renamed from input_fg for clarity
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    # The following parameters are no longer used by this function or the 'process' function it calls:
    # highres_scale,
    # highres_denoise,
    # lowres_denoise,
    # bg_source,
):
    # 1. Remove background from the original subject image
    subject_fg_np, alpha_mask_np = run_rmbg(original_subject_image)
    pil_foreground = Image.fromarray(subject_fg_np) # This is the original, non-relit foreground

    # Create a generator for consistent seeding if multiple RNG operations are needed.
    # This generator will be passed to both background generation and foreground relighting.
    rng = torch.Generator(device=device).manual_seed(int(seed))

    # 2. Generate new background images using the text prompt via Flux (T2I)
    # The 'process' function now uses the restored flux_pipe T2I logic.
    generated_backgrounds_np = process(
        prompt=prompt,
        image_width=image_width,
        image_height=image_height,
        num_samples=num_samples,
        seed=seed, # process will create its own generator from this seed for T2I
        steps=steps,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        cfg=cfg,
        # highres_scale, highres_denoise, lowres_denoise, bg_source are removed from process() call
    )

    # 3. Relight the extracted foreground using FluxImg2ImgPipeline (i2i_pipe)
    # TODO: Make relit_strength a configurable parameter in Predictor.predict()
    relit_strength = 0.5
    # Use a portion of total steps for relighting, e.g., half, or make it configurable
    relit_steps = max(1, int(steps * 0.75)) # Ensure at least 1 step

    print(f"Relighting foreground with strength: {relit_strength}, steps: {relit_steps}")

    if pil_foreground.mode != 'RGB': # Ensure foreground is RGB for i2i_pipe
        pil_foreground = pil_foreground.convert('RGB')

    relit_prompt = prompt + ", " + a_prompt # Using the combined prompt for relighting context

    # Use the i2i_pipe for relighting the foreground
    relit_fg_pil = i2i_pipe(
        prompt=relit_prompt,
        negative_prompt=n_prompt,
        image=pil_foreground, # Input the segmented foreground
        strength=relit_strength,
        num_inference_steps=relit_steps,
        guidance_scale=cfg, # Re-use CFG, or could be a separate parameter
        generator=rng, # Use the shared generator
    ).images[0] # Output is a list of images, take the first one

    # Convert the relighted PIL image back to numpy array
    relit_fg_np = np.array(relit_fg_pil.convert('RGB'))

    # 4. Composite the *relighted* subject foreground onto the generated backgrounds
    composited_results = []
    # Convert the original alpha mask (from run_rmbg) to PIL
    alpha_mask_pil = Image.fromarray((alpha_mask_np * 255).astype(np.uint8), mode='L')
    # The subject for compositing is now the relighted PIL image
    subject_to_composite_pil = relit_fg_pil # Already a PIL image

    for bg_np in generated_backgrounds_np:
        bg_pil = Image.fromarray(bg_np) # Convert current background to PIL

        # Resize relighted foreground and its alpha mask to match background dimensions
        if subject_to_composite_pil.size != bg_pil.size:
            print(f"Resizing relighted foreground from {subject_to_composite_pil.size} to {bg_pil.size} for compositing.")
            subject_to_composite_pil_resized = subject_to_composite_pil.resize(bg_pil.size, Image.LANCZOS)
            # Also resize the original alpha mask to the new size for correct pasting
            alpha_mask_pil_resized = alpha_mask_pil.resize(bg_pil.size, Image.LANCZOS)
        else:
            subject_to_composite_pil_resized = subject_to_composite_pil
            # If sizes match, alpha_mask_pil should already match subject_to_composite_pil's original size,
            # which is pil_foreground's size. So, this alpha also needs to match bg_pil.size.
            alpha_mask_pil_resized = alpha_mask_pil

        temp_bg = bg_pil.copy() # Make a copy of the current background
        # Paste the relighted foreground onto the background using the (resized) alpha mask
        temp_bg.paste(subject_to_composite_pil_resized, (0,0), mask=alpha_mask_pil_resized)
        composited_results.append(np.array(temp_bg))

    # Return the relighted foreground (as numpy array, before compositing) and the final composited images
    # The first returned element 'output_bg' in predict() was originally the non-relighted foreground.
    # Returning relit_fg_np here makes sense if the user wants the "relighted subject" itself.
    return relit_fg_np, composited_results


# class BGSource(Enum): # Removed as light_source parameter is removed
#     NONE = "None"
#     LEFT = "Left Light"
#     RIGHT = "Right Light"
#     TOP = "Top Light"
#     BOTTOM = "Bottom Light"


# def download_weights(url: str, dest: str) -> None: # Removed: Not used, HF handles downloads.
#     # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
#     start = time.time()
#     print("[!] Initiating download from URL: ", url)
#     print("[~] Destination path: ", dest)
#     if ".tar" in dest:
#         dest = os.path.dirname(dest)
#     command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
#     try:
#         print(f"[~] Running command: {' '.join(command)}")
#         subprocess.check_call(command, close_fds=False)
#     except subprocess.CalledProcessError as e:
#         print(
#             f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
#         )
#         raise
#     print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Global variables for pipeline and components
        global flux_pipe, i2i_pipe, rmbg, device, quick_prompts, quick_subjects
        # Schedulers are still defined globally but not explicitly set on the pipeline here.
        # They could be passed at inference time if desired.
        global ddim_scheduler, euler_a_scheduler, dpmpp_2m_sde_karras_scheduler


        """Load the model into memory to make running multiple predictions efficient"""
        # Removed model_files list and download loop for SD1.5/IC-Light
        # base_url = f"https://weights.replicate.delivery/default/IC-Light/{MODEL_CACHE}/"

        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        # BriaRMBG is still needed
        # Check if BriaRMBG tar needs specific download, or if from_pretrained handles it.
        # Assuming from_pretrained handles it for now.
        # bria_model_tar = "models--briaai--RMBG-1.4.tar"
        # bria_url = base_url + bria_model_tar # This base_url is for IC-Light, Bria might have a different one or be on HF hub
        # bria_dest_path = os.path.join(MODEL_CACHE, bria_model_tar)
        # if not os.path.exists(bria_dest_path.replace(".tar", "")):
        #      download_weights(bria_url, bria_dest_path) # This might need to be a direct HF download or from_pretrained

        rmbg = BriaRMBG.from_pretrained(
            "briaai/RMBG-1.4",
            cache_dir=MODEL_CACHE,
            local_files_only=False, # Set to False for initial download
        )

        # Load FLUX pipeline
        flux_model_id = "black-forest-labs/FLUX.1-dev"
        # Note: FLUX.1-dev is very large. Consider FLUX.1-schnell if memory/speed issues arise.
        # The pipeline will load all necessary components: transformer, text encoders, VAE, tokenizer.
        print(f"Loading FLUX pipeline: {flux_model_id}")
        flux_pipe = FluxPipeline.from_pretrained(
            flux_model_id,
            torch_dtype=torch.bfloat16, # FLUX is typically used with bfloat16
            cache_dir=MODEL_CACHE,
            local_files_only=False, # Set to False for initial download
        )
        print("FLUX pipeline loaded.")

        # Removed IC-Light specific UNet changes and weight loading.

        device = torch.device("cuda")
        flux_pipe.to(device) # Move all components of the pipeline to GPU
        rmbg = rmbg.to(device=device, dtype=torch.float32) # BriaRMBG might prefer float32

        # Instantiate FluxImg2ImgPipeline sharing components
        print("Instantiating FluxImg2ImgPipeline...")
        i2i_pipe = FluxImg2ImgPipeline(
            vae=flux_pipe.vae,
            text_encoder=flux_pipe.text_encoder,
            text_encoder_2=flux_pipe.text_encoder_2,
            tokenizer=flux_pipe.tokenizer,
            tokenizer_2=flux_pipe.tokenizer_2,
            transformer=flux_pipe.transformer,
            scheduler=flux_pipe.scheduler, # Use the same scheduler instance
        )
        # No need to i2i_pipe.to(device) if all components are already on device.
        # If components were not on device, then i2i_pipe.to(flux_pipe.device) or i2i_pipe.to(device) would be needed.
        print("FluxImg2ImgPipeline instantiated.")

        # SDP - Flux pipeline should handle this internally.
        # flux_pipe.transformer.set_attn_processor(AttnProcessor2_0()) # Not needed for Flux typically
        # flux_pipe.vae.set_attn_processor(AttnProcessor2_0()) # Not needed for Flux typically


        # Samplers - Keep existing ones, Flux might use its own default or these can be passed.
        # Re-initialize schedulers with their original full parameters
        # This ensures they retain their distinct configurations for dynamic selection.

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1
        )

        dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

        # To use a specific scheduler by default with flux_pipe:
        # from diffusers import EulerDiscreteScheduler
        # flux_pipe.scheduler = EulerDiscreteScheduler.from_config(flux_pipe.scheduler.config)
        # Or, if you want to use one of the pre-defined ones:
        # flux_pipe.scheduler = dpmpp_2m_sde_karras_scheduler

        # Removed StableDiffusionPipeline and StableDiffusionImg2ImgPipeline instantiations
        # t2i_pipe = StableDiffusionPipeline(...)
        # i2i_pipe = StableDiffusionImg2ImgPipeline(...)

        quick_prompts = [
            "sunshine from window",
            "neon light, city",
            "sunset over sea",
            "golden time",
            "sci-fi RGB glowing, cyberpunk",
            "natural lighting",
            "warm atmosphere, at home, bedroom",
            "magic lit",
            "evil, gothic, Yharnam",
            "light and shadow",
            "shadow from window",
            "soft studio lighting",
            "home atmosphere, cozy bedroom illumination",
            "neon, Wong Kar-wai, warm",
        ]
        quick_prompts = [[x] for x in quick_prompts]

        quick_subjects = [
            "beautiful woman, detailed face",
            "handsome man, detailed face",
        ]
        quick_subjects = [[x] for x in quick_subjects]

    def predict(
        self,
        front_image: Path = Input(
            description="The main foreground image to be relighted and placed on a new background."
        ),
        background_prompt: str = Input(
            description="A text prompt to generate the background image."
        ),
        appended_prompt: str = Input(
            default="best quality",
            description="Additional text to be appended to the main prompt, enhancing image quality",
        ),
        negative_prompt: str = Input(
            default="lowres, bad anatomy, bad hands, cropped, worst quality",
            description="A text description of attributes to avoid in the generated images",
        ),
        width: int = Input(
            default=1024, # Adjusted default for Flux (typically 1024x1024)
            description="The width of the generated images in pixels",
            choices=[256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1344, 1408, 1472, 1536],
        ),
        height: int = Input(
            default=1024, # Adjusted default for Flux
            description="The height of the generated images in pixels",
            choices=[256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1152, 1280, 1344, 1408, 1472, 1536],
        ),
        steps: int = Input(
            default=30, # Adjusted default for Flux (often higher than SD1.5)
            description="The number of diffusion steps to perform during generation (more steps generally improves image quality but increases processing time)",
            ge=1,
            le=100,
        ),
        cfg: float = Input( # This is guidance_scale for Flux
            default=3.5, # Adjusted default for Flux (often lower than SD, e.g. 3.0-5.0, or even 0 for FLUX.1-schnell)
            description="Classifier-Free Guidance scale - higher values encourage adherence to prompt, lower values encourage more creative interpretation. For FLUX.1-schnell, 0 is often used.",
            ge=0.0,
            le=32.0,
        ),
        # highres_scale, highres_denoise, lowres_denoise, light_source are removed.
        # TODO: Consider adding relit_strength here if desired by user.
        seed: int = Input(
            description="A fixed random seed for reproducible results (omit this parameter for a randomized seed)",
            default=None,
        ),
        number_of_images: int = Input(
            default=1,
            description="The number of unique images to generate from the given input and settings",
            ge=1,
            le=12,
        ),
        output_format: str = Input(
            description="The image file format of the generated output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="The image compression quality (for lossy formats like JPEG and WebP). 100 = best quality, 0 = lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Use new parameter names internally
        image_width = width
        image_height = height
        num_samples = number_of_images
        a_prompt = appended_prompt # appended_prompt is still used for full prompt construction
        n_prompt = negative_prompt

        # Print statements with new names and removed params
        print(f"[!] ({type(front_image)}) front_image={front_image}")
        print(f"[!] ({type(background_prompt)}) background_prompt={background_prompt}")
        print(f"[!] ({type(image_width)}) image_width={image_width}")
        print(f"[!] ({type(image_height)}) image_height={image_height}")
        print(f"[!] ({type(num_samples)}) num_samples={num_samples}")
        print(f"[!] ({type(seed)}) seed={seed}")
        print(f"[!] ({type(steps)}) steps={steps}")
        print(f"[!] ({type(a_prompt)}) a_prompt={a_prompt}")
        print(f"[!] ({type(n_prompt)}) n_prompt={n_prompt}")
        print(f"[!] ({type(cfg)}) cfg={cfg}")

        input_fg_np = np.array(Image.open(str(front_image))) if front_image else None
        if input_fg_np is None:
            raise ValueError("front_image is required.")


        with torch.inference_mode():
            # Call process_relight with updated parameter names and without removed ones
            output_bg, result_gallery = process_relight(
                original_subject_image=input_fg_np,
                prompt=background_prompt, # This is the main prompt for background and relighting context
                image_width=image_width,
                image_height=image_height,
                num_samples=num_samples,
                seed=seed,
                steps=steps,
                a_prompt=a_prompt, # Appended prompt
                n_prompt=n_prompt, # Negative prompt
                cfg=cfg,
                # Removed: highres_scale, highres_denoise, lowres_denoise, bg_source
                # These parameters are now fully removed from process_relight's signature
                # highres_scale=None,
                # highres_denoise=None,
                # lowres_denoise=None,
                # bg_source=None,
            )

        # Create a directory to save the output images
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        # Save the background image
        extension = output_format.lower()
        extension = "jpeg" if extension == "jpg" else extension
        # bg_path = os.path.join(output_dir, f"background.{extension}")
        save_params = {"format": extension.upper()}
        if output_format != "png":
            save_params["quality"] = output_quality
            save_params["optimize"] = True
        # Image.fromarray(output_bg).save(bg_path, **save_params)

        # Save the generated images
        output_paths = [] #[Path(bg_path)]
        for i, img in enumerate(result_gallery):
            img_path = os.path.join(output_dir, f"generated_{i}.{extension}")
            print(f"[~] Saving to {img_path}...")
            print(f"[~] Output format: {extension.upper()}")
            if output_format != "png":
                print(f"[~] Output quality: {output_quality}")

            Image.fromarray(img).save(img_path, **save_params)
            output_paths.append(Path(img_path))

        return output_paths
