# -- IMPORTS

from diffusers import StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel, EulerDiscreteScheduler;
from huggingface_hub import hf_hub_download;
from safetensors.torch import load_file;
import torch;
import sys;
import os;
import random;
import argparse;
from PIL import Image;
from tqdm import tqdm;

# -- FUNCTIONS

def resize_to_fit(
    image_path: str,
    target_size: int = 1024
    ):

    try:

        image = Image.open( image_path ).convert( 'RGB' );

    except Exception as exception:

        print( f"Failed to load {image_path}: {exception}" );
        return None;

    width, height = image.size;

    scale = target_size / max( width, height );

    new_width = int( width * scale );
    new_height = int( height * scale );

    new_width = ( new_width // 8 ) * 8;
    new_height = ( new_height // 8 ) * 8;

    new_width = max( new_width, 512 );
    new_height = max( new_height, 512 );

    if scale > 3.0:

        print( f"Warning: {os.path.basename( image_path )} is very small. "
              f"Upscaling by {scale:.1f}x may reduce quality." );

    if new_width != width or new_height != height:

        resized_image = image.resize( ( new_width, new_height ), Image.LANCZOS );
        print( f"Resized {os.path.basename( image_path )}: {width}x{height} → {new_width}x{new_height}" );
        return resized_image;

    else:

        print( f"No resize needed for {os.path.basename( image_path )}" );
        return image;

# ~~

def setup_pipeline(
    step_count: int
    ):

    print( "Loading SDXL Lightning model..." );

    base_model = "stabilityai/stable-diffusion-xl-base-1.0";
    lightning_model = "ByteDance/SDXL-Lightning";

    if step_count <= 2:

        checkpoint_name = "sdxl_lightning_2step_unet.safetensors";

    elif step_count <= 4:

        checkpoint_name = "sdxl_lightning_4step_unet.safetensors";

    else:

        checkpoint_name = "sdxl_lightning_8step_unet.safetensors";

    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base_model,
        torch_dtype = torch.float16,
        variant = "fp16"
    );

    checkpoint_path = hf_hub_download( lightning_model, checkpoint_name );

    state_dict = load_file( checkpoint_path, device = "cuda" );
    pipeline.unet.load_state_dict( state_dict );

    pipeline.unet = pipeline.unet.to( dtype = torch.float16 );

    pipeline.scheduler = (
        EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing = "trailing"
            )
        );

    pipeline.enable_attention_slicing();
    pipeline.enable_vae_slicing();
    pipeline.enable_vae_tiling();
    pipeline.enable_model_cpu_offload();

    print( "Model loaded successfully" );
    return pipeline;

# ~~

def process_image(
    pipeline,
    image_path: str,
    output_path: str,
    prompt: str,
    strength: float,
    step_count: int
    ):

    init_image = resize_to_fit( image_path );

    if init_image is None:

        return;

    print( f"Processing: {os.path.basename( image_path )}" );
    print( f"Prompt: {prompt}" );
    print( f"Strength: {strength} | Steps: {step_count}" );

    try:

        seed = random.randint( 1, 999999 );
        generator = torch.Generator( device = "cpu" ).manual_seed( seed );
        result = (
            pipeline(
                prompt = prompt,
                image = init_image,
                strength = strength,
                num_inference_steps = step_count,
                guidance_scale = 1.0,
                generator = generator
                ).images[ 0 ]
            );
        result.save( output_path, quality = 95 );
        print( f"Saved: {output_path}" );

    except Exception as exception:

        print( f"Processing failed: {exception}" );

# ~~

def main(
    ):

    parser = (
        argparse.ArgumentParser(
            description = "SDXL-Lightning image enhancement for low-end hardware. Accepts both single files and directories as input/output paths.",
            formatter_class = argparse.RawDescriptionHelpFormatter
            )
        );
    parser.add_argument( "input_path", help = "Input folder containing images or path to a single image file" );
    parser.add_argument( "output_path", help = "Output folder or file path" );
    parser.add_argument( "--strength", type = float, default = 0.7, help = "Denoising strength (0.0-1.0, default: 0.7)" );
    parser.add_argument( "--steps", type = int, default = 4, help = "Inference steps (2, 4, or 8, default: 4)" );
    parser.add_argument( "--prompt", type = str, default = "highly detailed, sharp, 4k, clear", help = "Enhancement prompt" );

    args = parser.parse_args();

    if not os.path.exists( args.input_path ):

        print( f"Input path not found: {args.input_path}" );
        sys.exit( 1 );

    is_input_file = os.path.isfile( args.input_path );

    if is_input_file:

        image_file_array = [ args.input_path ];
        print( f"Processing single file: {args.input_path}" );

    else:

        if os.path.isfile( args.output_path ):

            print( f"Error: Input is a directory but output is a file: {args.output_path}" );
            sys.exit( 1 );

        supported_extension_set = { '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp' };

        image_file_array = (
            [
                os.path.join( args.input_path, file_name )
                for file_name in os.listdir( args.input_path )
                if os.path.splitext( file_name.lower() )[ 1 ] in supported_extension_set
            ]
            );

        if not image_file_array:

            print( "No supported images found in input folder" );
            sys.exit( 0 );

        image_file_count = len( image_file_array );
        print( f"Found {image_file_count} images to process\n" );

    if not is_input_file or ( is_input_file and os.path.isdir( args.output_path ) ):

        os.makedirs( args.output_path, exist_ok = True );

    pipeline = setup_pipeline( args.steps );

    for image_file in tqdm( image_file_array, desc = "Overall progress" ):

        if is_input_file:

            if os.path.isdir( args.output_path ):

                file_name = os.path.basename( image_file );
                output_file = os.path.join( args.output_path, file_name );

            else:

                output_file = args.output_path;

        else:

            file_name = os.path.basename( image_file );
            output_file = os.path.join( args.output_path, file_name );

        if os.path.exists( output_file ):

            print( f"Keeping: {output_file}" );
            continue;

        process_image( pipeline, image_file, output_file,
                     args.prompt, args.strength, args.steps );

        if torch.cuda.is_available():

            torch.cuda.empty_cache();

    print( "\nAll images processed successfully!" );

# -- STATEMENTS

if __name__ == "__main__":

    main();
