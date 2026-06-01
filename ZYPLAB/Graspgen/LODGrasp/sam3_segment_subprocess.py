import argparse
import gc

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 segmentation in an isolated subprocess.")
    parser.add_argument("--rgb-npy", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-npz", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rgb_data = np.load(args.rgb_npy).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_data)

    print(f"[SAM3 subprocess] loading model: {args.checkpoint}", flush=True)
    model = build_sam3_image_model(checkpoint_path=args.checkpoint)
    processor = Sam3Processor(model)
    model.to("cuda")

    inference_state = processor.set_image(rgb_image)
    output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)
    masks = output["masks"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    np.savez_compressed(args.output_npz, masks=masks, scores=scores)

    del output, inference_state, processor, model, rgb_image, rgb_data
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[SAM3 subprocess] saved: {args.output_npz}", flush=True)


if __name__ == "__main__":
    main()
