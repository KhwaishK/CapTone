import yaml
import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Load params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Acsess Parameters
MAX_TOKENS = params["caption"]["max_tokens"]
MODEL = params["caption"]["model_name"]
ANNOTATION_PATH = params["evaluation"]["annotation_path"]
MINI_VAL_DIR = params["dataset"]["mini_val_dir"]


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load BLIP captioning model (v1)
processor = BlipProcessor.from_pretrained(MODEL)
model = BlipForConditionalGeneration.from_pretrained(MODEL).to(device)


# Caption generator function
def generate_caption(image_path, max_tokens= MAX_TOKENS):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(out[0], skip_special_tokens=True)


def load_validation_data(annotation_path, mini_val_dir):
    # Load annotations
    with open(annotation_path, "r") as f:
        val_annotations = json.load(f)

    images = {img['id']: img['file_name'] for img in val_annotations['images']}

    # Build dataframe
    rows = []
    for ann in val_annotations['annotations']:
        image_id = ann['image_id']
        file_name = images[image_id]
        caption = ann['caption']
        rows.append({"image_id": image_id, "file_name": file_name, "caption": caption})

    df_val = pd.DataFrame(rows)

    # Filter mini validation set
    df_val_mini = df_val[df_val['file_name'].isin(os.listdir(mini_val_dir))].reset_index(drop=True)

    print(f"Original val size: {len(df_val)}, Mini val size: {len(df_val_mini)}")
    return df_val_mini


# Evaluation function
def evaluate_captions(df_val_mini):
    # Run captioning + evaluation
    generated = []
    smooth_fn = SmoothingFunction().method1  

    for fname in tqdm(df_val_mini['file_name'], desc = "Generating Captions"):
        img_path = os.path.join(MINI_VAL_DIR, fname)
        pred_caption = generate_caption(img_path)
        true_caps = df_val_mini[df_val_mini['file_name'] == fname]['caption'].tolist()
        
        # BLEU score (against all ground truths)
        references = [cap.split() for cap in true_caps]
        candidate = pred_caption.split()
        bleu = sentence_bleu(references, candidate, smoothing_function=smooth_fn)
        
        generated.append({
            "file_name": fname,
            "predicted": pred_caption,
            "ground_truths": true_caps,
            "bleu_score": bleu
        })

    return pd.DataFrame(generated)


def main():
    # Load mini validation set
    df_val_mini = load_validation_data(ANNOTATION_PATH, MINI_VAL_DIR)

    # Run evaluation
    results_df = evaluate_captions(df_val_mini)
    print(results_df.head())

    # Average BLEU over test images
    avg_bleu = results_df['bleu_score'].mean()
    print(f"\nAverage BLEU score on mini_val subset: {avg_bleu:.4f}")

if __name__ == "__main__":
    main()
