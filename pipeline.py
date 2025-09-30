import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from keybert import KeyBERT
import ollama
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

BLIP_MODEL = params["caption"]["model_name"]
MAX_TOKENS = params["caption"]["max_tokens"]
TOP_K = params["hashtags"]["top_k"]
LLM_MODEL = params["llm"]["model_name"]


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(BLIP_MODEL)
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)


def generate_caption(image_path, max_tokens= MAX_TOKENS):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(out[0], skip_special_tokens=True)


kw_model = KeyBERT()

def generate_hashtags(caption, top_k= TOP_K):
    keywords = kw_model.extract_keywords(caption, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_k)
    hashtags = ["#" + kw[0].replace(" ", "") for kw in keywords]
    return hashtags


def rephrase_caption(base_caption, tone):
    prompt = f'Rewrite the caption "{base_caption}" in a {tone} tone.'
    response = ollama.chat(
        model= LLM_MODEL,  
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def process_image(image_path):
    caption = generate_caption(image_path)
    hashtags = generate_hashtags(caption)
    # tones_output = rephrase_caption(caption)
    return caption, hashtags

# if __name__ == "__main__":
#     test_image = "Data/samples/cat.jpg"   
#     process_image(test_image)
