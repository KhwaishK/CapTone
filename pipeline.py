import os

# Redirect Hugging Face caches to temporary storage to prevent 50GB limit issues
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub"

for d in ["/tmp/hf_cache", "/tmp/hf_home", "/tmp/hf_hub"]:
    os.makedirs(d, exist_ok=True)

import torch
import yaml
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from keybert import KeyBERT
from groq import Groq

load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY. Please set it in .env ")

client = Groq(api_key=GROQ_API_KEY)

config_path = os.path.join(os.path.dirname(__file__), "params.yaml")
with open(config_path) as f:
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
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content.strip()

def process_image(image_path):
    caption = generate_caption(image_path)
    hashtags = generate_hashtags(caption)
    return caption, hashtags
