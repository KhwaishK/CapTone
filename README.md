# 🖼️ AI Captioning & Rephrasing Tool

This project generates captions for images using **BLIP (Bootstrapped Language Image Pretraining)** and then rephrases them into different tones using **LLaMA 3.1**, with hashtag generation powered by **KeyBERT**.  
It also includes model evaluation on a mini COCO dataset subset, using **BLEU scores** for caption quality assessment.

---

## 🚀 Live Demo

🎯 Try the live app here:  
👉 [**Hugging Face Demo**](https://huggingface.co/spaces/KhwaishK/CapTone)

---

## 📸 Features

- 🧠 **Automatic Image Captioning** using Salesforce BLIP model  
- 🏷️ **Hashtag Generation** using KeyBERT keyword extraction  
- ✍️ **Tone Rephrasing** with LLaMA 3.1 (Funny, Professional, Poetic, and Marketing/Ad tones)  
- 🧪 **Model Evaluation** on a mini COCO dataset with BLEU scores  
- 🖥️ **Streamlit Web App** for easy user interaction  

---

## 🧩 Project Structure
```graphql
📦 AI-Captioning-Rephrasing-Tool
│
├── app.py                     # Streamlit-based user interface
├── pipeline.py                # Core pipeline: captioning, hashtags, rephrasing
├── prepare_mini_dataset.py    # Script to create a mini COCO validation dataset
├── model_testing.py           # Evaluates BLIP model performance on mini dataset (BLEU)
│
├── data/
│   ├── annotations/           # COCO-style annotation JSONs
│   ├── mini_val2014/          # Subset of COCO validation images
│   └── samples/               # Sample images for testing
│
├── params.yaml                # Configuration for model paths, UI options, and parameters
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository 
```bash
git clone https://github.com/KhwaishK/CapTone.git
cd AI-Captioning-Rephrasing-Tool
```

### 2️⃣ Create a virtual environment
```bash
python -m venv captionenv
captionenv\Scripts\activate   # (Windows)
# OR
source captionenv/bin/activate  # (Mac/Linux)
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### ▶️ Running the App Locally
```bash
streamlit run app.py
```
Then open the provided local URL (usually http://localhost:8501) in your browser.

---

## ⚡ How It Works
-Image Upload → User uploads an image (or uses default sample).

-Caption Generation → BLIP model generates a natural caption.

-Hashtag Extraction → KeyBERT extracts top keywords as hashtags.

-Tone Rephrasing → User selects a tone and LLaMA 3.1 rewrites the caption accordingly.
