import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model # type: ignore
import ollama
from duckduckgo_search import DDGS
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from typing import Optional
from uuid import uuid4
from pathlib import Path
import json
import shutil
import re

MODEL_NAME = "gemma3:4b"
TEMPERATURE = 1
MIN_P = 0.01
REPEAT_PENALTY = 1.0
TOP_K = 64
TOP_P = 0.95

class_labels = [
    "biotite",     # 0
    "bornite",     # 1
    "chrysocolla", # 2
    "malachite",   # 3
    "muscovite",   # 4
    "pyrite",      # 5
    "quartz"       # 6
]

SYSTEM_PROMPT = "Anda adalah asisten AI bernama Senku (tidak perlu memperkenalkan diri tiap saat). Berikan jawaban sesuai dengan kebutuhan pengguna dan jaga nada yang ramah dan profesional dengan bahasa Indonesia."

app = FastAPI()
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)

def check_mineral_image(image_path: str = None):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "user", 
                 "content": "Cukup jawab dengan \"ya\" atau \"tidak\". Apakah ini gambar mineral (batu)?",
                 "images": [image_path]
                }
            ],
            options={
                "temperature": TEMPERATURE,
                "min_p": MIN_P,
                "repeat_penalty": REPEAT_PENALTY,
                "top_k": TOP_K,
                "top_p": TOP_P
            }
        )
        # print(f"Response: {response['message']['content']}")
        return response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error checking mineral image: " + str(e))

def classify_mineral(image_path: str = None):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
        img = tf.keras.utils.img_to_array(img)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
        
        denseNet_model = load_model("densenet.h5")
        pred = denseNet_model.predict(img)
        pred_idx = np.argmax(pred)
        pred_label = class_labels[pred_idx]
        # print(f"Predicted label: {pred_label}")
        return pred_label
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error classifying mineral: " + str(e))

def check_search_cmd(query: str = None):
    try:      
        keywords = [
            "cari", "search", "browse", "browsing", "jual", "beli", "toko",
            "tokopedia", "shopee", "bukalapak", "olshop", "shop",
            "marketplace", "e-commerce", "temu"
        ]
        
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower:
                return True
        return False
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error checking search command: " + str(e))

def search_mineral(query: str):
    try:
        results = DDGS().text(
            keywords=query,
            region="wt-wt",
            safesearch="off",
            max_results=10
        )
        df = pd.DataFrame(results)
        # print(df['href'].tolist())
        return df['href'].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error searching mineral: " + str(e))

def generate_response(prompt: str, image_path: str = None):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        if image_path:
            messages[1]["images"] = [image_path]
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "min_p": MIN_P,
                "repeat_penalty": REPEAT_PENALTY,
                "top_k": TOP_K,
                "top_p": TOP_P
            }
        )
        # print(f"Response: {response['message']['content']}")
        return response['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating response: " + str(e))

@app.post("/chat")
async def chat(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    image_path = None
    if image:
        if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid image format. Only JPG, JPEG, and PNG are allowed.")
        
        image_path = TEMP_DIR / f"{uuid4()}.{image.filename.split('.')[-1]}"
        with image_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)
    try:
        if image_path:
            is_mineral = check_mineral_image(str(image_path))
            if any(word in is_mineral.lower() for word in ["ya", "yes", "yup", "y", "iya", "betul", "benar"]):
                pred_label = classify_mineral(str(image_path))
                if check_search_cmd(prompt):
                    search_results = search_mineral(prompt)
                    modified_query = re.sub(
                        r'\b(cari|carikan|search|jual|beli|jual-beli|toko|tokopedia|shopee|bukalapak|olshop|online shop|marketplace|e-commerce|ecommerce)\b',
                        '', prompt, flags=re.IGNORECASE
                    ).strip()
                    modified_query = f"Berikan saya informasi tentang {pred_label}. Konteks tambahan: {modified_query}"
                    response = generate_response(modified_query, str(image_path))

                    response_content = {
                        "response": response,
                        "mineral_label": pred_label,
                        "search": search_results
                    }
                else:
                    modified_query = f"{prompt}\nKonteks tambahan: gambar ini diklasifikasi sebagai {pred_label}"
                    response = generate_response(modified_query, str(image_path))

                    response_content = {
                        "response": response,
                        "mineral_label": pred_label,
                        "search": None
                    }
            else:
                response = generate_response(prompt, str(image_path))
                response_content = {
                    "response": response,
                    "mineral_label": None,
                    "search": None
                }
        else:
            response = generate_response(prompt)
            response_content = {
                "response": response,
                "mineral_label": None,
                "search": None
            }

        if image_path and image_path.exists():
            image_path.unlink()
        return response_content

    except HTTPException as e:
        return {"error": json.dumps(str(e.detail), ensure_ascii=False)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=2222)