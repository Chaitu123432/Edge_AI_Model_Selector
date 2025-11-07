# model_registry.py
"""
Model Registry for Edge AI Selector
----------------------------------
Central place to define models to be benchmarked and compared.
Each model is defined with a load() function returning a ready-to-run inference object.
"""

import torch
import torchvision.models as models
import numpy as np
import onnxruntime as ort

MODEL_REGISTRY = {}

# ========================
# 🖼️ 1. Image Classification
# ========================

def load_mobilenet_v2():
    model = models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
    return model, (1, 3, 224, 224)

def load_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1").eval()
    return model, (1, 3, 224, 224)

MODEL_REGISTRY["mobilenet_v2"] = {
    "task": "image_classification",
    "framework": "torch",
    "load_fn": load_mobilenet_v2
}
MODEL_REGISTRY["resnet18"] = {
    "task": "image_classification",
    "framework": "torch",
    "load_fn": load_resnet18
}

# ========================
# 🎯 2. Object Detection
# ========================

def load_yolov5n():
    """
    Loads YOLOv5n ONNX model (auto-downloads if not present).
    """
    import os, requests
    path = "yolov5n.onnx"
    if not os.path.exists(path):
        print("Downloading YOLOv5n ONNX...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n.onnx"
        r = requests.get(url)
        open(path, "wb").write(r.content)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, (1, 3, 640, 640)

MODEL_REGISTRY["yolov5n"] = {
    "task": "object_detection",
    "framework": "onnx",
    "load_fn": load_yolov5n
}

# ========================
# 🔊 3. Speech to Text (ASR)
# ========================

def load_whisper_tiny():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    model_name = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    return (model, processor), (1, 80, 3000)

MODEL_REGISTRY["whisper_tiny"] = {
    "task": "speech_to_text",
    "framework": "torch",
    "load_fn": load_whisper_tiny
}

# ========================
# 💬 4. NLP - Text Classification / Intent
# ========================

def load_distilbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    return (model, tokenizer), (1, 64)  # dummy input length

MODEL_REGISTRY["distilbert_base"] = {
    "task": "nlp_classification",
    "framework": "torch",
    "load_fn": load_distilbert
}

# ========================
# 🎧 5. Audio Classification (KWS)
# ========================

def load_yamnet():
    """
    YAMNet model via TorchHub (audio event classification)
    """
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    return model, (1, 1, 96, 64)

MODEL_REGISTRY["yamnet"] = {
    "task": "audio_classification",
    "framework": "torch",
    "load_fn": load_yamnet
}

# ========================
# Utility function
# ========================

def list_models():
    return list(MODEL_REGISTRY.keys())

def get_model(name):
    return MODEL_REGISTRY[name]["load_fn"]()

if __name__ == "__main__":
    print("✅ Registered Models:")
    for k,v in MODEL_REGISTRY.items():
        print(f"- {k} ({v['task']} - {v['framework']})")
