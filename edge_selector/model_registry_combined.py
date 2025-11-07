"""
Model Registry for Edge AI Selector (CPU-Only Version)
------------------------------------------------------
Defines models for benchmarking and comparison using PyTorch and Transformers.
All models run on CPU by default and can switch to GPU later if available.
"""

import torch
import torchvision.models as models
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from ultralytics import YOLO

MODEL_REGISTRY = {}

# ========================
# 🖼️ 1. Image Classification
# ========================

def load_mobilenet_v2():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    input_shape = (1, 3, 224, 224)
    return model, input_shape

def load_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    input_shape = (1, 3, 224, 224)
    return model, input_shape

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
# 🎯 2. Object Detection (YOLOv8n - CPU)
# ========================

def load_yolov8n_cpu():
    """
    Loads YOLOv8n from Ultralytics (PyTorch implementation).
    Automatically downloads if missing and runs on CPU.
    """
    model = YOLO("yolov8n.pt")  # Automatically downloads weights if not present
    model.to("cpu")
    input_shape = (1, 3, 640, 640)
    return model, input_shape

MODEL_REGISTRY["yolov8n"] = {
    "task": "object_detection",
    "framework": "torch",
    "load_fn": load_yolov8n_cpu
}

# ========================
# 🔊 3. Speech to Text (Whisper Tiny)
# ========================

def load_whisper_tiny():
    model_name = "openai/whisper-tiny.en"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to("cpu")
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
    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.to("cpu")
    return (model, tokenizer), (1, 64)

MODEL_REGISTRY["distilbert_base"] = {
    "task": "nlp_classification",
    "framework": "torch",
    "load_fn": load_distilbert
}

# ========================
# 🎧 5. Audio Classification (YAMNet)
# ========================

def load_yamnet():
    """
    Loads YAMNet (audio event classification).
    Uses torch.hub to fetch a pretrained model.
    """
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()
    input_shape = (1, 1, 96, 64)
    return model, input_shape

MODEL_REGISTRY["yamnet"] = {
    "task": "audio_classification",
    "framework": "torch",
    "load_fn": load_yamnet
}

# ========================
# Utility functions
# ========================

def list_models():
    return list(MODEL_REGISTRY.keys())

def get_model(name):
    return MODEL_REGISTRY[name]["load_fn"]()

if __name__ == "__main__":
    print("✅ Registered Models:")
    for k, v in MODEL_REGISTRY.items():
        print(f"- {k} ({v['task']} - {v['framework']})")
