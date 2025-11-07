# file: edge_selector/benchmark.py
import time, psutil, numpy as np, json, uuid, torch, warnings, os
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# optional imports for preprocessing; harness will fallback if missing
try:
    import librosa
except Exception:
    librosa = None

try:
    from transformers import WhisperProcessor
except Exception:
    WhisperProcessor = None

class BenchmarkHarness:
    """
    Universal benchmarking class for Torch models (CPU/GPU).
    - end_to_end: if True, includes preprocessing + decoding where possible
    - sample_paths: dict with 'audio' or 'image' files to drive end-to-end tests
    """
    def __init__(self, model_obj, framework, device='cpu', end_to_end=True, sample_paths=None):
        self.framework = framework
        self.device = device if device in ['cpu','cuda'] else 'cpu'
        self.end_to_end = bool(end_to_end)
        self.sample_paths = sample_paths or {}

        # Model tuple (model, processor) support
        if isinstance(model_obj, tuple):
            self.model = model_obj[0]
            self.processor = model_obj[1]
        else:
            self.model = model_obj
            self.processor = None

        # Put model into eval and to device
        if hasattr(self.model, "eval"):
            try:
                self.model.eval()
            except Exception:
                pass

        # Move model to CPU/GPU if possible
        try:
            if hasattr(self.model, "to"):
                target = 'cuda' if (self.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
                self.model.to(target)
        except Exception:
            pass

    def _make_dummy_audio(self, sr=16000, duration=2.0):
        """
        Create a synthetic realistic audio waveform (sine + low noise).
        Returns numpy float32 waveform and sample rate.
        """
        t = np.linspace(0, duration, int(sr * duration), False)
        freq = 440.0  # A4
        sine = 0.3 * np.sin(2 * np.pi * freq * t)
        noise = 0.01 * np.random.randn(len(t))
        audio = (sine + noise).astype(np.float32)
        return audio, sr

    def _preprocess_whisper(self, audio_np, sr):
        """
        Convert raw waveform -> model input features via WhisperProcessor if available.
        If processor missing, fallback to a random feature tensor shaped (1,80,3000).
        """
        if WhisperProcessor is not None and self.processor is not None:
            # WhisperProcessor handles resampling & log-mel conversion
            # it expects sampling_rate=16000 usually
            proc = self.processor
            # processor.feature_extractor expects sampling_rate param sometimes:
            try:
                feats = proc.feature_extractor(audio_np, sampling_rate=sr, return_tensors="pt").input_features
            except Exception:
                # fallback: make random features
                feats = torch.randn(1, 80, 3000)
            return feats
        else:
            # fallback random feature tensor
            return torch.randn(1, 80, 3000)

    def _preprocess_yamnet(self, audio_np, sr):
        """
        Convert waveform -> log-mel spectrogram (96x64 typical dims).
        Uses librosa if available, else generates random mel.
        """
        if librosa is not None:
            # resample to 16000 if needed
            target_sr = 16000
            if sr != target_sr:
                audio_np = librosa.resample(audio_np.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            # compute mel spectrogram
            mel = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=96, n_fft=1024, hop_length=160)
            # log amplitude
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # typical vggish expects frames of shape (96,64) or similar; we will pad/truncate
            # ensure shape (1,1,96,64)
            if log_mel.shape[1] < 64:
                pad_width = 64 - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0,0),(0,pad_width)), mode='constant')
            log_mel = log_mel[:, :64]
            tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()
            return tensor
        else:
            # fallback random mel-like tensor
            return torch.randn(1,1,96,64)

    def run(self, input_shape=(1,3,224,224), runs=10):
        latencies = []

        # prepare sample inputs if end_to_end True
        audio_path = self.sample_paths.get('audio')
        image_path = self.sample_paths.get('image')

        # preload audio / image if files exist
        pre_audio_np, pre_audio_sr = None, None
        if self.end_to_end and audio_path and os.path.exists(audio_path):
            # try librosa load
            if librosa is not None:
                try:
                    pre_audio_np, pre_audio_sr = librosa.load(audio_path, sr=None)
                except Exception:
                    pre_audio_np, pre_audio_sr = None, None

        for i in range(runs):
            start = time.time()
            try:
                # === Whisper (ASR) ===
                if "WhisperForConditionalGeneration" in str(type(self.model)):
                    # get audio (real file if present else synth)
                    if self.end_to_end and pre_audio_np is not None:
                        audio_np, sr = pre_audio_np, pre_audio_sr
                    else:
                        audio_np, sr = self._make_dummy_audio(sr=16000, duration=2.0)

                    feats = self._preprocess_whisper(audio_np, sr)
                    feats = feats.to('cuda' if (self.device=='cuda' and torch.cuda.is_available()) else 'cpu')
                    with torch.no_grad():
                        # call generate — keep generation small to measure decode cost
                        out = self.model.generate(input_features=feats, max_new_tokens=5)
                        # we don't need to decode token ids here; generation itself costs time

                # === DistilBERT / text models ===
                elif "Bert" in str(type(self.model)) or "DistilBert" in str(type(self.model)):
                    if self.end_to_end and self.processor is not None:
                        # For text models, create a long-ish sample string
                        inputs = self.processor("This is a sample sentence for benchmarking edge model selector", return_tensors="pt", truncation=True, padding=True)
                        # move tensors to device if needed
                        inputs = {k: v.to('cuda' if (self.device=='cuda' and torch.cuda.is_available()) else 'cpu') for k,v in inputs.items()}
                        with torch.no_grad():
                            _ = self.model(**inputs)
                    else:
                        # quick synthetic token ids if no processor
                        dummy = torch.randint(0, 1000, (1,64))
                        dummy = dummy.to('cuda' if (self.device=='cuda' and torch.cuda.is_available()) else 'cpu')
                        with torch.no_grad():
                            _ = self.model(dummy)

                # === YAMNet / VGGish ===
                elif "VGGish" in str(type(self.model)) or "yamnet" in str(type(self.model)).lower():
                    if self.end_to_end:
                        if pre_audio_np is not None:
                            audio_np, sr = pre_audio_np, pre_audio_sr
                        else:
                            audio_np, sr = self._make_dummy_audio(sr=16000, duration=2.0)
                        mel = self._preprocess_yamnet(audio_np, sr)
                        mel = mel.to('cuda' if (self.device=='cuda' and torch.cuda.is_available()) else 'cpu')
                        with torch.no_grad():
                            _ = self.model(mel)
                    else:
                        # core inference on a fake mel spectrogram
                        t = torch.randn(*input_shape).to('cuda' if (self.device=='cuda' and torch.cuda.is_available()) else 'cpu')
                        with torch.no_grad():
                            _ = self.model(t)

                # === YOLO / vision or generic models ===
                else:
                    # if we have an image sample and end_to_end, try to read and preprocess
                    if self.end_to_end and image_path and os.path.exists(image_path):
                        # lightweight image preprocess (PIL + resize / normalize)
                        from PIL import Image
                        img = Image.open(image_path).convert('RGB').resize((input_shape[3], input_shape[2]))
                        arr = np.array(img).astype(np.float32) / 255.0
                        arr = np.transpose(arr, (2,0,1))[None,...]
                        t = torch.from_numpy(arr)
                    else:
                        t = torch.from_numpy(np.random.rand(*input_shape).astype(np.float32))

                    t = t.to('cuda' if (self.device=='cuda' and torch.cuda.is_available()) else 'cpu')
                    with torch.no_grad():
                        _ = self.model(t)

            except Exception as e:
                # log and skip this iteration (preserve overall run)
                print(f"⚠️ Inference skipped ({i}) for {type(self.model).__name__}: {e}")
                continue

            latencies.append((time.time() - start) * 1000)

        mem_mb = psutil.Process().memory_info().rss / (1024*1024)
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        p90_latency = float(np.percentile(latencies, 90)) if latencies else 0.0

        return {
            'run_id': str(uuid.uuid4()),
            'framework': self.framework,
            'latency_ms': avg_latency,
            'latency_p90': p90_latency,
            'mem_mb': mem_mb,
            'device': self.device
        }
