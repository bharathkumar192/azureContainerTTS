import os
import logging
import warnings
import asyncio
import signal
import sys
import time
import fastapi
import struct
import json
import io
import subprocess
from typing import Optional, AsyncGenerator, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
import numpy as np
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import uvicorn

# Suppress redundant CUDA banners and verbose logging before importing torch/vLLM
os.environ["PYTORCH_CUDA_ERROR_REPORTING"] = "0"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "120"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Configure logging to reduce noise
for logger_name in ("torch", "vllm", "transformers"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)

import torch

# ===== CONFIGURATION CONSTANTS =====
MODEL_ID = "bharathkumar1922001/10speaker-aws-9ksteps-7epochs-8B2G"
VEENA_CACHE_PATH = "/mnt/model-cache/veena"
SNAC_CACHE_PATH = "/mnt/model-cache/snac"
AVAILABLE_SPEAKERS = [
    "aisha", "anika", "arfa", "asmr", "nikita", "raju", "rhea",
    "ruhaan", "sangeeta", "shayana",
]
DEFAULT_SPEAKER = "shayana"

# Speaker mapping: Customer-facing name -> Internal model name
SPEAKER_MAPPING = {
    "charu_soft": "aisha",
    "ishana_spark": "anika", 
    "kyra_prime": "arfa",
    "mohini_whispers": "asmr",
    "keerti_joy": "nikita",
    "varun_chat": "raju",
    "soumya_calm": "rhea",
    "agastya_impact": "ruhaan",
    "maitri_connect": "sangeeta",
    "vinaya_assist": "shayana",
}

# Load speaker details from JSON file
def load_speaker_details():
    # Try multiple locations for the speakers.json file
    possible_paths = [
        "/app/speakers.json",  # Container location
        os.path.join(os.path.dirname(__file__), "speakers.json"),  # Local development
        "speakers.json"  # Current directory fallback
    ]
    
    for speakers_file_path in possible_paths:
        try:
            with open(speakers_file_path, "r", encoding="utf-8") as f:
                print(f"Loaded speaker details from: {speakers_file_path}")
                return json.load(f)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as e:
            print(f"Error parsing speakers.json at {speakers_file_path}: {e}")
            raise
    
    # If none found, use fallback
    print("Warning: speakers.json not found in any location, using fallback speaker configuration")
    return {
        "vinaya_assist": {
            "id": "vinaya_assist",
            "name": "Vinaya Assist",
            "description": "Assistant & Helpful",
            "gender": "female",
            "language": "hindi",
            "use_cases": ["virtual_assistant", "tutorials", "instructions"],
            "voice_characteristics": ["helpful", "clear", "instructional"]
        }
    }

# Customer-facing speaker details loaded from JSON
SPEAKER_DETAILS = load_speaker_details()

DEFAULT_CUSTOMER_SPEAKER = "vinaya_assist"
DEFAULT_INTERNAL_SPEAKER = SPEAKER_MAPPING[DEFAULT_CUSTOMER_SPEAKER]

# Reverse mapping: Internal model name -> Customer-facing name (for efficiency)
INTERNAL_TO_CUSTOMER_MAPPING = {v: k for k, v in SPEAKER_MAPPING.items()}

SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
TARGET_AUDIO_SAMPLING_RATE = 24000
TOKENISER_LENGTH = 128256
START_OF_SPEECH_TOKEN = TOKENISER_LENGTH + 1
END_OF_SPEECH_TOKEN = TOKENISER_LENGTH + 2
START_OF_HUMAN_TOKEN = TOKENISER_LENGTH + 3
END_OF_HUMAN_TOKEN = TOKENISER_LENGTH + 4
START_OF_AI_TOKEN = TOKENISER_LENGTH + 5
END_OF_AI_TOKEN = TOKENISER_LENGTH + 6
AUDIO_CODE_BASE_OFFSET = TOKENISER_LENGTH + 10
PAD_TOKEN_ID = 128263

# Streaming window configuration
SNAC_WINDOW_SIZE_TOKENS = 28
SNAC_HOP_SIZE_TOKENS = 7
NUM_LLM_FRAMES_PER_DECODE_CHUNK = 4

GPU_CONFIG = "H100"
TORCH_CUDA_ARCH_LIST_FOR_BUILD = "9.0"
CONTAINER_TIMEOUT = 600
VLLM_INSTALL_VERSION = "0.8.5"

# Analytics configuration
ENABLE_ANALYTICS = True
ANALYTICS_OVERHEAD_THRESHOLD_MS = 1.0

# Audio format configuration
AudioFormat = Literal["wav", "raw_pcm", "opus", "webm"]

AUDIO_FORMAT_CONFIGS = {
    "wav": {
        "media_type": "audio/wav",
        "requires_header": True,
        "streamable": False,
        "latency": "high"
    },
    "raw_pcm": {
        "media_type": "audio/pcm",
        "requires_header": False,
        "streamable": True,
        "latency": "lowest"
    },
    "opus": {
        "media_type": "audio/opus",
        "requires_header": False,
        "streamable": True,
        "latency": "low"
    },
    "webm": {
        "media_type": "audio/webm",
        "requires_header": True,
        "streamable": True,
        "latency": "medium"
    }
}


# ===== AZURE KEY VAULT HELPER =====
class AzureSecretManager:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.vault_url = os.environ.get("AZURE_KEY_VAULT_URL", "https://veena-tts-kv.vault.azure.net/")
        self.client = SecretClient(vault_url=self.vault_url, credential=self.credential)
        
    def get_secret(self, secret_name: str) -> str:
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            print(f"Failed to get secret {secret_name}: {e}")
            return os.environ.get(secret_name.upper().replace('-', '_'), "")


# ===== REQUEST MODEL =====
class TTSRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech")
    speaker_id: str = Field(default=DEFAULT_CUSTOMER_SPEAKER, description="The speaker voice ID to use", json_schema_extra={"enum": list(SPEAKER_MAPPING.keys())})
    max_new_tokens: int = Field(default=1536, ge=100, le=3072)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    streaming: bool = Field(default=True, description="Enable streaming response")
    output_format: AudioFormat = Field(default="wav", description="Audio output format")
    seed: Optional[int] = Field(default=None)

    @validator("speaker_id")
    def validate_speaker(cls, v):
        speaker_key = v.lower()
        
        # Check if it's a valid customer-facing speaker ID
        if speaker_key in SPEAKER_MAPPING:
            return speaker_key
            
        # For backward compatibility, also accept internal speaker names
        if speaker_key in INTERNAL_TO_CUSTOMER_MAPPING:
            return INTERNAL_TO_CUSTOMER_MAPPING[speaker_key]
            
        # Neither customer nor internal speaker ID found
        available_speakers = list(SPEAKER_MAPPING.keys())
        raise ValueError(
            f"Speaker '{v}' not available. Choose from: {', '.join(available_speakers)}"
        )

    @validator("output_format")
    def validate_output_format(cls, v):
        if v not in AUDIO_FORMAT_CONFIGS:
            available_formats = list(AUDIO_FORMAT_CONFIGS.keys())
            raise ValueError(f"Format '{v}' not supported. Choose from: {', '.join(available_formats)}")
        return v


# ===== ANALYTICS CLASS =====
class GenerationAnalytics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.first_token_time = None
        self.first_audio_time = None
        self.end_time = None
        self.total_tokens = 0
        self.snac_tokens = 0
        self.audio_chunks = 0
        self.snac_decode_time = 0.0
        self.llm_generation_time = 0.0
    
    def mark_first_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()
    
    def mark_first_audio(self):
        if self.first_audio_time is None:
            self.first_audio_time = time.time()
    
    def mark_end(self):
        self.end_time = time.time()
    
    def add_snac_decode_time(self, duration: float):
        self.snac_decode_time += duration
    
    def get_metrics(self) -> Dict[str, Any]:
        total_time = (self.end_time or time.time()) - self.start_time
        ttft = (self.first_token_time - self.start_time) if self.first_token_time else None
        ttfa = (self.first_audio_time - self.start_time) if self.first_audio_time else None
        
        return {
            "total_time_s": round(total_time, 3),
            "ttft_s": round(ttft, 3) if ttft else None,
            "ttfa_s": round(ttfa, 3) if ttfa else None,
            "tokens_per_second": round(self.total_tokens / total_time, 2) if total_time > 0 else 0,
            "snac_tokens": self.snac_tokens,
            "audio_chunks": self.audio_chunks,
            "snac_decode_time_s": round(self.snac_decode_time, 3),
            "snac_decode_overhead_pct": round((self.snac_decode_time / total_time) * 100, 1) if total_time > 0 else 0
        }


# ===== AUDIO PROCESSOR CLASS =====
class AudioProcessor:
    def __init__(self, sample_rate: int = TARGET_AUDIO_SAMPLING_RATE):
        self.sample_rate = sample_rate
        self.opus_encoder = None
        self.webm_process = None
        
    def create_wav_header(self, sr=TARGET_AUDIO_SAMPLING_RATE, ch=1, bps=16) -> bytes:
        """Create WAV header for streaming"""
        data_chunk_size_for_streaming = 0 
        riff_chunk_size = 36 + data_chunk_size_for_streaming 
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", riff_chunk_size, 
            b"WAVE", b"fmt ", 16, 1, ch, sr,
            sr * ch * (bps // 8), ch * (bps // 8), bps,
            b"data", data_chunk_size_for_streaming 
        )
    
    def create_webm_header(self) -> bytes:
        """Create minimal WebM header for streaming"""
        webm_header = bytes([
            0x1A, 0x45, 0xDF, 0xA3,  # EBML signature
            0x9F, 0x42, 0x86, 0x81, 0x01,  # EBML version
            0x42, 0x86, 0x81, 0x01,  # EBML read version
            0x42, 0x85, 0x81, 0x01,  # EBML max ID length
            0x42, 0x87, 0x81, 0x08,  # EBML max size length
        ])
        return webm_header
    
    def process_audio_chunk(self, audio_bytes: bytes, output_format: AudioFormat) -> bytes:
        """Process audio chunk based on output format"""
        if output_format == "wav":
            return audio_bytes
        elif output_format == "raw_pcm":
            return audio_bytes
        elif output_format == "opus":
            return self._encode_opus_chunk(audio_bytes)
        elif output_format == "webm":
            return self._encode_webm_chunk(audio_bytes)
        else:
            return audio_bytes
    
    def _encode_opus_chunk(self, audio_bytes: bytes) -> bytes:
        """Encode audio chunk to Opus format"""
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            opus_frame = bytearray([0x78])
            opus_frame.extend(len(audio_data).to_bytes(2, 'little'))
            opus_frame.extend(audio_bytes)
            return bytes(opus_frame)
        except Exception as e:
            print(f"Opus encoding error: {e}")
            return audio_bytes
    
    def _encode_webm_chunk(self, audio_bytes: bytes) -> bytes:
        """Encode audio chunk to WebM format"""
        try:
            webm_block = bytearray([0xA3])
            webm_block.extend(len(audio_bytes).to_bytes(4, 'big'))
            webm_block.extend(audio_bytes)
            return bytes(webm_block)
        except Exception as e:
            print(f"WebM encoding error: {e}")
            return audio_bytes


# ===== SNAC PROCESSOR CLASS =====
class SNACProcessor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.num_levels = 3
        self.llm_codebook_base_offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    def load(self):
        from snac import SNAC
        print("Loading SNAC model...")
        
        # Try loading from cache first
        if os.path.exists(SNAC_CACHE_PATH) and os.listdir(SNAC_CACHE_PATH):
            try:
                print(f"Loading SNAC from cache: {SNAC_CACHE_PATH}")
                self.model = SNAC.from_pretrained(SNAC_CACHE_PATH).eval().to(self.device)
                self._post_load_setup()
                return
            except Exception as e:
                print(f"Failed to load from cache: {e}")
        
        # Download from HuggingFace and cache
        print(f"Downloading SNAC from HuggingFace: {SNAC_MODEL_NAME}")
        self.model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(self.device)
        
        # Save to cache
        try:
            os.makedirs(SNAC_CACHE_PATH, exist_ok=True)
            self.model.save_pretrained(SNAC_CACHE_PATH)
            print(f"Cached SNAC model to: {SNAC_CACHE_PATH}")
        except Exception as e:
            print(f"Failed to cache SNAC model: {e}")
            
        self._post_load_setup()

    def _post_load_setup(self):
        """Setup after model is loaded"""
        if hasattr(self.model, 'quantizer') and hasattr(self.model.quantizer, 'layers'):
            self.num_levels = len(self.model.quantizer.layers)
        elif hasattr(self.model, 'quantizer') and hasattr(self.model.quantizer, 'n_q'):
            self.num_levels = self.model.quantizer.n_q
        
        if self.num_levels == 3:
            self.level_strides = [4, 2, 1]
        elif self.num_levels == 4:
            self.level_strides = [8, 4, 2, 1]
        else:
            self.level_strides = [2 ** (self.num_levels - 1 - i) for i in range(self.num_levels)]
        
        try:
            if hasattr(torch, "compile") and self.device == "cuda":
                if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 0):
                    self.model.decoder = torch.compile(self.model.decoder, dynamic=True)
                    print("SNAC decoder compiled with torch.compile")
        except Exception:
            pass
        
        self._warmup()
        print("SNAC model ready")

    def _warmup(self):
        if not self.model:
            return
        with torch.inference_mode():
            dummy_data_for_snac_levels = [
                torch.randint(0, 4096, (1, 4), device=self.device, dtype=torch.int32),
                torch.randint(0, 4096, (1, 8), device=self.device, dtype=torch.int32),
                torch.randint(0, 4096, (1, 16), device=self.device, dtype=torch.int32)
            ]
            if self.num_levels != 3:
                dummy_data_for_snac_levels = [
                    torch.randint(0, 4096, (1, 4 * (2**i)), device=self.device, dtype=torch.int32) 
                    for i in range(self.num_levels)
                ]
            try:
                _ = self.model.decode(dummy_data_for_snac_levels[:self.num_levels])
            except Exception:
                pass

    def decode_chunk(self, snac_tokens_global_ids: List[int]) -> bytes:
        if not self.model or not snac_tokens_global_ids: 
            return b""
        
        if len(snac_tokens_global_ids) % 7 != 0:
            valid_len = (len(snac_tokens_global_ids) // 7) * 7
            if valid_len == 0: 
                return b""
            snac_tokens_global_ids = snac_tokens_global_ids[:valid_len]

        num_coarse_frames = len(snac_tokens_global_ids) // 7
        if num_coarse_frames == 0: 
            return b""

        codes_lvl0_local, codes_lvl1_local, codes_lvl2_local = [], [], []
        for i in range(0, len(snac_tokens_global_ids), 7):
            codes_lvl0_local.append(snac_tokens_global_ids[i]   - self.llm_codebook_base_offsets[0])
            codes_lvl1_local.append(snac_tokens_global_ids[i+1] - self.llm_codebook_base_offsets[1])
            codes_lvl1_local.append(snac_tokens_global_ids[i+4] - self.llm_codebook_base_offsets[4])
            codes_lvl2_local.append(snac_tokens_global_ids[i+2] - self.llm_codebook_base_offsets[2])
            codes_lvl2_local.append(snac_tokens_global_ids[i+3] - self.llm_codebook_base_offsets[3])
            codes_lvl2_local.append(snac_tokens_global_ids[i+5] - self.llm_codebook_base_offsets[5])
            codes_lvl2_local.append(snac_tokens_global_ids[i+6] - self.llm_codebook_base_offsets[6])
        
        hier_for_snac_decode_direct = [
            torch.tensor(codes_lvl0_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl1_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl2_local, dtype=torch.int32, device=self.device).unsqueeze(0)
        ]

        for lvl_tensor in hier_for_snac_decode_direct:
            if torch.any((lvl_tensor < 0) | (lvl_tensor > 4095)):
                return b""

        with torch.inference_mode():
            audio_hat = self.model.decode(hier_for_snac_decode_direct)
        
        if audio_hat is None or audio_hat.numel() == 0:
            return b""

        audio_data_np = audio_hat.squeeze().clamp(-1, 1).cpu().numpy()
        return (audio_data_np * 32767).astype(np.int16).tobytes()

    def decode_window_and_get_hop_slice(self, snac_tokens_global_ids: List[int]) -> bytes:
        """Decode a window of SNAC tokens and return only the hop slice to avoid overlap artifacts."""
        if not self.model or not snac_tokens_global_ids:
            return b""
            
        if len(snac_tokens_global_ids) != SNAC_WINDOW_SIZE_TOKENS:
            return self.decode_chunk(snac_tokens_global_ids)
        
        num_coarse_frames = len(snac_tokens_global_ids) // 7
        codes_lvl0_local, codes_lvl1_local, codes_lvl2_local = [], [], []
        
        for i in range(0, len(snac_tokens_global_ids), 7):
            codes_lvl0_local.append(snac_tokens_global_ids[i]   - self.llm_codebook_base_offsets[0])
            codes_lvl1_local.append(snac_tokens_global_ids[i+1] - self.llm_codebook_base_offsets[1])
            codes_lvl1_local.append(snac_tokens_global_ids[i+4] - self.llm_codebook_base_offsets[4])
            codes_lvl2_local.append(snac_tokens_global_ids[i+2] - self.llm_codebook_base_offsets[2])
            codes_lvl2_local.append(snac_tokens_global_ids[i+3] - self.llm_codebook_base_offsets[3])
            codes_lvl2_local.append(snac_tokens_global_ids[i+5] - self.llm_codebook_base_offsets[5])
            codes_lvl2_local.append(snac_tokens_global_ids[i+6] - self.llm_codebook_base_offsets[6])
        
        hier_for_snac_decode_direct = [
            torch.tensor(codes_lvl0_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl1_local, dtype=torch.int32, device=self.device).unsqueeze(0),
            torch.tensor(codes_lvl2_local, dtype=torch.int32, device=self.device).unsqueeze(0)
        ]
        
        for lvl_tensor in hier_for_snac_decode_direct:
            if torch.any((lvl_tensor < 0) | (lvl_tensor > 4095)):
                return b""
        
        with torch.inference_mode():
            audio_hat = self.model.decode(hier_for_snac_decode_direct)
        
        if audio_hat is None or audio_hat.numel() == 0:
            return b""
        
        if num_coarse_frames == NUM_LLM_FRAMES_PER_DECODE_CHUNK and audio_hat.shape[-1] >= 4096:
            audio_slice_for_yield = audio_hat[:, :, 2048:4096]
        else:
            samples_per_frame = audio_hat.shape[-1] // num_coarse_frames
            hop_frames = SNAC_HOP_SIZE_TOKENS // 7
            start_idx = samples_per_frame
            end_idx = start_idx + (samples_per_frame * hop_frames)
            audio_slice_for_yield = audio_hat[:, :, start_idx:end_idx]
        
        audio_data_np = audio_slice_for_yield.squeeze().clamp(-1, 1).cpu().numpy()
        return (audio_data_np * 32767).astype(np.int16).tobytes()


# ===== MAIN TTS API CLASS =====
class VeenaTTSAPI:
    def __init__(self):
        self._ready_event = asyncio.Event()
        self._failed_event = asyncio.Event()
        self._warmup_error = None
        self._shutdown_in_progress = False
        
        # Set device early
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        # Initialize components
        self.audio_processor = AudioProcessor()
        self.secret_manager = AzureSecretManager()
        
        # Create FastAPI app
        self.app = self._create_app()
        
        # Set up signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_in_progress = True
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def initialize(self):
        """Initialize the TTS system"""
        try:
            print("🔥 Initializing Veena TTS API...")
            
            # Get HuggingFace token from Azure Key Vault
            hf_token = self.secret_manager.get_secret("hf-token")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            
            print("🔥 Warming up tokenizer...")
            await self._init_tokenizer()
            
            print("🔥 Warming up vLLM engine...")
            await self._init_vllm_engine()
            
            print("🔥 Warming up SNAC processor...")
            self.snac_processor = SNACProcessor(self.device)
            self.snac_processor.load()
            
            print("✅ WARMUP COMPLETE - System ready!")
            self._ready_event.set()
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self._warmup_error = str(e)
            self._failed_event.set()
            raise

    async def _ensure_ready(self):
        """Wait for the system to be ready before processing requests"""
        if self._failed_event.is_set():
            raise RuntimeError(f"System failed to initialize: {self._warmup_error}")
        
        await self._ready_event.wait()
        
        if not hasattr(self, 'llm_engine') or not hasattr(self, 'snac_processor'):
            raise RuntimeError("System failed to initialize properly")

    def _get_model_path(self):
        """Get the best available model path"""
        if os.path.exists(VEENA_CACHE_PATH) and os.listdir(VEENA_CACHE_PATH):
            print(f"Using model from cache: {VEENA_CACHE_PATH}")
            return VEENA_CACHE_PATH
        
        print(f"Using model from HuggingFace: {MODEL_ID}")
        return MODEL_ID

    async def _init_tokenizer(self):
        from transformers import AutoTokenizer
        print("Loading tokenizer...")
        
        if os.path.exists(VEENA_CACHE_PATH) and os.listdir(VEENA_CACHE_PATH):
            try:
                print(f"Loading tokenizer from cache: {VEENA_CACHE_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    VEENA_CACHE_PATH,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_fast=True,
                )
                print("✓ Loaded tokenizer from cache")
                return
            except Exception as e:
                print(f"Failed to load from cache: {e}")
        
        print(f"Downloading tokenizer from HuggingFace: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=True,
        )
        
        # Cache the tokenizer
        try:
            os.makedirs(VEENA_CACHE_PATH, exist_ok=True)
            self.tokenizer.save_pretrained(VEENA_CACHE_PATH)
            print(f"Cached tokenizer to: {VEENA_CACHE_PATH}")
        except Exception as e:
            print(f"Failed to cache tokenizer: {e}")
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = PAD_TOKEN_ID
        print("✓ Tokenizer ready")

    async def _init_vllm_engine(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        print(f"Initializing vLLM engine (v{VLLM_INSTALL_VERSION})...")
        
        model_path = self._get_model_path()

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            use_v2_block_manager=True,
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("✓ vLLM engine created")
        await self._warmup_llm()

    async def _warmup_llm(self):
        from vllm import SamplingParams, TokensPrompt
        print("Warming up vLLM engine...")
        warm_text_for_vllm_prompt = f"{DEFAULT_INTERNAL_SPEAKER}: नमस्ते"
        warm_prompt_ids_list = self._format_prompt(warm_text_for_vllm_prompt)
        
        tokens_prompt_for_warmup = TokensPrompt(prompt_token_ids=warm_prompt_ids_list)
        
        stop_ids = list(set([END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN, self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else END_OF_AI_TOKEN]))
        sampling_params = SamplingParams(max_tokens=10, temperature=0.1, stop_token_ids=stop_ids)
        req_id = f"vllm_warmup_{time.time_ns()}"
        
        try:
            results_gen = self.llm_engine.generate(
                prompt=tokens_prompt_for_warmup,
                sampling_params=sampling_params,
                request_id=req_id,
            )
            async for out in results_gen:
                if out.finished:
                    break
            print("✓ vLLM warmup complete")
        except Exception as e:
            print(f"vLLM warmup failed: {e}")

    def _format_prompt(self, text_with_speaker: str) -> List[int]:
        speaker_tokens = self.tokenizer.encode(text_with_speaker, add_special_tokens=False)
        return [
            START_OF_HUMAN_TOKEN,
            *speaker_tokens,
            END_OF_HUMAN_TOKEN,
            START_OF_AI_TOKEN,
            START_OF_SPEECH_TOKEN,
        ]

    async def _generate_audio_unified(
        self, request: TTSRequest, analytics: GenerationAnalytics
    ) -> AsyncGenerator[bytes, None]:
        from vllm import SamplingParams, TokensPrompt

        await self._ensure_ready()

        internal_speaker_id = SPEAKER_MAPPING.get(request.speaker_id, request.speaker_id)
        prompt_ids_list = self._format_prompt(f"{internal_speaker_id}: {request.text}")
        tokens_prompt_for_generation = TokensPrompt(prompt_token_ids=prompt_ids_list)
        
        stop_ids = list(set([END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN, self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else END_OF_AI_TOKEN]))
        
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=max(0.01, request.temperature) if request.temperature > 0 else 0.0,
            top_p=request.top_p if request.temperature > 0 else 1.0,
            repetition_penalty=request.repetition_penalty,
            stop_token_ids=stop_ids,
            seed=request.seed,
        )
        req_id = f"tts_{int(time.time()*1000)}_{np.random.randint(1000)}"

        if AUDIO_FORMAT_CONFIGS[request.output_format]["requires_header"]:
            if request.output_format == "wav":
                yield self.audio_processor.create_wav_header()
            elif request.output_format == "webm":
                yield self.audio_processor.create_webm_header()

        snac_token_buffer: List[int] = []
        previous_output_length_in_ids = 0
        first_token_marked = False

        try:
            results_stream = self.llm_engine.generate(
                prompt=tokens_prompt_for_generation,
                sampling_params=sampling_params,
                request_id=req_id,
            )
            async for result in results_stream:
                if not result.outputs: 
                    continue
                
                current_generated_token_ids = result.outputs[0].token_ids
                newly_generated_token_ids = current_generated_token_ids[previous_output_length_in_ids:]
                previous_output_length_in_ids = len(current_generated_token_ids)

                if newly_generated_token_ids and not first_token_marked:
                    if analytics:
                        analytics.mark_first_token()
                    first_token_marked = True

                if analytics:
                    analytics.total_tokens += len(newly_generated_token_ids)

                for token_id in newly_generated_token_ids:
                    if AUDIO_CODE_BASE_OFFSET <= token_id < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
                        snac_token_buffer.append(token_id)
                        if analytics:
                            analytics.snac_tokens += 1

                while len(snac_token_buffer) >= SNAC_WINDOW_SIZE_TOKENS:
                    current_window_tokens = snac_token_buffer[:SNAC_WINDOW_SIZE_TOKENS]
                    
                    snac_start = time.time() if analytics else None
                    audio_bytes_slice = await asyncio.get_event_loop().run_in_executor(
                        None, self.snac_processor.decode_window_and_get_hop_slice, current_window_tokens
                    )
                    if analytics:
                        snac_duration = time.time() - snac_start
                        analytics.add_snac_decode_time(snac_duration)
                    
                    if audio_bytes_slice:
                        processed_audio = self.audio_processor.process_audio_chunk(audio_bytes_slice, request.output_format)
                        if analytics:
                            analytics.mark_first_audio()
                            analytics.audio_chunks += 1
                        yield processed_audio
                    
                    snac_token_buffer = snac_token_buffer[SNAC_HOP_SIZE_TOKENS:]
                
                if result.finished: 
                    break
                    
        except Exception as e:
            print(f"[{req_id}] Generation error: {e}")

        if len(snac_token_buffer) >= 7:
            final_flush_len = (len(snac_token_buffer) // 7) * 7
            if final_flush_len > 0:
                chunk_to_decode = snac_token_buffer[:final_flush_len]
                snac_start = time.time() if analytics else None
                audio_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self.snac_processor.decode_chunk, chunk_to_decode
                )
                if analytics:
                    analytics.add_snac_decode_time(time.time() - snac_start)
                if audio_bytes: 
                    processed_audio = self.audio_processor.process_audio_chunk(audio_bytes, request.output_format)
                    if analytics:
                        analytics.audio_chunks += 1
                    yield processed_audio
        
        if analytics:
            analytics.mark_end()

    def _create_app(self):
        app = fastapi.FastAPI(
            title="Veena TTS API - Maya Research",
            version="1.0.0",
            description="Advanced Text-to-Speech API with 10 Hindi voices and multiple output formats",
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/generate", response_class=Response)
        async def generate_endpoint(request: TTSRequest) -> Response:
            analytics = GenerationAnalytics() if ENABLE_ANALYTICS else None
            media_type = AUDIO_FORMAT_CONFIGS[request.output_format]["media_type"]
            
            if request.streaming:
                async def stream_with_analytics():
                    async for chunk in self._generate_audio_unified(request, analytics):
                        yield chunk
                    
                    if analytics:
                        metrics = analytics.get_metrics()
                        overhead_ms = (time.time() - analytics.start_time) * 1000 - metrics.get("total_time_s", 0) * 1000
                        if overhead_ms <= ANALYTICS_OVERHEAD_THRESHOLD_MS:
                            print(f"Analytics: {metrics}")
                
                return StreamingResponse(stream_with_analytics(), media_type=media_type)
            else:
                all_audio_bytes = b""
                async for chunk in self._generate_audio_unified(request, analytics):
                    all_audio_bytes += chunk
                
                headers = {}
                if analytics:
                    metrics = analytics.get_metrics()
                    headers.update({
                        "X-TTFT": str(metrics.get("ttft_s", 0)),
                        "X-TTFA": str(metrics.get("ttfa_s", 0)),
                        "X-Total-Time": str(metrics.get("total_time_s", 0)),
                        "X-Tokens-Per-Second": str(metrics.get("tokens_per_second", 0)),
                        "X-SNAC-Tokens": str(metrics.get("snac_tokens", 0)),
                        "X-Audio-Chunks": str(metrics.get("audio_chunks", 0)),
                        "X-Output-Format": request.output_format,
                    })
                
                return Response(content=all_audio_bytes, media_type=media_type, headers=headers)

        @app.get("/")
        async def root_endpoint() -> Dict[str, Any]:
            return {
                "product": "Veena",
                "company": "Maya Research", 
                "description": "Advanced Text-to-Speech API with 10 Hindi voices and multiple output formats",
                "version": app.version,
                "status": "operational",
                "developers": ["Dheemanth", "Bharath Kumar"],
                "languages_supported": ["Hindi"],
                "total_speakers": len(SPEAKER_DETAILS),
                "speaker_types": {"female_voices": 8, "male_voices": 2},
                "audio_formats": list(AUDIO_FORMAT_CONFIGS.keys()),
                "features": [
                    "Real-time streaming TTS",
                    "Multiple voice personalities", 
                    "Professional audio quality",
                    "Low latency generation",
                    "Multiple output formats (WAV, RAW_PCM, OPUS, WebM)"
                ],
                "endpoints": {
                    "generate_audio": "/generate",
                    "list_speakers": "/speakers", 
                    "speaker_details": "/speakers/{speaker_id}",
                    "system_status": "/status",
                    "health_check": "/health",
                    "audio_formats": "/formats"
                },
                "default_speaker": DEFAULT_CUSTOMER_SPEAKER,
                "analytics_enabled": ENABLE_ANALYTICS
            }

        @app.get("/formats")
        async def formats_endpoint() -> JSONResponse:
            return JSONResponse(content={
                "supported_formats": AUDIO_FORMAT_CONFIGS,
                "default_format": "wav",
                "streaming_recommended": ["raw_pcm", "opus", "webm"],
                "browser_compatibility": {
                    "wav": "Universal support",
                    "raw_pcm": "Requires Web Audio API",
                    "opus": "Modern browsers via MediaSource",
                    "webm": "Best native browser support"
                }
            })

        @app.get("/status")
        async def status_endpoint() -> Dict[str, Any]:
            if self._ready_event.is_set():
                state = "ready"
            elif self._failed_event.is_set():
                state = "failed"
            else:
                state = "initializing"
            
            response = {
                "ready": self._ready_event.is_set(),
                "state": state,
                "gpu": GPU_CONFIG,
                "version": app.version,
                "shutdown_in_progress": self._shutdown_in_progress
            }
            
            if self._failed_event.is_set() and self._warmup_error:
                response["error"] = self._warmup_error
                
            return response

        @app.get("/health")
        async def health_endpoint() -> Dict[str, Any]:
            return {
                "status": "healthy" if self._ready_event.is_set() else "warming_up", 
                "gpu": GPU_CONFIG,
                "ready": self._ready_event.is_set(),
                "shutdown_in_progress": self._shutdown_in_progress
            }

        @app.get("/speakers")
        async def speakers_endpoint() -> JSONResponse:
            speakers_sorted = sorted(SPEAKER_DETAILS.values(), key=lambda s: s["name"].lower())
            payload = {
                "speakers": speakers_sorted,
                "total_count": len(SPEAKER_DETAILS),
                "default_speaker": DEFAULT_CUSTOMER_SPEAKER,
                "languages_supported": ["Hindi"]
            }
            return JSONResponse(content=payload, headers={"Cache-Control": "public, max-age=3600"})

        @app.get("/speakers/{speaker_id:path}")
        async def speaker_detail_endpoint(speaker_id: str) -> JSONResponse:
            speaker_key = speaker_id.lower()
            
            if speaker_key not in SPEAKER_DETAILS:
                available_speakers = list(SPEAKER_DETAILS.keys())
                payload = {
                    "error": "Speaker not found",
                    "speaker_id": speaker_id,
                    "available_speakers": available_speakers
                }
                return JSONResponse(status_code=404, content=payload, headers={"Cache-Control": "public, max-age=3600"})
            
            speaker_info = SPEAKER_DETAILS[speaker_key].copy()
            speaker_info["internal_model_id"] = SPEAKER_MAPPING.get(speaker_key, "unknown")
            return JSONResponse(content=speaker_info, headers={"Cache-Control": "public, max-age=3600"})

        return app

    async def cleanup(self):
        print("Veena TTS API shutting down...")
        self._shutdown_in_progress = True
        
        try:
            if hasattr(self, 'llm_engine') and self.llm_engine is not None:
                try:
                    if hasattr(self.llm_engine, 'shutdown_engine'):
                        await self.llm_engine.shutdown_engine()
                    elif hasattr(self.llm_engine, 'shutdown'):
                        if asyncio.iscoroutinefunction(self.llm_engine.shutdown):
                            await self.llm_engine.shutdown()
                        else:
                            self.llm_engine.shutdown()
                    print("✓ vLLM engine shutdown complete")
                except Exception as e:
                    print(f"Error during vLLM shutdown: {e}")
                finally:
                    self.llm_engine = None

            if hasattr(self, 'snac_processor') and hasattr(self.snac_processor, 'model') and self.snac_processor.model is not None:
                del self.snac_processor.model
                self.snac_processor.model = None
                print("✓ SNAC processor cleaned up")
                
            if hasattr(self, 'tokenizer'): 
                del self.tokenizer
                print("✓ Tokenizer cleaned up")
            
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✓ CUDA cache cleared")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("✓ Cleanup complete")


# Global TTS API instance
tts_api = None

async def startup():
    global tts_api
    tts_api = VeenaTTSAPI()
    await tts_api.initialize()

async def shutdown():
    global tts_api
    if tts_api:
        await tts_api.cleanup()

# Create the FastAPI app
app = fastapi.FastAPI()

@app.on_event("startup")
async def startup_event():
    await startup()

@app.on_event("shutdown") 
async def shutdown_event():
    await shutdown()

# Mount the TTS API routes
@app.middleware("http")
async def ensure_tts_ready(request, call_next):
    global tts_api
    if tts_api and hasattr(tts_api, 'app'):
        # Forward request to TTS API
        return await tts_api.app(request.scope, request.receive, request.headers.get("x-forwarded-proto", "http"))
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="TTS API not ready")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True,
        workers=1
    )