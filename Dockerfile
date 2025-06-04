# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV MAX_JOBS=4
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV CUDA_LAUNCH_BLOCKING=0
ENV PYTORCH_CUDA_ERROR_REPORTING=0
ENV VLLM_ENGINE_ITERATION_TIMEOUT_S=120
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libopus-dev \
    build-essential \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip and install basic packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install ninja packaging wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM with retry and fallback
RUN pip3 install --upgrade 'vllm==0.8.5' || \
    pip3 install --upgrade vllm --no-build-isolation || \
    pip3 install --upgrade vllm --no-deps --force-reinstall

# Install optional performance packages
RUN pip3 install flash-attn --no-build-isolation || echo 'Flash attention optional, continuing...'
RUN pip3 install flashinfer -i https://flashinfer.ai/whl/cu121 || echo 'FlashInfer optional, continuing...'

# Install Python dependencies
RUN pip3 install \
    transformers==4.52.3 \
    tokenizers==0.21.1 \
    accelerate \
    snac \
    huggingface_hub \
    hf_transfer \
    soundfile \
    numpy \
    scipy \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.9,<3.0" \
    xformers \
    ray \
    sentencepiece \
    protobuf \
    opuslib \
    av \
    azure-keyvault-secrets \
    azure-identity \
    azure-storage-file-share

# Create application directory
WORKDIR /app

# Create directories for model cache and file share mount
RUN mkdir -p /mnt/model-cache/veena /mnt/model-cache/snac

# Copy application files
COPY app.py /app/
COPY speakers.json /app/ 

# Create a startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for Azure File Share to be mounted if specified\n\
if [ ! -z "$AZURE_STORAGE_ACCOUNT" ] && [ ! -z "$AZURE_STORAGE_SHARE" ]; then\n\
    echo "Waiting for Azure File Share to be available..."\n\
    timeout=60\n\
    while [ $timeout -gt 0 ] && [ ! -d "/mnt/model-cache" ]; do\n\
        sleep 1\n\
        timeout=$((timeout-1))\n\
    done\n\
    \n\
    if [ ! -d "/mnt/model-cache" ]; then\n\
        echo "Warning: Model cache directory not available, using local storage"\n\
        mkdir -p /tmp/model-cache/veena /tmp/model-cache/snac\n\
        export VEENA_CACHE_PATH="/tmp/model-cache/veena"\n\
        export SNAC_CACHE_PATH="/tmp/model-cache/snac"\n\
    fi\n\
fi\n\
\n\
# Start the application\n\
exec python3 /app/app.py\n' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set the startup script as entrypoint
ENTRYPOINT ["/app/start.sh"]