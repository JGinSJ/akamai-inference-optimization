FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04

# Suppress interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, and venv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies.
# Attention optimization uses sdpa via torch.nn.functional.scaled_dot_product_attention
# (built-in, no extra install) — dispatches to flash attention kernels automatically
# on Ada and newer GPUs at runtime.
COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the serving package
COPY serve/ serve/

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
