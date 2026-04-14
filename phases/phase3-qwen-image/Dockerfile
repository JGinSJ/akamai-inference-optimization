FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Suppress interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11, pip, and venv.
# Note: python3-pip is not a standard Ubuntu 22.04 apt package name.
# If this layer fails, replace python3-pip with python3-pip (which
# installs pip for the system python3) and invoke pip via
# "python3.11 -m pip" instead of pip3 directly.
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
# flash-attn lines in requirements.txt are comments — pip skips them safely.
# flash-attn itself requires a CUDA toolchain and must be installed
# separately if USE_FLASH_ATTN=1 is needed; see serve/model_optimized.py.
COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the serving package
COPY serve/ serve/

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
