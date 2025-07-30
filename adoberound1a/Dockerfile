FROM python:3.10-slim

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    git \
    wget \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create work directory
WORKDIR /app

# Set up virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip + wheel
RUN pip install --upgrade pip setuptools wheel

# Install core scientific stack manually
RUN pip install numpy==1.21.6 torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Clone and install Detectron2 manually (v0.6)
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    sed -i 's/1.10.0/1.12.1/g' setup.py && \
    sed -i 's/numpy>=1.20/numpy==1.21.6/' setup.py && \
    pip install -e .

# Install layoutparser and dependencies
RUN pip install layoutparser==0.3.4 fvcore==0.1.5.post20221221

# Copy only requirements to install remaining
COPY requirements.txt .
RUN sed -i '/numpy/d;/torch/d;/torchvision/d;/detectron2/d;/fvcore/d;/layoutparser/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full codebase
COPY . .

# Ensure model is available
RUN mkdir -p mask_rcnn_X_101_32x8d_FPN_3x && \
    wget -q https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1 -O mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth

# Run the app
CMD ["python", "main.py"]
