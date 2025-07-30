FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------
# 🧱 Install system packages
# ---------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-jpn \
    poppler-utils \
    libgl1-mesa-glx \
    git \
    wget \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ---------------------
# 📁 Set working directory
# ---------------------
WORKDIR /app

# ---------------------
# 🐍 Set up virtual environment
# ---------------------
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---------------------
# 📦 Install Python dependencies
# ---------------------
RUN pip install --upgrade pip setuptools wheel

# Install pytesseract and OCR wrapper
RUN pip install pytesseract

RUN pip install "numpy>=1.22.4,<2.0" torch==1.12.1+cpu torchvision==0.13.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install compatible numpy, torch, torchvision
#RUN pip install numpy==1.22.4 torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu


# LayoutParser & dependencies
RUN pip install layoutparser==0.3.4 \
    fvcore==0.1.5.post20221221

# ---------------------
# 🔁 Detectron2 Setup
# ---------------------
# Clone Detectron2 and install
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    cd detectron2 && \
    git checkout v0.6 && \
    sed -i 's/Image.LINEAR/Image.BILINEAR/' detectron2/data/transforms/transform.py && \
    pip install --no-build-isolation .

# ---------------------
# 📜 Install additional requirements
# ---------------------
COPY requirements.txt .
RUN sed -i '/numpy/d;/torch/d;/torchvision/d;/detectron2/d;/fvcore/d;/layoutparser/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 🔁 Fix NumPy compatibility again (some packages might upgrade it)
RUN pip install "numpy>=1.22.4,<2.0" --force-reinstall

# ---------------------
# 📁 Copy full codebase
# ---------------------
COPY . .

# ---------------------
# ⬇️ Download model weights
# ---------------------
RUN mkdir -p mask_rcnn_X_101_32x8d_FPN_3x && \
    wget -q https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1 -O mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth

# ---------------------
# 🚀 Run the application
# ---------------------
CMD ["python", "main.py"]
