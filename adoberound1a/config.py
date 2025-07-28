# config.py

PDF_PATH = "input/input.pdf"
OUTPUT_DIR = "outputs"
MODEL_CONFIG = 'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config'
MODEL_PATH = 'mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth'
LABEL_MAP = {0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"}
