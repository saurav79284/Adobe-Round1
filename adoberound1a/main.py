import cv2
import numpy as np
import pytesseract
import layoutparser as lp
from pdf2image import convert_from_path
import torch
import os
import json
from sklearn.cluster import KMeans
from config import PDF_PATH, OUTPUT, MODEL_CONFIG, LABEL_MAP, MODEL_PATH 

# Configuration
PDF_PATH = PDF_PATH
OUTPUT_DIR = OUTPUT
TESSERACT_PATH = r'/usr/bin/tesseract'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Load fine-tuned layout detection model"""
    return lp.Detectron2LayoutModel(
        config_path=MODEL_CONFIG,
        model_path=MODEL_PATH ,
        label_map=LABEL_MAP,
        device="cuda" if torch.cuda.is_available() else "cpu",
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85,
            "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.3,
            "MODEL.RPN.PRE_NMS_TOPK_TEST", 1000,
            "MODEL.RPN.POST_NMS_TOPK_TEST", 500
        ]
    )

def pdf_to_images(pdf_path):
    """Enhanced PDF to image conversion with optimal preprocessing"""
    images = convert_from_path(
        pdf_path,
        dpi=500,
        fmt='png',
        thread_count=4,
        grayscale=False,
        size=(3000, None),
        use_pdftocairo=True
    )
    
    processed_images = []
    for img in images:
        np_img = np.array(img)
        
        # Advanced contrast enhancement
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0 + np.std(l)/25, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Edge-preserving denoising
        lab = cv2.merge((cl, a, b))
        np_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        np_img = cv2.fastNlMeansDenoisingColored(np_img, None, 10, 10, 7, 21)
        
        processed_images.append(np_img)
    
    return processed_images

def extract_heading_features(block, image):
    """Advanced heading feature extraction with multi-language support"""
    try:
        x1, y1, x2, y2 = block.coordinates
        segment = block.crop_image(image)
        
        # Multi-stage preprocessing
        gray = cv2.cvtColor(segment, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 7)
        
        # Morphological enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Language detection
        custom_config = r'--oem 3 --psm 6'
        try:
            temp_text = pytesseract.image_to_string(processed, config=custom_config+' -l eng')
            lang = 'eng+jpn' if any(ord(c) > 127 for c in temp_text) else 'eng'
        except:
            lang = 'eng'
        
        # Final OCR
        data = pytesseract.image_to_data(processed,
                                       config=f'{custom_config} -l {lang}',
                                       output_type=pytesseract.Output.DICT)
        
        # Robust text validation
        valid_texts = []
        heights = []
        confidences = []
        
        for i, (t, conf) in enumerate(zip(data['text'], data['conf'])):
            if int(conf) > 70 and t.strip():
                valid_texts.append(t)
                heights.append(data['height'][i])
                confidences.append(float(conf)/100)
        
        if not valid_texts:
            return None
            
        # Calculate weighted features
        avg_height = np.average(heights, weights=confidences)
        boldness = 1 - (np.mean(gray)/255)
        
        return {
            'text': ' '.join(valid_texts).strip(),
            'font_size': float(avg_height),
            'area': float((x2 - x1) * (y2 - y1)),
            'centered': float(1 - (abs((x1 + x2)/2 - image.shape[1]/2) / (image.shape[1]/2))),
            'all_caps': float(sum(t.isupper() for t in valid_texts) / len(valid_texts)),
            'boldness': float(boldness),
            'position': float(y1),
            'block': block,
            'language': lang.split('+')[0]
        }
    except Exception as e:
        print(f"Skipping block due to error: {e}")
        return None

def classify_headings(all_headings):
    """Machine learning enhanced heading classification"""
    if not all_headings:
        return []

    features = ['font_size', 'area', 'centered', 'all_caps', 'boldness']
    
    # Robust normalization with outlier handling
    values = {f: [h[f] for h in all_headings] for f in features}
    q1 = {f: np.percentile(vals, 25) for f, vals in values.items()}
    q3 = {f: np.percentile(vals, 75) for f, vals in values.items()}
    iqr = {f: q3[f]-q1[f] for f in features}
    
    for heading in all_headings:
        heading['normalized_features'] = {}
        for f in features:
            val = max(q1[f]-1.5*iqr[f], min(q3[f]+1.5*iqr[f], heading[f]))
            heading['normalized_features'][f] = val

    # Dynamic weights based on feature variance
    variances = {f: np.var([h['normalized_features'][f] for h in all_headings]) 
                for f in features}
    total_variance = sum(variances.values())
    weights = {f: 0.8*(variances[f]/total_variance) + 0.2*(1/len(features)) 
              for f in features}
    
    # Calculate scores with position bonus
    max_page = max(h['page'] for h in all_headings) or 1
    for heading in all_headings:
        position_bonus = 0.1 * (1 - heading['page']/max_page)
        heading['score'] = sum(
            weights[f] * heading['normalized_features'][f] 
            for f in features
        ) + position_bonus

    # K-means clustering for level assignment
    X = np.array([[h['score']] for h in all_headings])
    
    if len(X) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        clusters = sorted(np.unique(kmeans.cluster_centers_))
        h1_thresh = clusters[2]
        h2_thresh = clusters[1]
    else:
        h1_thresh = np.percentile(X, 85) if len(X) > 0 else 0
        h2_thresh = np.percentile(X, 60) if len(X) > 0 else 0

    for heading in all_headings:
        if heading['score'] >= h1_thresh:
            heading['level'] = 'H1'
        elif heading['score'] >= h2_thresh:
            heading['level'] = 'H2'
        else:
            heading['level'] = 'H3'

    return all_headings

def validate_headings(headings):
    """Ensure proper heading hierarchy"""
    current_level = 'H1'
    for i in range(len(headings)):
        if headings[i]['level'] == 'H1':
            current_level = 'H1'
        elif headings[i]['level'] == 'H2' and current_level in ['H1', 'H2']:
            current_level = 'H2'
        elif headings[i]['level'] == 'H3' and current_level in ['H2', 'H3']:
            current_level = 'H3'
        else:
            if headings[i]['level'] == 'H2':
                headings[i]['level'] = 'H3'
            elif headings[i]['level'] == 'H1':
                headings[i]['level'] = 'H2'
    return headings

def process_pdf(pdf_path, model):
    """Complete PDF processing pipeline"""
    images = pdf_to_images(pdf_path)
    all_headings = []
    
    for page_num, image in enumerate(images, 1):
        print(f"Processing page {page_num}/{len(images)}")
        np_image = np.array(image)
        layout = model.detect(np_image)
        
        for block in layout:
            if block.type == "Title":
                features = extract_heading_features(block, np_image)
                if features:
                    features['page'] = page_num
                    all_headings.append(features)
    
    classified = classify_headings(all_headings)
    validated = validate_headings(classified)
    return sorted(validated, key=lambda x: (x['page'], x['position']))

def export_results(headings, output_dir):
    """Generate JSON output matching the specified schema"""
    os.makedirs(output_dir, exist_ok=True)

    # Get document title (first H1 or filename)
    title = os.path.splitext(os.path.basename(PDF_PATH))[0]
    for heading in headings:
        if heading['level'] == 'H1':
            title = heading['text']
            break

    # Prepare outline
    outline = []
    for heading in headings:
        if heading['level'] not in ['H1', 'H2', 'H3']:
            continue
        outline.append({
            "level": heading['level'],
            "text": heading['text'],
            "page": heading['page']
        })

    # Save JSON
    doc_structure = {"title": title, "outline": outline}
    json_path = os.path.join(output_dir, "document_outline.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(doc_structure, f, indent=2, ensure_ascii=False)
    print(f"Document outline saved to {json_path}")

def visualize_pages(pdf_path, headings, output_dir):
    """Generate visualizations with debugging"""
    try:
        images = pdf_to_images(pdf_path)
        colors = {'H1': (0, 0, 255), 'H2': (0, 165, 255), 'H3': (0, 255, 0)}
        
        for page_num, image in enumerate(images, 1):
            page_headings = [h for h in headings if h['page'] == page_num]
            if not page_headings:
                continue
                
            np_image = np.array(image)
            for heading in page_headings:
                try:
                    x1, y1, x2, y2 = heading['block'].coordinates
                    color = colors[heading['level']]
                    cv2.rectangle(np_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(np_image, heading['level'],
                               (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                except Exception as e:
                    print(f"Error visualizing heading on page {page_num}: {e}")
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"page_{page_num}_headings.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
            
    except Exception as e:
        print(f"Visualization failed: {e}")

def main():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    torch.set_num_threads(4)
    
    print("Loading model...")
    model = load_model()
    
    print(f"Processing PDF: {PDF_PATH}")
    headings = process_pdf(PDF_PATH, model)
    
    print("\nDocument Heading Hierarchy:")
    for heading in headings[:20]:  # Print first 20 headings
        print(f"Page {heading['page']}: {heading['level']} - {heading['text']}")
    
    print("\nGenerating outputs...")
    export_results(headings, OUTPUT_DIR)
    visualize_pages(PDF_PATH, headings, OUTPUT_DIR)

if __name__ == "__main__":
    main()