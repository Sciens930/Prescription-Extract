import streamlit as st
import logging
from PIL import Image, ImageEnhance, ImageFilter
from deep_translator import GoogleTranslator
import langdetect
import uuid
import re
from google.cloud import documentai_v1 as documentai
import io
import os
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Document AI client
PROJECT_ID = "gen-lang-client-0639839201"
LOCATION = "us"
PROCESSOR_ID = "8a2f83abda5b5297"  # TODO: Update with the new processor ID

def get_documentai_client():
    try:
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path:
            st.write(f"GOOGLE_APPLICATION_CREDENTIALS is set to: {creds_path}")
            if os.path.exists(creds_path):
                st.write("Credentials file exists and is accessible.")
            else:
                st.error("Credentials file does not exist or is not accessible at the specified path.")
        else:
            st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")

        client = documentai.DocumentProcessorServiceClient()
        st.write("Document AI client initialized successfully.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Document AI client: {e}")
        return None

# Fallback test to confirm text presence using OpenCV
def test_text_presence(image):
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            return True, f"Detected {len(contours)} contours, indicating potential text presence."
        else:
            return False, "No contours detected; image may be blank or text not recognizable."
    except Exception as e:
        return False, f"Error in text presence test: {e}"

# Preprocess image for better OCR results
def preprocess_image_for_ocr(image):
    try:
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        st.write(f"Deskewed image by {angle:.2f} degrees.")
        img = Image.frombytes('L', (gray.shape[1], gray.shape[0]), gray)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        base_width = 3000
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

# Extract text from image using Google Document AI
def extract_text_from_image(image_obj):
    try:
        client = get_documentai_client()
        if not client:
            return None, "Document AI client initialization failed."

        text_present, text_presence_msg = test_text_presence(image_obj)
        update_log(f"Fallback text presence test: {text_presence_msg}")

        best_text = ""
        best_rotation = 0
        rotations = [0, 90, 180, 270]

        for angle in rotations:
            rotated_image = image_obj.rotate(angle, expand=True)
            img_byte_arr = io.BytesIO()
            rotated_image.save(img_byte_arr, format='PNG')
            raw_image = img_byte_arr.getvalue()
            name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
            input_config = documentai.RawDocument(content=raw_image, mime_type="image/png")
            request = documentai.ProcessRequest(
                name=name,
                raw_document=input_config
            )
            result = client.process_document(request=request)
            document = result.document
            text = document.text
            update_log(f"Text extracted at rotation {angle} degrees: {text[:100]}... (first 100 chars)")
            if hasattr(result, 'human_review_status'):
                update_log(f"Human review status: {result.human_review_status.state}")
            if len(text.strip()) > len(best_text):
                best_text = text
                best_rotation = angle

        if best_rotation != 0:
            st.write(f"Best OCR result obtained by rotating image {best_rotation} degrees.")

        if not best_text.strip():
            return None, "OCR detected no text. Try a clearer image."
        return best_text, None
    except Exception as e:
        return None, f"Error during OCR with Document AI: {e}"

# Organize text into sections
def organize_text_into_sections(text):
    sections = {
        "Header": [],
        "Superscription": [],
        "Inscription": [],
        "Subscription": [],
        "Signent": [],
        "Footer": []
    }
    current_section = "Header"
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "(Superscription)" in line:
            current_section = "Superscription"
            continue
        elif "Inscription)" in line or "Registration)" in line:
            current_section = "Inscription"
            continue
        elif "(Subscription)" in line:
            current_section = "Subscription"
            continue
        elif "(Signent" in line:
            current_section = "Signent"
            continue
        elif "EXP DATE" in line or "Exp date" in line or "DATE" in line or "Date" in line:
            current_section = "Footer"
        sections[current_section].append(line)
    
    return sections

# Format sections into a string with separators
def format_sections(sections, title):
    formatted = f"{title}\n" + "=" * 50 + "\n\n"
    for section_name, content in sections.items():
        if content:
            formatted += f"{section_name}\n" + "-" * 30 + "\n"
            formatted += "\n".join(content) + "\n\n"
    return formatted

# Clean text for better language detection
def clean_text_for_language_detection(text):
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned.strip()

# Detect language and translate text to English
def translate_text_to_english(text):
    try:
        lines = text.split('\n')
        translated_lines = []
        detected_langs = set()

        for line in lines:
            if not line.strip():
                translated_lines.append(line)
                continue
            if re.search(r'[\u0900-\u097F]', line):  # Hindi
                translator = GoogleTranslator(source='hi', target='en')
                translated_line = translator.translate(line)
                translated_lines.append(translated_line)
                detected_langs.add('hi')
            elif re.search(r'[\u0400-\u04FF]', line):  # Cyrillic
                translator = GoogleTranslator(source='ru', target='en')
                translated_line = translator.translate(line)
                translated_lines.append(translated_line)
                detected_langs.add('ru')
            elif re.search(r'[\u0530-\u058F]', line):  # Armenian
                translator = GoogleTranslator(source='hy', target='en')
                translated_line = translator.translate(line)
                translated_lines.append(translated_line)
                detected_langs.add('hy')
            elif re.search(r'[\u0590-\u05FF]', line):  # Hebrew
                translator = GoogleTranslator(source='he', target='en')
                translated_line = translator.translate(line)
                translated_lines.append(translated_line)
                detected_langs.add('he')
            elif re.search(r'[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]', line):  # Korean
                translator = GoogleTranslator(source='ko', target='en')
                translated_line = translator.translate(line)
                translated_lines.append(translated_line)
                detected_langs.add('ko')
            else:
                cleaned_line = clean_text_for_language_detection(line)
                if cleaned_line:
                    try:
                        detected_lang = langdetect.detect(cleaned_line)
                        detected_langs.add(detected_lang)
                        if detected_lang != 'en':
                            translator = GoogleTranslator(source=detected_lang, target='en')
                            translated_line = translator.translate(line)
                            translated_lines.append(translated_line)
                        else:
                            translated_lines.append(line)
                    except:
                        translated_lines.append(line)
                else:
                    translated_lines.append(line)

        translated_text = '\n'.join(translated_lines)
        update_log(f"Detected languages: {detected_langs}")
        if 'en' not in detected_langs:
            update_log("No English detected; translated all text.")
        return translated_text, detected_langs
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        return text, None

# Update processing log in Streamlit
def update_log(message):
    st.session_state.log.append(message)
    st.session_state.log_display = "\n".join(st.session_state.log)

# Main Streamlit app
def main():
    st.title("Prescription Extractor")
    st.write("Upload a prescription image to extract full details.")

    # Initialize session state
    if 'log' not in st.session_state:
        st.session_state.log = []
        st.session_state.log_display = ""
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    if 'raw_sections' not in st.session_state:
        st.session_state.raw_sections = None

    # File uploader
    uploaded_file = st.file_uploader("Choose a prescription image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Prescription', use_column_width=True)

        if st.button("Extract Full Details"):
            st.session_state.log = []
            update_log("1. Preprocessing image...")

            try:
                processed_image = preprocess_image_for_ocr(image)
                update_log("Image preprocessed successfully.")
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                return

            update_log("2. Performing OCR with Google Document AI...")
            text, error = extract_text_from_image(processed_image)
            if error:
                st.error(error)
                update_log(f"OCR Error: {error}")
                return
            update_log(f"   Raw OCR Text (full):\n{text}")
            st.session_state.raw_text = text

            update_log("3. Translating text to English...")
            translated_text, detected_langs = translate_text_to_english(text)
            st.session_state.translated_text = translated_text

            # Organize raw text into sections
            raw_sections = organize_text_into_sections(text)
            st.session_state.raw_sections = raw_sections

            update_log("4. Processing complete!")

    if st.session_state.raw_text is not None:
        raw_sections = st.session_state.raw_sections or organize_text_into_sections(st.session_state.raw_text)
        translated_sections = organize_text_into_sections(st.session_state.translated_text)
        raw_formatted = format_sections(raw_sections, "Raw OCR Text")
        translated_formatted = format_sections(translated_sections, "Translated Text")
        full_details = f"{raw_formatted}\n{translated_formatted}"
        st.download_button(
            label="Download Full Details",
            data=full_details,
            file_name=f"prescription_full_details_{str(uuid.uuid4())}.txt",
            mime="text/plain",
            key="download_full_details"
        )

    st.subheader("Processing Log")
    st.text(st.session_state.log_display)

if __name__ == "__main__":
    main()