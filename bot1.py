import streamlit as st
from PIL import Image
import pytesseract
import PyPDF2
import tempfile
import os
from transformers import pipeline
import time

# Load the Hugging Face model pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

text_generator = load_pipeline()

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def analyze_medical_report(content, content_type):
    prompt = "Analyze this medical report concisely. Provide key findings, diagnoses, and recommendations:\n\n"
    
    full_prompt = prompt + content

    for attempt in range(MAX_RETRIES):
        try:
            response = text_generator(full_prompt, max_length=512, num_return_sequences=1)
            return response[0]["generated_text"]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"An error occurred. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)

def fallback_analysis(content, content_type):
    st.warning("Using fallback analysis method due to issues.")
    word_count = len(content.split())
    return f"""
    Fallback Analysis:
    1. Document Type: {'Image' if content_type == 'image' else 'Text-based medical report'}
    2. Word Count: Approximately {word_count} words
    3. Content: The document appears to contain medical information, but detailed analysis is unavailable.
    4. Recommendation: Please review the document manually or consult with a healthcare professional for accurate interpretation.
    """

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(image):
    """
    Extract text from an image using Tesseract OCR.
    """
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Failed to extract text from the image. Error: {str(e)}")
        return ""

def main():
    st.title("AI-driven Medical Report Analyzer (With OCR)")
    st.write("Upload a medical report (image or PDF) for analysis")

    file_type = st.radio("Select file type:", ("Image", "PDF"))

    if file_type == "Image":
        uploaded_file = st.file_uploader("Choose a medical report image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            image = Image.open(tmp_file_path)
            st.image(image, caption="Uploaded Medical Report", use_column_width=True)

            if st.button("Analyze Image Report"):
                with st.spinner("Extracting text from image..."):
                    extracted_text = extract_text_from_image(image)

                if extracted_text:
                    with st.spinner("Analyzing the extracted text..."):
                        analysis = analyze_medical_report(extracted_text, "image")
                        st.subheader("Analysis Results:")
                        st.write(analysis)
                else:
                    st.error("No text could be extracted from the image. Please try again with a clearer image.")

            os.unlink(tmp_file_path)

    else:  # PDF
        uploaded_file = st.file_uploader("Choose a medical report PDF", type=["pdf"])
        if uploaded_file is not None:
            st.write("PDF uploaded successfully")

            if st.button("Analyze PDF Report"):
                with st.spinner("Analyzing the medical report PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    with open(tmp_file_path, 'rb') as pdf_file:
                        pdf_text = extract_text_from_pdf(pdf_file)

                    analysis = analyze_medical_report(pdf_text, "text")
                    st.subheader("Analysis Results:")
                    st.write(analysis)

                    os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
