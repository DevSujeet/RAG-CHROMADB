from pdf2image import convert_from_path
from pytesseract import image_to_string
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader

def extract_text_pypdf2_optimized(pdf_path):
    """
    Page-by-Page Processing
    Process and extract text one page at a time, rather than loading the entire PDF at once.
    :param pdf_path:
    :return:
    """
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        yield page.extract_text()
# Usage
# pdf_path = "large.pdf"
# for page_text in extract_text_pypdf2_optimized(pdf_path):
#     print(page_text)  # Process each page's text incrementally



def extract_text_ocr_optimized(pdf_path, chunk_size=10):
    """
    Chunked OCR
    If OCR is required, process the PDF in smaller chunks (e.g., 10 pages at a time).
    :param pdf_path:
    :param chunk_size:
    :return:
    """
    images = convert_from_path(pdf_path)
    for i in range(0, len(images), chunk_size):
        chunk_images = images[i:i+chunk_size]
        text = ""
        for img in chunk_images:
            text += image_to_string(img)
        yield text



def extract_page_text(page):
    return page.extract_text()

def extract_text_multithreaded(pdf_path):
    """
    Multithreading/Multiprocessing
    Use multithreading or multiprocessing to extract text from multiple pages or chunks simultaneously.
    :param pdf_path:
    :return:
    """
    reader = PdfReader(pdf_path)
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_page_text, reader.pages)
    return "\n".join(results)


def extract_text_to_file(pdf_path, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for page_text in extract_text_pypdf2_optimized(pdf_path):
            file.write(page_text + "\n")


import re

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_and_clean(pdf_path):
    """
    Preprocess During Extraction
    Clean and preprocess the text while extracting it (e.g., remove unnecessary characters, normalize whitespace).
    :param pdf_path:
    :return:
    """
    for page_text in extract_text_pypdf2_optimized(pdf_path):
        yield clean_text(page_text)
