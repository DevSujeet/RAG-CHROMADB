from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pytesseract import image_to_string
import os

class FileReadException(Exception):
    """
        Class for pdf read Execution related exceptions
    """
    def __init__(self,
                 code,
                 message="",
                 display_message=''):
        self.code = code
        self.message = message
        self.displayMessage = display_message
        super().__init__(self.message)


def __extract_text_pypdf2(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def __extract_text_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += image_to_string(img)
    return text

def extract_text(pdf_path, use_ocr=False):
    try:
        if use_ocr:
            return __extract_text_ocr(pdf_path)
        else:
            try:
                return __extract_text_pypdf2(pdf_path)
            except Exception as e:
                print(f"Failed with PyPDF2, falling back to OCR: {e}")
                return __extract_text_ocr(pdf_path)
    except Exception as e:
        FileReadException(code=600, message="Unable to extract from given pdf file")


# pdf_path = "sample.pdf"
# extracted_text = extract_text(pdf_path)
# print(extracted_text)
