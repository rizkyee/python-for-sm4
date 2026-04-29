import pdfplumber
from docx import Document
import config
import os
import uuid

def convert(file_id):
    input_path = os.path.join(config.UPLOAD_FOLDER, f"{file_id}.pdf")

    if not os.path.exists(input_path):
        raise Exception("File tidak ditemukan")

    doc = Document()

    with pdfplumber.open(input_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                doc.add_paragraph(text)

    output_id = str(uuid.uuid4())
    output_filename = f"{output_id}.docx"
    output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)

    doc.save(output_path)

    return {
        "file_id": output_id,
        "filename": output_filename
    }