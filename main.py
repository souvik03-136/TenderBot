import os
import json
import re
import logging
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import pandas as pd
import camelot
from pdf2image import convert_from_path
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('extraction.log'), logging.StreamHandler()]
)


class PDFTableExtractor:
    def __init__(self):
        self.table_title_variants = re.compile(
            r'(approved[\s-]*makes?|approved[\s-]*manufacturers?)',
            re.IGNORECASE
        )

        # Set Tesseract path if needed (check your installation)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def detect_pdf_type(self, pdf_path):
        """Detect if PDF is text-based or scanned."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0]
                if len(first_page.chars) > 0:  # Check for text characters
                    return "digital"
            return "scanned"
        except Exception as e:
            logging.error(f"PDF detection failed: {str(e)}")
            return "unknown"

    def extract_digital_table(self, pdf_path):
        """Extract tables from digital PDFs using multiple methods."""
        try:
            # Try Camelot first
            tables = camelot.read_pdf(pdf_path, flavor="lattice", pages="all")
            for table in tables:
                if self._is_target_table(table.df):
                    return self._clean_table(table.df)

            # Fallback to pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table and self._is_target_table(pd.DataFrame(table)):
                        return self._clean_table(pd.DataFrame(table))

            logging.warning(f"No valid table found in digital PDF: {pdf_path}")
            return None
        except Exception as e:
            logging.error(f"Digital extraction failed: {str(e)}")
            return None

    def extract_scanned_table(self, pdf_path):
        """Extract tables from scanned PDFs using enhanced OCR."""
        try:
            images = convert_from_path(pdf_path, dpi=300)
            all_text = []

            for img in images:
                # Preprocess image
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # OCR with improved configuration
                text = pytesseract.image_to_string(
                    thresh,
                    config='--psm 6 -c preserve_interword_spaces=1',
                    output_type=pytesseract.Output.DICT
                )['text']

                all_text.append(text)

            return self._parse_ocr_text("\n".join(all_text))
        except Exception as e:
            logging.error(f"Scanned PDF processing failed: {str(e)}")
            return None

    def _parse_ocr_text(self, text):
        """Improved OCR text to table conversion with alignment detection."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        table_data = []
        current_row = []
        prev_positions = []

        for line in lines:
            # Find text elements with their positions
            words = pytesseract.image_to_data(line, output_type=pytesseract.Output.DICT)
            x_positions = [int(x) for x, word in zip(words['left'], words['text']) if word.strip()]

            if not x_positions:
                continue

            # Detect column positions
            if not prev_positions:
                prev_positions = x_positions
                current_row = [line]
            else:
                # Check if aligned with previous columns
                if abs(x_positions[0] - prev_positions[0]) < 50:
                    table_data.append(current_row)
                    current_row = [line]
                    prev_positions = x_positions
                else:
                    current_row.append(line)

        if current_row:
            table_data.append(current_row)

        if not table_data:
            return None

        # Convert to DataFrame for validation
        df = pd.DataFrame(table_data)
        if self._is_target_table(df):
            return self._clean_table(df)
        return None

    def _is_target_table(self, df):
        """Enhanced table validation with multiple checks."""
        # Check first 3 rows for header pattern
        for i in range(min(3, len(df))):
            header_text = ' '.join(df.iloc[i].astype(str)).lower()
            if self.table_title_variants.search(header_text):
                return True
        return False

    def _clean_table(self, df):
        """Improved table cleaning with error handling."""
        try:
            # Find header row
            header_idx = next(
                i for i, row in df.iterrows()
                if self.table_title_variants.search(' '.join(row.astype(str)).lower())
            )

            # Clean headers
            df.columns = df.iloc[header_idx].apply(lambda x: str(x).lower().strip())
            cleaned = df.drop(index=range(header_idx + 1)).reset_index(drop=True)

            # Select relevant columns
            columns = [col for col in cleaned.columns
                       if any(kw in col for kw in ['item', 'manufacturer', 'make'])]

            if len(columns) < 2:
                raise ValueError("Insufficient columns found")

            return cleaned[columns].to_dict('records')

        except (StopIteration, ValueError, KeyError) as e:
            logging.warning(f"Table cleaning failed: {str(e)}")
            return None

    def process_pdf(self, pdf_path):
        """Processing pipeline with enhanced error handling."""
        try:
            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError("Not a PDF file")

            logging.info(f"Processing file: {pdf_path}")
            doc_type = self.detect_pdf_type(pdf_path)
            table_data = None

            if doc_type == "digital":
                table_data = self.extract_digital_table(pdf_path)
            elif doc_type == "scanned":
                table_data = self.extract_scanned_table(pdf_path)

            if not table_data:
                raise ValueError("No valid table found in document")

            return {
                "file": os.path.basename(pdf_path),
                "approved_makes": table_data,
                "status": "success"
            }

        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            return {
                "file": os.path.basename(pdf_path),
                "error": str(e),
                "status": "failed"
            }


if __name__ == "__main__":
    extractor = PDFTableExtractor()
    output = []

    pdf_folder = "Pdfs"
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder '{pdf_folder}' not found")

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            result = extractor.process_pdf(pdf_path)
            output.append(result)

    with open("output.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info("Processing complete. Results saved to output.json")