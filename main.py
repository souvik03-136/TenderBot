import os
import json
import re
import logging
import fitz
import pdfplumber
import pytesseract
import pandas as pd
import numpy as np
import cv2
from pdf2image import convert_from_path
from transformers import pipeline
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import fuzz, process
import camelot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFTableExtractor:
    def __init__(self):
        self.table_title_pattern = re.compile(
            r'(approved[\s-]*(?:makes?|manufacturers?|vendors?|suppliers?)|'
            r'(allowed|authorized)[\s-]+(?:makes?|brands?)|'
            r'list\s+of\s+(?:approved|authorized)?[\s-]*(?:makes?|manufacturers?|vendors?))',
            re.IGNORECASE
        )
        try:
            self.tapas = pipeline("table-question-answering",
                                  model="google/tapas-base-finetuned-wtq")
        except Exception as e:
            logger.error(f"Failed to initialize TAPAS model: {str(e)}")
            self.tapas = None

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        logger.info("Tesseract configured successfully")

    def detect_pdf_type(self, pdf_path: str) -> str:
        """Improved hybrid PDF detection"""
        try:
            text_content = ""
            with fitz.open(pdf_path) as doc:
                for page in doc.pages(0, 2):  # Check first 3 pages
                    text_content += page.get_text().lower()

            if self.table_title_pattern.search(text_content):
                return "digital"  # Force digital processing if table title exists

            # Fallback to original detection
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0]
                return "digital" if len(first_page.chars) > 50 else "scanned"

        except Exception as e:
            logger.error(f"PDF detection failed: {str(e)}")
            return "unknown"

    def digital_extraction_strategy(self, pdf_path: str) -> Optional[pd.DataFrame]:
        """Add detailed logging for extraction process"""
        strategies = [
            self._extract_with_pymupdf,
            self._extract_with_pdfplumber,
            self._extract_with_camelot
        ]

        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(strategy, pdf_path): strategy for strategy in strategies}
            for future in futures:
                try:
                    result = future.result(timeout=45)
                    if result is not None:
                        logger.debug(f"Raw extraction result from {futures[future].__name__}:\n{result.head(2)}")
                        if self._is_valid_table(result):
                            logger.info(f"Valid table found via {futures[future].__name__}")
                            return result
                        else:
                            logger.debug(f"Table from {futures[future].__name__} failed validation")
                except Exception as e:
                    logger.debug(f"Extraction failed: {str(e)}")
        return None

    def _extract_with_camelot(self, pdf_path: str) -> Optional[pd.DataFrame]:
        """Enhanced Camelot extraction with improved settings"""
        try:
            tables = camelot.read_pdf(
                pdf_path,
                flavor='stream',
                pages='1-5',
                strip_text='\n',
                suppress_stdout=True,
                row_tol=15,
                edge_tol=30,
                columns=['50,200,400'],  # Adjust based on typical column positions
                split_text=True
            )

            best_table = None
            for table in tables:
                if table.parsing_report['accuracy'] > 60:
                    # Normalize column headers
                    headers = [str(cell).lower().strip() for cell in table.cells[0]]
                    if any('manufacturer' in h for h in headers) and any('item' in h for h in headers):
                        df = table.df
                        df.columns = df.iloc[0]  # Use first row as header
                        return df[1:].reset_index(drop=True)

            return best_table
        except Exception as e:
            logger.debug(f"Camelot error: {str(e)}")
        return None

    def _extract_with_pdfplumber(self, pdf_path: str) -> Optional[pd.DataFrame]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:  # Iterate through all pages
                    table = page.extract_table()
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        logger.debug(f"PDFPlumber raw extract:\n{df.head(2)}")
                        if self._is_valid_table(df):
                            return df
            return None
        except Exception as e:
            logger.debug(f"PDFPlumber error: {str(e)}")
        return None

    def _extract_with_pymupdf(self, pdf_path: str) -> Optional[pd.DataFrame]:
        """Improved table detection with header verification"""
        try:
            doc = fitz.open(pdf_path)
            for page in doc.pages(0, 4):  # Check first 5 pages
                text = page.get_text("text", flags=fitz.TEXT_PRESERVE_IMAGES).lower()
                if self.table_title_pattern.search(text):
                    tables = page.find_tables()
                    if tables:
                        df = tables[0].to_pandas()
                        if self._is_valid_table(df):
                            return df
                        else:
                            logger.debug("PyMuPDF table found but failed validation")
            return None
        except Exception as e:
            logger.debug(f"PyMuPDF error: {str(e)}")
        return None



    def process_scanned_pdf(self, pdf_path: str) -> Optional[List[Dict]]:
        """Optimized OCR processing with resource limits"""
        try:
            images = convert_from_path(
                pdf_path,
                dpi=200,
                first_page=1,
                last_page=5,
                thread_count=2,
                fmt='jpeg',
                use_pdftocairo=True
            )

            for img in images:
                img.thumbnail((1240, 1754))  # Reduce size for processing
                processed = self._preprocess_image(img)

                text = pytesseract.image_to_string(
                    processed,
                    config='--psm 6',
                    output_type=pytesseract.Output.DICT
                )['text']

                if self.table_title_pattern.search(text):
                    return self._parse_ocr_text(text)

            return None
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            return None

    def _preprocess_image(self, img):
        """Fast preprocessing pipeline"""
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def _parse_ocr_text(self, text: str) -> List[Dict]:
        """Improved parsing for manufacturer lists"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        table_data = []

        # Find table start using flexible matching
        start_idx = next((i for i, line in enumerate(lines)
                          if re.search(r'(manufacturer|supplier|brand)\b', line, re.I)), -1)

        if start_idx == -1:
            return []

        # Extract relevant lines using enhanced pattern
        pattern = re.compile(r'(\d+)\s+(.*?)\s{2,}(.*?)$')
        for line in lines[start_idx + 1:]:
            if re.search(r'page \d+|cont\.?$', line, re.I):
                break
            match = pattern.match(line)
            if match:
                table_data.append({
                    "item": match.group(2).strip(),
                    "manufacturer": match.group(3).strip()
                })

        return table_data

    def _is_valid_table(self, df: pd.DataFrame) -> bool:
        """Improved table validation with flexible column matching"""
        if df.empty or len(df.columns) < 2:
            return False

        required_keywords = {
            'item': ['item', 'desc', 'material', 'product', 'description'],
            'manufacturer': ['manufacturer', 'maker', 'supplier', 'brand', 'vendor', 'name of manufacturer']
        }

        found = {'item': False, 'manufacturer': False}
        for col in df.columns.str.lower():
            for key, keywords in required_keywords.items():
                if any(fuzz.partial_ratio(kw, col) > 75 for kw in keywords):
                    found[key] = True

        logger.debug(f"Column validation - Item: {found['item']}, Manufacturer: {found['manufacturer']}")
        return found['item'] and found['manufacturer']

    def process_pdf(self, pdf_path: str) -> Dict:
        """Memory-optimized processing pipeline"""
        result = {
            "file_name": os.path.basename(pdf_path),
            "tables": [],
            "status": "success"
        }

        doc = None  # Initialize to prevent "Unresolved reference"
        pdf = None  # Initialize to prevent "Unresolved reference"

        try:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Processing: {result['file_name']}")

            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError("Not a PDF file")

            pdf_type = self.detect_pdf_type(pdf_path)
            logger.info(f"PDF type: {pdf_type}")

            table_data = None
            if pdf_type == "digital":
                table_data = self.digital_extraction_strategy(pdf_path)
            elif pdf_type == "scanned":
                table_data = self.process_scanned_pdf(pdf_path)
            else:
                raise ValueError("Unsupported PDF format")

            if table_data is not None and not table_data.empty if isinstance(table_data, pd.DataFrame) else table_data:
                result["tables"].append({
                    "title": "Approved Makes and Manufacturers",
                    "data": table_data.to_dict(orient='records') if isinstance(table_data, pd.DataFrame) else table_data
                })
            else:
                raise ValueError("No valid table found")

        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"Processing failed: {str(e)}")

        # Cleanup resources
        if doc is not None:
            doc.close()
        if pdf is not None:
            pdf.close()

        logger.info(f"Processing completed: {result['status']}")
        return result


if __name__ == "__main__":
    extractor = PDFTableExtractor()
    output = []

    pdf_folder = "Pdf"
    if not os.path.exists(pdf_folder):
        logger.error(f"PDF folder '{pdf_folder}' not found")
        exit(1)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files")

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        try:
            result = extractor.process_pdf(pdf_path)
            output.append(result)

            # Write results incrementally
            with open("output.json", "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Fatal error processing {filename}: {str(e)}")

    logger.info(f"Processing complete. Results saved to output.json")