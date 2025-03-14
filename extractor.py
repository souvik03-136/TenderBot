import os
import re
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import camelot
from typing import Optional, List, Dict
from config import logger
from exceptions import TableNotFoundError
from tapas_handler import TapasHandler

import cv2
import numpy as np
from PIL import Image
import ctypes
from ctypes.util import find_library

from gemini_handler import GeminiHandler
import tempfile


class PDFTableExtractor:
    def __init__(self):
        print("Initializing PDFTableExtractor...")
        logger.info("Initializing PDFTableExtractor.")
        # Regex to identify pages likely containing manufacturer tables.
        self.table_title_pattern = re.compile(
            r'\b(approved|authorized|standard|makes?|manufacturers?|vendors?|brands?|suppliers?)\b.*'
            r'\b(list|schedule|details|specification)\b',
            re.IGNORECASE | re.DOTALL
        )
        self.table_pattern = re.compile(r'(approved[\s-]*(makes?|manufacturers?))', re.IGNORECASE)
        self.item_pattern = re.compile(r'^\d+[\.\)]\s*')
        self.tapas = TapasHandler()
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Initialize Gemini for handling scanned PDFs.
        self.gemini = GeminiHandler(os.getenv("GEMINI_API_KEY"))
        self.gemini_prompt = """
Analyze the document and find the manufacturer list table with columns:
1. "DESCRIPTION OF ITEM" - Names of construction materials/components
2. "NAME OF MANUFACTURER" - Approved companies/brands

Handle these patterns:
- Multi-line manufacturer entries (e.g., "M/s ABC,
M/s XYZ")
- Alternate spellings of M/s (Ms, Ms., M/s.)
- Combine manufacturers split across pages

Return JSON format:
{
    "tables": [{
        "title": "LIST OF MANUFACTURERS",
        "data": [
            {
                "item": "Cement",
                "approved_make": ["UltraTech", "ACC", "Ambuja"]
            },
            ...
        ]
    }]
}
"""

    def _find_manufacturer_table(self, pdf_path: str) -> Optional[pd.DataFrame]:
        print(f"Finding manufacturer table in {pdf_path}...")
        logger.info(f"Finding manufacturer table in {pdf_path}...")
        try:
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if self.table_title_pattern.search(text):
                        logger.info("Found a potential table title on a page.")
                        table = page.extract_table()
                        if table and len(table) > 1:
                            df = self._clean_table(table)
                            tables.append(df)
            if tables:
                logger.info("Manufacturer tables extracted successfully.")
                return pd.concat(tables).dropna(how='all')
            else:
                logger.warning("No manufacturer tables found in the document.")
                return None
        except Exception as e:
            logger.error(f"Error locating table: {str(e)}")
            print(f"Error in _find_manufacturer_table: {str(e)}")
            return None

    def _clean_table(self, table_data: list) -> pd.DataFrame:
        print("Cleaning extracted table data...")
        logger.info("Cleaning table data.")
        # Normalize the header row.
        columns = [str(col).strip() if col is not None else '' for col in table_data[0]]
        df = pd.DataFrame(table_data[1:], columns=columns)
        if not df.empty:
            df = df.dropna(how='all').dropna(axis=1, how='all')
            first_col = df.columns[0]
            df[first_col] = df[first_col].apply(
                lambda x: re.sub(r'^\d+[\.\)]?\s*', '', str(x).strip())
                if pd.notnull(x) else ''
            )
            df = df[df[first_col].str.len() > 0]
        return df

    def _process_digital(self, pdf_path: str) -> List[Dict]:
        print("Processing digital PDF...")
        logger.info("Processing digital PDF.")
        try:
            table_df = self._find_manufacturer_table(pdf_path)
            if table_df is not None:
                direct_results = self._direct_parse(table_df)
                if self._validate_results(direct_results):
                    return direct_results

            logger.info("Falling back to TAPAS for digital PDF processing.")
            tapas_results = self.tapas.parse_table(table_df)
            if self._validate_results(tapas_results):
                return tapas_results

            logger.info("Falling back to Camelot for digital PDF processing.")
            return self._camelot_fallback(pdf_path)
        except Exception as e:
            logger.error(f"Digital processing failed: {str(e)}")
            print(f"Error in _process_digital: {str(e)}")
            return []

    def _direct_parse(self, df: pd.DataFrame) -> List[Dict]:
        print("Directly parsing table data...")
        logger.info("Directly parsing table data.")
        results = []
        item_col, make_col = self._identify_columns(df)
        if not item_col or not make_col:
            logger.warning("Could not identify required columns for direct parsing.")
            return []
        for _, row in df.iterrows():
            item = str(row[item_col]).strip()
            if re.match(r'^\d+$', item):
                continue
            # Normalize manufacturer names.
            makes = re.sub(r'\n', ', ', str(row[make_col]))
            makes = re.sub(r'm/s', '[[M_S]]', makes, flags=re.IGNORECASE)
            parts = re.split(r'[,;/]', makes)
            make_list = [p.strip().replace("[[M_S]]", "M/s") for p in parts if p.strip()]
            if item and make_list:
                results.append({
                    "item": item,
                    "approved_make": ", ".join(make_list)
                })
        logger.info("Direct parsing completed.")
        return results

    def _parse_text_data(self, text: str) -> List[Dict]:
        print("Parsing OCR text data...")
        logger.info("Parsing OCR text data from image conversion.")
        results = []
        current_item = None
        manufacturers = []
        for line in text.split('\n'):
            line = re.sub(r'\s+', ' ', line).strip()
            if not line:
                continue
            if self.table_pattern.search(line):
                continue
            if re.match(r'^\d+[\.\)]', line):
                if current_item and manufacturers:
                    results.append({
                        "item": current_item,
                        "approved_make": ", ".join(manufacturers)
                    })
                current_item = re.sub(self.item_pattern, '', line).strip()
                manufacturers = []
            elif current_item:
                line = re.sub(r'm/s', '[[M_S]]', line, flags=re.IGNORECASE)
                parts = re.split(r'[,;/]', line)
                manufacturers.extend([p.strip().replace("[[M_S]]", "M/s") for p in parts if p.strip()])
        if current_item and manufacturers:
            results.append({
                "item": current_item,
                "approved_make": ", ".join(manufacturers)
            })
        logger.info("Finished parsing OCR text data.")
        return results

    def _identify_columns(self, df: pd.DataFrame) -> tuple:
        print("Identifying columns for parsing...")
        logger.info("Identifying columns for direct parsing.")
        column_aliases = {
            'description': {'item', 'material', 'description', 'product'},
            'manufacturer': {'make', 'manufacturer', 'brand', 'supplier'}
        }
        best_match = {'description': (None, 0), 'manufacturer': (None, 0)}
        for col in df.columns:
            col_lower = str(col).lower()
            desc_score = len(column_aliases['description'].intersection(col_lower.split()))
            if desc_score > best_match['description'][1]:
                best_match['description'] = (col, desc_score)
            manu_score = len(column_aliases['manufacturer'].intersection(col_lower.split()))
            if manu_score > best_match['manufacturer'][1]:
                best_match['manufacturer'] = (col, manu_score)
        logger.info(f"Identified columns - Description: {best_match['description'][0]}, Manufacturer: {best_match['manufacturer'][0]}")
        return (best_match['description'][0], best_match['manufacturer'][0])

    def _validate_results(self, data: List[Dict]) -> bool:
        # Check that each entry contains both an item and its corresponding manufacturer(s)
        return len([item for item in data if item.get('item') and item.get('approved_make')]) > 0

    def _check_ghostscript_installed(self) -> bool:
        # Verify that Ghostscript is installed (needed for Camelot on Windows).
        try:
            arch = ctypes.sizeof(ctypes.c_voidp) * 8
            lib_name = "gsdll{}{}.dll".format("", arch)
            gs_path = find_library(lib_name)
            if gs_path:
                logger.info(f"Ghostscript found: {gs_path}")
                return True
            else:
                logger.error("Ghostscript library not found.")
                return False
        except Exception as e:
            logger.error(f"Error checking Ghostscript installation: {str(e)}")
            return False

    def _camelot_fallback(self, pdf_path: str) -> List[Dict]:
        print("Using Camelot as a fallback method...")
        logger.info("Using Camelot fallback for table extraction.")
        if not self._check_ghostscript_installed():
            error_message = ("Ghostscript is not installed. "
                             "Please install it following the instructions at: "
                             "https://camelot-py.readthedocs.io/en/master/user/install-deps.html")
            logger.error(error_message)
            print("Error in _camelot_fallback:", error_message)
            return []
        try:
            tables = camelot.read_pdf(pdf_path, pages='all')
            results = []
            for table in tables:
                df = table.df
                df.columns = df.iloc[0]  # Use the first row as header.
                df = df[1:]
                direct_results = self._direct_parse(df)
                if self._validate_results(direct_results):
                    results.extend(direct_results)
            logger.info("Camelot fallback extraction successful.")
            return results
        except Exception as e:
            logger.error(f"Camelot fallback failed: {str(e)}")
            print(f"Error in _camelot_fallback: {str(e)}")
            return []

    def _is_scanned(self, pdf_path: str) -> bool:
        print("Checking if the PDF is scanned...")
        logger.info("Determining if PDF contains searchable text.")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        logger.info("PDF contains searchable text.")
                        return False
            logger.info("No searchable text found; PDF appears to be scanned.")
            return True
        except Exception as e:
            logger.error(f"Error checking if PDF is scanned: {str(e)}")
            print(f"Error in _is_scanned: {str(e)}")
            return True

    def _preprocess_image(self, image):
        # Convert image to grayscale and apply thresholding for better OCR results.
        cv_image = np.array(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return Image.fromarray(thresh)

    def _process_scanned(self, pdf_path: str) -> List[Dict]:
        print("Processing scanned PDF with Gemini...")
        logger.info("Processing scanned PDF using Gemini integration.")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Converting PDF pages to images in {temp_dir}")
                logger.info(f"Converting PDF pages to images in temporary directory: {temp_dir}")
                images = convert_from_path(pdf_path, output_folder=temp_dir)
                results = []
                for i, image in enumerate(images):
                    img_path = os.path.join(temp_dir, f"page_{i + 1}.png")
                    image.save(img_path, "PNG")
                    print(f"Saved page {i + 1} as image: {img_path}")
                    logger.info(f"Saved page {i + 1} as image: {img_path}")
                    print("Sending image to Gemini for table extraction...")
                    logger.info(f"Sending image {img_path} to Gemini.")
                    gemini_result = self.gemini.extract_table_from_image(img_path, self.gemini_prompt)
                    print(f"Gemini result for page {i + 1}: {gemini_result}")
                    logger.info(f"Received Gemini result for page {i + 1}: {gemini_result}")
                    if gemini_result and 'tables' in gemini_result:
                        results.extend(gemini_result['tables'])
                    else:
                        print(f"Warning: Gemini did not return expected table data for page {i + 1}.")
                        logger.warning(f"Gemini did not return expected 'tables' for page {i + 1}.")
                merged_results = self._merge_results(results)
                print("Merged results from Gemini:", merged_results)
                logger.info(f"Merged Gemini results: {merged_results}")
                return merged_results
        except Exception as e:
            logger.error(f"Gemini processing failed: {str(e)}")
            print(f"Error in _process_scanned: {str(e)}")
            return self._fallback_ocr_processing(pdf_path)

    def _fallback_ocr_processing(self, pdf_path: str) -> List[Dict]:
        logger.info("Using OCR as a fallback for scanned PDF processing.")
        try:
            images = convert_from_path(pdf_path)
            full_text = ""
            for image in images:
                processed_image = self._preprocess_image(image)
                text = pytesseract.image_to_string(processed_image)
                full_text += text + "\n"
            return self._parse_text_data(full_text)
        except Exception as e:
            logger.error(f"OCR fallback failed: {str(e)}")
            print(f"Error in _fallback_ocr_processing: {str(e)}")
            return []

    def _merge_results(self, results) -> List[Dict]:
        # Merge table data from multiple pages by combining manufacturer lists and removing duplicates.
        merged = {}
        for table in results:
            for entry in table.get('data', []):
                item = entry['item'].strip()
                makes = [m.strip() for m in entry['approved_make'].split(',')]
                if item in merged:
                    merged[item]['makes'].extend(makes)
                    merged[item]['makes'] = list(set(merged[item]['makes']))
                else:
                    merged[item] = {'item': item, 'makes': list(set(makes))}
        return [v for v in merged.values()]

    def process_pdf(self, pdf_path: str) -> Dict:
        print(f"Processing PDF: {pdf_path}")
        logger.info(f"Processing PDF: {pdf_path}")
        result = {
            "file_name": os.path.basename(pdf_path),
            "tables": [],
            "status": "success"
        }
        try:
            if self._is_scanned(pdf_path):
                data = self._process_scanned(pdf_path)
            else:
                data = self._process_digital(pdf_path)
            if not self._validate_results(data):
                raise TableNotFoundError("No valid manufacturer data found")
            result["tables"].append({
                "title": "LIST OF MANUFACTURERS",
                "data": data
            })
        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"PDF processing failed: {str(e)}")
            print(f"Error in process_pdf: {str(e)}")
        return result
