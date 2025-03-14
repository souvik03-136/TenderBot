import os
import re
import json
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
import camelot
from rapidfuzz import fuzz, process
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor
from config import logger
from exceptions import TableNotFoundError
from tapas_handler import TapasHandler

# For image pre-processing
import cv2
import numpy as np
from PIL import Image
import ctypes
from ctypes.util import find_library

class PDFTableExtractor:
    def __init__(self):
        print("Initializing PDFTableExtractor...")
        logger.info("Initializing PDFTableExtractor.")
        self.table_title_pattern = re.compile(
            r'(approved[\s-]*(makes?|manufacturers?)|'
            r'(list\s+of\s+(approved\s+)?manufacturers?))',
            re.IGNORECASE
        )
        self.table_pattern = re.compile(r'(approved[\s-]*(makes?|manufacturers?))', re.IGNORECASE)
        self.item_pattern = re.compile(r'^\d+[\.\)]\s*')
        self.tapas = TapasHandler()
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def _find_manufacturer_table(self, pdf_path: str) -> Optional[pd.DataFrame]:
        print(f"Finding manufacturer table in {pdf_path}...")
        logger.info(f"Finding manufacturer table in {pdf_path}...")
        try:
            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if self.table_title_pattern.search(text):
                        logger.info("Table title found on a page.")
                        table = page.extract_table()
                        if table and len(table) > 1:
                            df = self._clean_table(table)
                            tables.append(df)
            if tables:
                logger.info("Manufacturer tables successfully extracted.")
                return pd.concat(tables).dropna(how='all')
            else:
                logger.warning("No tables found.")
                return None

        except Exception as e:
            logger.error(f"Table location failed: {str(e)}")
            print(f"Error in _find_manufacturer_table: {str(e)}")
            return None

    def _clean_table(self, table_data: list) -> pd.DataFrame:
        print("Cleaning extracted table...")
        logger.info("Cleaning extracted table.")
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        first_col = df.columns[0]
        df[first_col] = df[first_col].apply(lambda x: re.sub(r'^\d+[\.\)]?\s*', '', str(x)))
        df = df[~df[first_col].str.match(r'^\d+$')]
        return df.dropna(how='all').dropna(axis=1, how='all')

    def _process_digital(self, pdf_path: str) -> List[Dict]:
        print("Processing digital PDF...")
        logger.info("Processing digital PDF.")
        try:
            table_df = self._find_manufacturer_table(pdf_path)
            if table_df is not None:
                direct_results = self._direct_parse(table_df)
                if self._validate_results(direct_results):
                    return direct_results

            # Fallback to TAPAS
            logger.info("Falling back to TAPAS processing.")
            tapas_results = self.tapas.parse_table(table_df)
            if self._validate_results(tapas_results):
                return tapas_results

            # Final fallback to Camelot
            logger.info("Falling back to Camelot processing.")
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
            logger.warning("Direct parse could not identify required columns.")
            return []

        for _, row in df.iterrows():
            item = str(row[item_col]).strip()
            if re.match(r'^\d+$', item):
                continue
            makes = str(row[make_col]).strip()
            makes = re.sub(r'm/s', '[[M_S]]', makes, flags=re.IGNORECASE)
            parts = re.split(r'[,;/]', makes)
            make_list = [
                p.strip().replace("[[M_S]]", "M/s").replace("\n", " ")
                for p in parts if p.strip()
            ]
            if item and make_list:
                results.append({
                    "Item": item,
                    "Approved Makes": make_list
                })
        logger.info("Direct parse completed.")
        return results

    def _parse_text_data(self, text: str) -> List[Dict]:
        print("Parsing OCR text data...")
        logger.info("Parsing OCR text data.")
        results = []
        current_item = None
        manufacturers = []

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            if self.table_pattern.search(line):
                continue

            if re.match(r'^\d+[\.\)]', line):
                if current_item and manufacturers:
                    results.append({
                        "Item": current_item,
                        "Approved Makes": manufacturers
                    })
                current_item = re.sub(self.item_pattern, '', line).strip()
                manufacturers = []
            elif current_item:
                line = re.sub(r'm/s', '[[M_S]]', line, flags=re.IGNORECASE)
                parts = re.split(r'[,;/]', line)
                manufacturers.extend([
                    p.strip().replace("[[M_S]]", "M/s").replace("\n", " ")
                    for p in parts if p.strip()
                ])

        if current_item and manufacturers:
            results.append({
                "Item": current_item,
                "Approved Makes": manufacturers
            })

        logger.info("OCR text data parsing completed.")
        return results

    def _identify_columns(self, df: pd.DataFrame) -> tuple:
        print("Identifying columns for direct parsing...")
        logger.info("Identifying columns for direct parsing.")
        item_col = process.extractOne(
            'description', df.columns,
            scorer=fuzz.token_set_ratio,
            score_cutoff=40
        )
        make_col = process.extractOne(
            'manufacturer', df.columns,
            scorer=fuzz.token_set_ratio,
            score_cutoff=40
        )
        identified_item = item_col[0] if item_col else None
        identified_make = make_col[0] if make_col else None
        logger.info(f"Identified columns - Description: {identified_item}, Manufacturer: {identified_make}")
        return (identified_item, identified_make)

    def _validate_results(self, data: List[Dict]) -> bool:
        valid = bool(data) and all(
            isinstance(item.get('Item'), str) and len(item.get('Approved Makes', [])) > 0
            for item in data
        )
        logger.info(f"Validation of results: {'passed' if valid else 'failed'}.")
        return valid

    def _check_ghostscript_installed(self) -> bool:
        """
        Check if Ghostscript is installed by trying to locate the Ghostscript DLL.
        This check is platform-dependent. For Windows, it looks for gsdll32.dll or gsdll64.dll.
        """
        try:
            arch = ctypes.sizeof(ctypes.c_voidp) * 8
            lib_name = "gsdll{}{}.dll".format("", arch)  # e.g. gsdll64.dll on 64-bit
            gs_path = find_library(lib_name)
            if gs_path:
                logger.info(f"Found Ghostscript library: {gs_path}")
                return True
            else:
                logger.error("Ghostscript library not found.")
                return False
        except Exception as e:
            logger.error(f"Error checking Ghostscript installation: {str(e)}")
            return False

    def _camelot_fallback(self, pdf_path: str) -> List[Dict]:
        print("Using Camelot fallback...")
        logger.info("Using Camelot fallback...")

        # Check if Ghostscript is installed before attempting Camelot parsing
        if not self._check_ghostscript_installed():
            error_message = ("Ghostscript is not installed. "
                             "You can install it using the instructions here: "
                             "https://camelot-py.readthedocs.io/en/master/user/install-deps.html")
            logger.error(error_message)
            print("Error in _camelot_fallback:", error_message)
            return []

        try:
            tables = camelot.read_pdf(pdf_path, pages='all')
            results = []
            for table in tables:
                df = table.df
                df.columns = df.iloc[0]  # Assume first row as header
                df = df[1:]
                direct_results = self._direct_parse(df)
                if self._validate_results(direct_results):
                    results.extend(direct_results)
            logger.info("Camelot fallback completed.")
            return results
        except Exception as e:
            logger.error(f"Camelot fallback failed: {str(e)}")
            print(f"Error in _camelot_fallback: {str(e)}")
            return []

    def _is_scanned(self, pdf_path: str) -> bool:
        print("Checking if PDF is scanned...")
        logger.info("Checking if PDF is scanned...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        logger.info("PDF is not scanned (contains text).")
                        return False
            logger.info("PDF appears to be scanned (no text found).")
            return True
        except Exception as e:
            logger.error(f"Scanning check failed: {str(e)}")
            print(f"Error in _is_scanned: {str(e)}")
            return True

    def _preprocess_image(self, image):
        """
        Preprocess a PIL image for better OCR accuracy:
        - Convert to grayscale.
        - Apply thresholding to obtain a binary image.
        """
        # Convert PIL Image to numpy array
        cv_image = np.array(image)
        # Convert RGB to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        # Apply thresholding using Otsu's method
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def _process_scanned(self, pdf_path: str) -> List[Dict]:
        print("Processing scanned PDF using OCR...")
        logger.info("Processing scanned PDF using OCR.")
        try:
            images = convert_from_path(pdf_path)
            full_text = ""
            for image in images:
                # Preprocess the image before OCR
                processed_image = self._preprocess_image(image)
                # Convert the processed image (numpy array) back to a PIL Image
                pil_image = Image.fromarray(processed_image)
                text = pytesseract.image_to_string(pil_image)
                full_text += text + "\n"
            return self._parse_text_data(full_text)
        except Exception as e:
            logger.error(f"Scanned processing failed: {str(e)}")
            print(f"Error in _process_scanned: {str(e)}")
            return []

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
                "title": "Approved Makes and Manufacturers",
                "data": data
            })

        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"Processing failed: {str(e)}")
            print(f"Error in process_pdf: {str(e)}")

        return result
