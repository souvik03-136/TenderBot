import re
import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
from typing import List, Dict
from rapidfuzz import fuzz, process
from config import logger  # import logger from our config module

class TapasHandler:
    def __init__(self):
        print("Initializing TapasHandler...")
        logger.info("Initializing TapasHandler.")
        self.tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        self.model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")

    def parse_table(self, df: pd.DataFrame) -> List[Dict]:
        # Process a DataFrame using TAPAS with enhanced column handling
        print("Parsing table using TAPAS...")
        logger.info("Parsing table using TAPAS.")
        try:
            if df is None or df.empty or len(df.columns) < 2:
                logger.warning("DataFrame is None, empty, or has less than 2 columns.")
                return []

            item_col, make_col = self._identify_columns(df)
            if not item_col or not make_col:
                logger.warning("Required columns not identified.")
                return []

            return self._process_with_tapas(df[[item_col, make_col]])
        except Exception as e:
            logger.error(f"TAPAS processing failed: {str(e)}")
            print(f"Error in parse_table: {str(e)}")
            return []

    def _identify_columns(self, df: pd.DataFrame) -> tuple:
        # Identify columns using fuzzy matching
        print("Identifying columns in DataFrame...")
        logger.info("Identifying columns in DataFrame.")
        item_col = process.extractOne(
            'item', df.columns,
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
        logger.info(f"Identified columns - Item: {identified_item}, Manufacturer: {identified_make}")
        return (identified_item, identified_make)

    def _process_with_tapas(self, df: pd.DataFrame) -> List[Dict]:
        # Process identified columns with TAPAS
        print("Processing DataFrame with TAPAS...")
        logger.info("Processing DataFrame with TAPAS.")
        inputs = self.tokenizer(
            table=df,
            queries=["List items with their manufacturers"],
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        outputs = self.model(**inputs)
        results = self.tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach(),
            outputs.logits_aggregation.detach()
        )
        return self._format_results(results[0])

    def _format_results(self, raw_results) -> List[Dict]:
        # Format TAPAS raw results into structured data
        print("Formatting TAPAS results...")
        logger.info("Formatting TAPAS results.")
        processed = []
        for row in raw_results:
            if len(row) >= 2:
                item = str(row[0]).strip()
                makes = str(row[1]).strip()
                # Use a temporary marker to preserve M/s and split properly
                makes = re.sub(r'm/s', '[[M_S]]', makes, flags=re.IGNORECASE)
                parts = re.split(r'[,;/]|(?<=\))\s*', makes)
                make_list = [
                    re.sub(r'\s+', ' ', p.strip())
                    for p in parts
                    if p.strip() and re.search(r'[a-zA-Z]{3}', p)
                ]
                make_list = [p.replace("[[M_S]]", "M/s ") for p in make_list]
                if item and make_list:
                    processed.append({
                        "Item": item,
                        "Approved Makes": make_list
                    })
        logger.info("Finished formatting TAPAS results.")
        return processed
