import os
import json
from concurrent.futures import ProcessPoolExecutor
from extractor import PDFTableExtractor
from config import logger


def main():
    print("Starting PDF extraction process...")
    logger.info("Starting PDF extraction process...")
    extractor = PDFTableExtractor()
    output = []
    pdf_folder = "Pdf"

    try:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF file(s) in folder '{pdf_folder}'.")
    except Exception as e:
        logger.error(f"Error accessing folder {pdf_folder}: {str(e)}")
        print(f"Error accessing folder {pdf_folder}: {str(e)}")
        return

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extractor.process_pdf, os.path.join(pdf_folder, f))
                   for f in pdf_files]
        for future in futures:
            output.append(future.result())

    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Processing complete. Results saved to output.json")
    print("Processing complete. Results saved to output.json")


if __name__ == "__main__":
    main()
