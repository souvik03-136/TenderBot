import os
import json
from flask import Flask, request, jsonify
from concurrent.futures import ProcessPoolExecutor
from extractor import PDFTableExtractor
from config import logger

app = Flask(__name__)
RESULTS_FILE = "output.json"
extractor = PDFTableExtractor()


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"message": "PDF Extraction API is running"}), 200


@app.route("/extract", methods=["POST"])
def extract_from_directory():
    data = request.get_json()

    if not data or "directory" not in data:
        return jsonify({"error": "Missing 'directory' parameter"}), 400

    pdf_folder = data["directory"]

    if not os.path.exists(pdf_folder) or not os.path.isdir(pdf_folder):
        return jsonify({"error": f"Invalid directory: {pdf_folder}"}), 400

    try:
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
        logger.info(f"Found {len(pdf_files)} PDF file(s) in '{pdf_folder}'.")

        if not pdf_files:
            return jsonify({"message": "No PDFs found in the directory"}), 200

        output = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(extractor.process_pdf, os.path.join(pdf_folder, f))
                       for f in pdf_files]

            for future in futures:
                output.append(future.result())

        # Save results
        with open(RESULTS_FILE, "w") as f:
            json.dump(output, f, indent=2)

        return jsonify({"message": "Processing complete", "data": output}), 200

    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        return jsonify({"error": f"Error processing PDFs: {str(e)}"}), 500


@app.route("/results", methods=["GET"])
def get_results():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results available"}), 404

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    return jsonify({"message": "Results retrieved successfully", "data": data}), 200


if __name__ == "__main__":
    app.run(debug=True)
