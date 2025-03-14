# TenderBot Task

This project is designed to extract manufacturer table data from PDF documents. It supports both digital PDFs (with embedded text) and scanned PDFs (using OCR and image-based processing). The extraction logic leverages several open-source libraries and machine learning models, including:

- **pdfplumber, Camelot, pdf2image:** For PDF parsing and table extraction.
- **pytesseract & OpenCV:** For image preprocessing and OCR on scanned PDFs.
- **Google TAPAS:** For table question answering on extracted table data.
- **Google Gemini:** For extracting tables from images when the PDF is scanned.
- **Flask:** To expose the extraction functionality as a web API.

The final results are output as JSON and saved in an `output.json` file.

## Project Structure

- **`PDFTableExtractor`**  
  This class handles the extraction process:
  - **Digital PDF Processing:** Uses `pdfplumber` to extract text and tables, cleans the data, and applies direct parsing. If the direct approach fails, it falls back to using TAPAS (via the `TapasHandler`).
  - **Scanned PDF Processing:** Converts PDF pages to images using `pdf2image`, preprocesses the images with OpenCV, and then sends them to Google Gemini via the `GeminiHandler` for table extraction. If Gemini fails, it falls back to OCR processing with Tesseract.

- **`TapasHandler`**  
  This class uses the pre-trained [TAPAS model](https://huggingface.co/google/tapas-base-finetuned-wtq) to process tables. It identifies the relevant columns using fuzzy matching and processes the table data to output structured JSON.

- **`GeminiHandler`**  
  This class interacts with the Google Gemini API to extract tables from images. It sends image data along with a prompt that describes the expected table format, parses the Gemini response, and converts it into a JSON structure.

- **`Flask API`**  
  The project includes a Flask server that listens for API requests. When a POST request is made to the `/extract` endpoint, the API processes the PDFs in the given directory and returns the extraction result in JSON format (also saved to `output.json`).

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/souvik03-136/TenderBot.git
   cd TenderBot
   ```

2. **Install dependencies:**

   Make sure you have Python 3.7+ installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt --use-deprecated=legacy-resolver          
   ```

   The `requirements.txt` includes libraries like `pdfplumber`, `camelot-py`, `pytesseract`, `transformers`, `rapidfuzz`, `opencv-python`, and others.

3. **Environment Variables:**

   Create a `.env` file in the project root and add your Gemini API key:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Configure Tesseract:**

   Ensure that [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) is installed on your system. Update the Tesseract command path in the code (if needed):

   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

5. **Run the Flask API Server:**

   Start the API server by running:

   ```bash
   python main.py
   ```

 
### How to Use the API

You can call the API using Postman, cURL, or any HTTP client. The API expects a JSON body with the key `"directory"` pointing to the folder containing the PDFs to process.

#### Using Postman

1. **Open Postman** and create a new request.
2. **Set the Request Method and URL:**  
   - Choose `POST` as the request method.
   - Enter the URL: `http://localhost:5000/extract`
3. **Configure the Headers:**  
   - Add a header with the key `Content-Type` and the value `application/json`.
4. **Create the Request Body:**  
   - Navigate to the **Body** tab.
   - Select the **raw** option.
   - Choose **JSON** from the dropdown.
   - Enter the following JSON (update the directory path as needed):

   ```json
   {
       "directory": "C:\\Users\\souvi\\OneDrive\\Documents\\TenderBot\\Pdf"
   }
   ```

5. **Send the Request:**  
   - Click the **Send** button.
   - The API will process the PDFs in the specified directory and return the extraction results in JSON format. The output is also saved to an `output.json` file in the project directory.

#### Using cURL

You can also test the API using the following cURL command:

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"directory\":\"C:\\\\Users\\\\souvi\\\\OneDrive\\\\Documents\\\\TenderBot\\\\Pdf\"}" http://localhost:5000/extract
```

Both methods will trigger the PDF processing workflow, which involves:
- Checking if each PDF is digital or scanned.
- Extracting tables using libraries such as pdfplumber, Camelot, TAPAS, and Gemini.
- Returning the extracted manufacturer table data in JSON format.

The final extraction result is available in the API response as well as in an `output.json` file in your project directory.
## Output

The JSON result includes:
- **`file_name`**: The name of the processed PDF.
- **`tables`**: An array of extracted tables. Each table object contains a title (e.g., "LIST OF MANUFACTURERS") and the data extracted.
- **`status`**: A status message indicating whether the extraction was successful or if it failed (along with an error message).

## Additional Information

- **Error Handling:**  
  The project logs errors using a custom logger and prints messages to the console. In case of any extraction failures (e.g., if no valid manufacturer data is found), the status in the JSON response is updated to `"failed"` with the corresponding error message.

- **Extensibility:**  
  The modular design allows you to easily extend or replace components (for example, updating the Gemini prompt or adding new methods for table detection).

- **Logging:**  
  All major steps and errors are logged in `extraction.log` as well as printed to the console for debugging purposes.

## Conclusion

This project automates the extraction of manufacturer table data from various PDF formats using a combination of traditional PDF parsing libraries and modern machine learning models. By integrating multiple approaches, it ensures robust handling of both digital and scanned PDFs, providing a comprehensive solution for your table extraction needs.

