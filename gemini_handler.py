from google import genai
from config import logger
import json
import mimetypes

class GeminiHandler:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def extract_table_from_image(self, image_path, prompt):
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/png"

            contents = {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": mime_type,
                        "data": image_bytes
                    }}
                ]
            }

            response = self.client.generate_content(
                model="gemini-1.5-pro",
                contents=contents
            )

            if not response.candidates:
                logger.error("Empty response from Gemini")
                return None

            if response.candidates[0].finish_reason != "STOP":
                logger.error(f"Generation stopped early: {response.candidates[0].finish_reason}")
                return None

            return self._parse_gemini_response(response.text)
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return None

    def _parse_gemini_response(self, text):
        try:
            logger.debug(f"Raw Gemini response: {text}")
            text = text.replace("```json", "").replace("```", "")
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parsing Gemini response: {str(e)}")
            logger.debug(f"Problematic response text: {text}")
            return None
