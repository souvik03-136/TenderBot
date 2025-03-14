import warnings
import logging
import os
from dotenv import load_dotenv

warnings.filterwarnings(
    action="ignore",
    message=".*resume_download is deprecated and will be removed in version 1.0.0.*",
    category=FutureWarning
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
