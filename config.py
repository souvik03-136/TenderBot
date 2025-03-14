import warnings
import logging

# Suppress the specific FutureWarning from huggingface_hub regarding resume_download
warnings.filterwarnings(
    action="ignore",
    message=".*resume_download is deprecated and will be removed in version 1.0.0.*",
    category=FutureWarning
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
