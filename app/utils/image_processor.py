from PIL import Image
import io
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Utility class for image processing operations
    """
    
    @staticmethod
    def validate_image_format(image_data: bytes) -> bool:
        """
        Validate image format and integrity
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            return True
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return False
    
    @staticmethod
    def load_image(image_data: bytes) -> Optional[Image.Image]:
        """
        Load image from bytes and convert to RGB
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def get_image_info(image: Image.Image) -> Dict:
        """
        Get basic image information
        """
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format
        }
    
    @staticmethod
    def validate_image_size(image: Image.Image, min_size: tuple = (100, 100)) -> bool:
        """
        Validate image meets minimum size requirements
        """
        return image.size[0] >= min_size[0] and image.size[1] >= min_size[1]