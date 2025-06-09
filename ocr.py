import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import logging
import os
import json
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodOCRProcessor:
    """
    OCR Processor specifically designed for food product recognition
    Optimized for detecting brands, product names, and packaging text
    """
    
    def __init__(self, languages=['en']):
        """
        Initialize OCR processor with specified languages
        Args:
            languages: List of language codes for OCR (default: ['en'])
        """
        self.easyocr_reader = easyocr.Reader(languages)
        self.food_keywords = self._load_food_keywords()
        
    def _load_food_keywords(self) -> Dict[str, List[str]]:
        """
        Load food-related keywords for better recognition
        Returns dictionary of food categories and related terms
        """
        return {
            "brands": [
                "coca-cola", "pepsi", "nestle", "unilever", "kraft", "kellogg",
                "pringles", "doritos", "lay's", "oreo", "kitkat", "snickers",
                "mars", "cadbury", "hershey", "ferrero", "nutella", "twix"
            ],
            "product_types": [
                "biscuit", "chips", "cookies", "chocolate", "candy", "snacks",
                "cereal", "crackers", "wafer", "drink", "juice", "soda",
                "milk", "yogurt", "cheese", "bread", "cake", "pastry"
            ],
            "descriptors": [
                "original", "classic", "new", "extra", "mega", "mini", "family",
                "pack", "size", "flavor", "taste", "crispy", "crunchy", "soft",
                "sweet", "salty", "spicy", "hot", "mild", "light", "diet"
            ],
            "measurements": [
                "gm", "kg", "ml", "ltr", "oz", "lb", "pack", "pieces", "count"
            ]
        }
    
    def preprocess_image_for_food(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image specifically for food packaging OCR
        Optimized for colorful packaging and various lighting conditions
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert to different color spaces for better text detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Multiple thresholding techniques for different text types
            # Adaptive threshold for varied lighting
            adaptive_thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Otsu's threshold for clear text
            _, otsu_thresh = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Combine thresholds
            combined_thresh = cv2.bitwise_or(adaptive_thresh, otsu_thresh)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return None
    
    def extract_text_with_tesseract(self, image_path: str) -> Dict[str, any]:
        """
        Extract text using Tesseract OCR with food-specific configuration
        """
        try:
            processed_img = self.preprocess_image_for_food(image_path)
            if processed_img is None:
                processed_img = cv2.imread(image_path)
            
            # Custom config for food packaging text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?@#$%^&*()_+-=[]{}|";:<>?/~` '
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter and organize results
            extracted_text = []
            text_blocks = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        extracted_text.append(text)
                        text_blocks.append({
                            'text': text,
                            'confidence': data['conf'][i],
                            'bbox': [data['left'][i], data['top'][i], data['width'][i], data['height'][i]]
                        })
            
            return {
                'text': ' '.join(extracted_text),
                'blocks': text_blocks,
                'method': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Error in Tesseract OCR: {e}")
            return {'text': '', 'blocks': [], 'method': 'tesseract'}
    
    def extract_text_with_easyocr(self, image_path: str) -> Dict[str, any]:
        """
        Extract text using EasyOCR optimized for food packaging
        """
        try:
            # EasyOCR works better with original colored images for food packaging
            results = self.easyocr_reader.readtext(image_path, detail=1)
            
            extracted_text = []
            text_blocks = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Lower threshold for food packaging
                    extracted_text.append(text)
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            return {
                'text': ' '.join(extracted_text),
                'blocks': text_blocks,
                'method': 'easyocr'
            }
            
        except Exception as e:
            logger.error(f"Error in EasyOCR: {e}")
            return {'text': '', 'blocks': [], 'method': 'easyocr'}
    
    def detect_food_objects(self, image_path: str) -> List[Dict]:
        """
        Detect food-related objects and packaging elements
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangular packages (common in food packaging)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Filter small contours
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Classify based on shape and size
                    if len(approx) == 4 and 0.3 < aspect_ratio < 3:
                        # Rectangular package
                        detected_objects.append({
                            "type": "package",
                            "coordinates": [x, y, w, h],
                            "confidence": 0.8,
                            "aspect_ratio": aspect_ratio
                        })
                    elif len(approx) > 6:
                        # Circular/oval (bottles, cans)
                        detected_objects.append({
                            "type": "container",
                            "coordinates": [x, y, w, h],
                            "confidence": 0.7,
                            "aspect_ratio": aspect_ratio
                        })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Error in food object detection: {e}")
            return []
    
    def identify_food_brands_and_products(self, text_data: Dict) -> Dict[str, any]:
        """
        Identify food brands and products from extracted text
        """
        try:
            all_text = text_data.get('text', '').lower()
            text_blocks = text_data.get('blocks', [])
            
            identified_brands = []
            identified_products = []
            identified_descriptors = []
            
            # Check for brands
            for brand in self.food_keywords['brands']:
                if brand.lower() in all_text:
                    identified_brands.append(brand)
            
            # Check for product types
            for product in self.food_keywords['product_types']:
                if product.lower() in all_text:
                    identified_products.append(product)
            
            # Check for descriptors
            for descriptor in self.food_keywords['descriptors']:
                if descriptor.lower() in all_text:
                    identified_descriptors.append(descriptor)
            
            # Extract potential brand/product names from high-confidence blocks
            potential_names = []
            for block in text_blocks:
                if block['confidence'] > 0.7:  # High confidence text
                    text = block['text'].strip()
                    if len(text) > 2 and text.isalpha():  # Likely a brand/product name
                        potential_names.append(text)
            
            return {
                'identified_brands': list(set(identified_brands)),
                'identified_products': list(set(identified_products)),
                'identified_descriptors': list(set(identified_descriptors)),
                'potential_names': potential_names[:5],  # Top 5 candidates
                'full_text': all_text
            }
            
        except Exception as e:
            logger.error(f"Error in brand/product identification: {e}")
            return {
                'identified_brands': [],
                'identified_products': [],
                'identified_descriptors': [],
                'potential_names': [],
                'full_text': ''
            }
    
    def process_food_image(self, image_path: str) -> Dict[str, any]:
        """
        Complete food image processing pipeline
        Returns comprehensive analysis of food packaging
        """
        try:
            logger.info(f"Processing food image: {image_path}")
            
            # Extract text using both OCR methods
            tesseract_result = self.extract_text_with_tesseract(image_path)
            easyocr_result = self.extract_text_with_easyocr(image_path)
            
            # Detect objects
            detected_objects = self.detect_food_objects(image_path)
            
            # Choose better OCR result based on confidence and text length
            if len(easyocr_result['text']) > len(tesseract_result['text']):
                primary_ocr = easyocr_result
                secondary_ocr = tesseract_result
            else:
                primary_ocr = tesseract_result
                secondary_ocr = easyocr_result
            
            # Identify brands and products
            brand_analysis = self.identify_food_brands_and_products(primary_ocr)
            
            # Combine results
            result = {
                'primary_ocr': primary_ocr,
                'secondary_ocr': secondary_ocr,
                'detected_objects': detected_objects,
                'brand_analysis': brand_analysis,
                'processing_summary': {
                    'total_text_length': len(primary_ocr['text']),
                    'objects_detected': len(detected_objects),
                    'brands_found': len(brand_analysis['identified_brands']),
                    'products_found': len(brand_analysis['identified_products']),
                    'confidence_score': self._calculate_confidence_score(primary_ocr, brand_analysis)
                }
            }
            
            logger.info(f"Food processing completed: {result['processing_summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in food image processing: {e}")
            return {
                'primary_ocr': {'text': '', 'blocks': [], 'method': 'error'},
                'secondary_ocr': {'text': '', 'blocks': [], 'method': 'error'},
                'detected_objects': [],
                'brand_analysis': {
                    'identified_brands': [],
                    'identified_products': [],
                    'identified_descriptors': [],
                    'potential_names': [],
                    'full_text': ''
                },
                'processing_summary': {
                    'total_text_length': 0,
                    'objects_detected': 0,
                    'brands_found': 0,
                    'products_found': 0,
                    'confidence_score': 0.0
                }
            }
    
    def _calculate_confidence_score(self, ocr_result: Dict, brand_analysis: Dict) -> float:
        """
        Calculate overall confidence score for food recognition
        """
        try:
            # Base score from OCR confidence
            ocr_score = 0.0
            if ocr_result['blocks']:
                confidences = [block['confidence'] for block in ocr_result['blocks']]
                ocr_score = sum(confidences) / len(confidences) / 100  # Normalize to 0-1
            
            # Bonus for identified brands/products
            brand_bonus = len(brand_analysis['identified_brands']) * 0.2
            product_bonus = len(brand_analysis['identified_products']) * 0.1
            
            # Text length factor
            text_factor = min(len(brand_analysis['full_text']) / 100, 1.0)
            
            final_score = min((ocr_score + brand_bonus + product_bonus) * text_factor, 1.0)
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

# Utility functions for external use
def create_food_ocr_processor(languages=['en']) -> FoodOCRProcessor:
    """
    Factory function to create FoodOCRProcessor instance
    """
    return FoodOCRProcessor(languages)

def process_single_food_image(image_path: str, languages=['en']) -> Dict[str, any]:
    """
    Convenience function to process a single food image
    """
    processor = FoodOCRProcessor(languages)
    return processor.process_food_image(image_path)
