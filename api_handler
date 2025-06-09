import logging
from typing import Dict, Any, Optional
from google.generativeai import configure, GenerativeModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIHandler:
    """
    Handles API communications with Google Gemini AI
    Supports custom prompts for different use cases
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize API Handler
        Args:
            api_key: Google Gemini API key
            model_name: Model name to use (default: gemini-1.5-flash)
        """
        try:
            configure(api_key=api_key)
            self.model = GenerativeModel(model_name)
            self.model_name = model_name
            logger.info(f"API Handler initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing API Handler: {e}")
            raise
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response from AI model using custom prompt
        Args:
            prompt: The complete prompt to send to the model
            max_retries: Number of retry attempts on failure
        Returns:
            Generated response text
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response (attempt {attempt + 1}/{max_retries})")
                response = self.model.generate_content(prompt)
                
                if hasattr(response, 'text') and response.text:
                    logger.info("Response generated successfully")
                    return response.text
                else:
                    logger.warning("Empty response received from API")
                    return "No response received from the AI model."
                    
            except Exception as e:
                logger.error(f"Error in API call (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return f"Error in AI processing after {max_retries} attempts: {str(e)}"
                
        return "Failed to generate response after multiple attempts."

class PromptManager:
    """
    Manages different types of prompts for various use cases
    Allows custom prompt injection and template management
    """
    
    def __init__(self):
        self.prompt_templates = {}
        self.custom_prompts = {}
    
    def add_prompt_template(self, template_name: str, template: str):
        """
        Add a new prompt template
        Args:
            template_name: Name of the template
            template: Template string with placeholders
        """
        self.prompt_templates[template_name] = template
        logger.info(f"Added prompt template: {template_name}")
    
    def add_custom_prompt(self, prompt_name: str, prompt: str):
        """
        Add a custom prompt
        Args:
            prompt_name: Name of the custom prompt
            prompt: Complete prompt string
        """
        self.custom_prompts[prompt_name] = prompt
        logger.info(f"Added custom prompt: {prompt_name}")
    
    def get_prompt_template(self, template_name: str) -> Optional[str]:
        """
        Get a prompt template by name
        """
        return self.prompt_templates.get(template_name)
    
    def get_custom_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Get a custom prompt by name
        """
        return self.custom_prompts.get(prompt_name)
    
    def build_prompt_from_template(self, template_name: str, **kwargs) -> Optional[str]:
        """
        Build a prompt from template with provided variables
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in template
        Returns:
            Built prompt string or None if template not found
        """
        template = self.get_prompt_template(template_name)
        if template:
            try:
                return template.format(**kwargs)
            except KeyError as e:
                logger.error(f"Missing template variable: {e}")
                return None
        return None
    
    def list_available_prompts(self) -> Dict[str, list]:
        """
        List all available prompts and templates
        """
        return {
            'templates': list(self.prompt_templates.keys()),
            'custom_prompts': list(self.custom_prompts.keys())
        }

class FoodAnalysisProcessor:
    """
    Processes food recognition data and generates AI analysis
    Integrates OCR results with custom AI prompts
    """
    
    def __init__(self, api_handler: APIHandler, prompt_manager: PromptManager):
        """
        Initialize processor with API handler and prompt manager
        """
        self.api_handler = api_handler
        self.prompt_manager = prompt_manager
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """
        Setup default prompt templates for food analysis
        """
        # Default food analysis template
        food_analysis_template = """
You are a Food Recognition Expert. Analyze the following food product information extracted from an image:

Extracted Text: {extracted_text}
Identified Brands: {identified_brands}
Identified Products: {identified_products}
Detected Objects: {detected_objects}
Processing Confidence: {confidence_score}

{custom_instructions}

Please provide your analysis based on the above information.
"""
        
        # Default brand identification template  
        brand_identification_template = """
You are a Brand Recognition Specialist. Based on the following OCR data from food packaging:

Text Data: {extracted_text}
Potential Brand Names: {potential_names}
Product Context: {product_context}

{custom_instructions}

Identify the brand and product with confidence levels.
"""
        
        self.prompt_manager.add_prompt_template("food_analysis", food_analysis_template)
        self.prompt_manager.add_prompt_template("brand_identification", brand_identification_template)
    
    def process_food_analysis(self, 
                            ocr_data: Dict[str, Any], 
                            custom_prompt: Optional[str] = None,
                            template_name: Optional[str] = None,
                            additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process food analysis using OCR data and custom prompts
        Args:
            ocr_data: Data from OCR processing
            custom_prompt: Custom prompt to use (overrides template)
            template_name: Name of template to use
            additional_context: Additional context for the analysis
        Returns:
            Analysis results
        """
        try:
            # Extract relevant data from OCR results
            brand_analysis = ocr_data.get('brand_analysis', {})
            processing_summary = ocr_data.get('processing_summary', {})
            
            extracted_text = brand_analysis.get('full_text', '')
            identified_brands = ', '.join(brand_analysis.get('identified_brands', []))
            identified_products = ', '.join(brand_analysis.get('identified_products', []))
            detected_objects = str(len(ocr_data.get('detected_objects', [])))
            confidence_score = processing_summary.get('confidence_score', 0.0)
            
            # Build prompt
            final_prompt = ""
            
            if custom_prompt:
                # Use custom prompt directly
                final_prompt = custom_prompt
                # Replace placeholders if they exist
                final_prompt = final_prompt.replace("{extracted_text}", extracted_text)
                final_prompt = final_prompt.replace("{identified_brands}", identified_brands)
                final_prompt = final_prompt.replace("{identified_products}", identified_products)
                final_prompt = final_prompt.replace("{detected_objects}", detected_objects)
                final_prompt = final_prompt.replace("{confidence_score}", str(confidence_score))
                
            elif template_name:
                # Use template
                template_vars = {
                    'extracted_text': extracted_text,
                    'identified_brands': identified_brands,
                    'identified_products': identified_products,
                    'detected_objects': detected_objects,
                    'confidence_score': confidence_score,
                    'custom_instructions': additional_context.get('instructions', '') if additional_context else ''
                }
                final_prompt = self.prompt_manager.build_prompt_from_template(template_name, **template_vars)
                
            else:
                # Use default food analysis template
                template_vars = {
                    'extracted_text': extracted_text,
                    'identified_brands': identified_brands,
                    'identified_products': identified_products,
                    'detected_objects': detected_objects,
                    'confidence_score': confidence_score,
                    'custom_instructions': additional_context.get('instructions', '') if additional_context else 'Provide a comprehensive analysis of this food product.'
                }
                final_prompt = self.prompt_manager.build_prompt_from_template("food_analysis", **template_vars)
            
            if not final_prompt:
                return {
                    'success': False,
                    'error': 'Failed to build prompt',
                    'analysis': 'Could not generate analysis due to prompt building error.'
                }
            
            # Generate AI response
            logger.info("Generating AI analysis for food product")
            analysis = self.api_handler.generate_response(final_prompt)
            
            return {
                'success': True,
                'analysis': analysis,
                'prompt_used': 'custom' if custom_prompt else (template_name or 'food_analysis'),
                'ocr_summary': processing_summary,
                'identified_items': {
                    'brands': brand_analysis.get('identified_brands', []),
                    'products': brand_analysis.get('identified_products', []),
                    'descriptors': brand_analysis.get('identified_descriptors', [])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in food analysis processing: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': 'Error occurred during analysis processing.'
            }
    
    def process_brand_identification(self, 
                                   ocr_data: Dict[str, Any],
                                   custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Specialized processing for brand identification
        """
        try:
            brand_analysis = ocr_data.get('brand_analysis', {})
            
            if custom_prompt:
                final_prompt = custom_prompt
            else:
                template_vars = {
                    'extracted_text': brand_analysis.get('full_text', ''),
                    'potential_names': ', '.join(brand_analysis.get('potential_names', [])),
                    'product_context': ', '.join(brand_analysis.get('identified_products', [])),
                    'custom_instructions': 'Focus on identifying the main brand and product name with confidence levels.'
                }
                final_prompt = self.prompt_manager.build_prompt_from_template("brand_identification", **template_vars)
            
            if not final_prompt:
                return {'success': False, 'error': 'Failed to build brand identification prompt'}
            
            analysis = self.api_handler.generate_response(final_prompt)
            
            return {
                'success': True,
                'brand_analysis': analysis,
                'identified_brands': brand_analysis.get('identified_brands', []),
                'potential_names': brand_analysis.get('potential_names', [])
            }
            
        except Exception as e:
            logger.error(f"Error in brand identification: {e}")
            return {'success': False, 'error': str(e)}

# Factory functions for easy initialization
def create_api_handler(api_key: str, model_name: str = "gemini-1.5-flash") -> APIHandler:
    """
    Create API Handler instance
    """
    return APIHandler(api_key, model_name)

def create_prompt_manager() -> PromptManager:
    """
    Create Prompt Manager instance
    """
    return PromptManager()

def create_food_processor(api_key: str, model_name: str = "gemini-1.5-flash") -> FoodAnalysisProcessor:
    """
    Create complete Food Analysis Processor with API handler and prompt manager
    """
    api_handler = create_api_handler(api_key, model_name)
    prompt_manager = create_prompt_manager()
    return FoodAnalysisProcessor(api_handler, prompt_manager)
