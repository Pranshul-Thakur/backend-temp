import os
import logging
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.generativeai import configure, GenerativeModel
from werkzeug.utils import secure_filename
from flask_caching import Cache
from PIL import Image
import io

configure(api_key="")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated allowed extensions for images
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process image and prepare it for Gemini Vision API"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if image is too large (optional - Gemini can handle large images)
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return img_byte_arr
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def get_food_analysis(image_data, analysis_type):
    """Use Gemini Vision to analyze food in the image"""
    model = GenerativeModel("gemini-1.5-flash")
    
    if analysis_type == "Food Detection":
        prompt = """
        Analyze this image and provide the following information about the food items:
        
        1. Identify all food items visible in the image
        2. Provide the name of the main dish/food item
        3. List the ingredients you can identify
        4. Estimate the portion size (small, medium, large)
        5. Provide nutritional information if possible (calories, main nutrients)
        6. Rate the healthiness of the food (1-10 scale)
        
        If no food is detected, please mention that clearly.
        """
    elif analysis_type == "Nutritional Analysis":
        prompt = """
        Analyze this food image and provide detailed nutritional information:
        
        1. Identify the food item(s)
        2. Estimate calories per serving
        3. Break down macronutrients (carbs, protein, fat)
        4. List key vitamins and minerals present
        5. Assess the nutritional value (high, medium, low)
        6. Provide health benefits or concerns
        7. Suggest healthier alternatives if applicable
        
        If no food is detected, please mention that clearly.
        """
    else:  # Recipe Suggestion
        prompt = """
        Analyze this food image and provide recipe-related information:
        
        1. Identify the dish/food item
        2. List the main ingredients you can see
        3. Suggest the cooking method used
        4. Provide a brief recipe or cooking instructions
        5. Suggest variations or improvements
        6. Recommend similar dishes
        7. Estimate preparation and cooking time
        
        If no food is detected or if it's a packaged/processed food, mention that clearly.
        """
    
    try:
        # Create the image part for Gemini
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_data
        }
        
        response = model.generate_content([prompt, image_part])
        return response.text if hasattr(response, 'text') else "No response received."
    except Exception as e:
        logger.error(f"Error in AI processing: {e}")
        return f"Error in AI processing: {str(e)}"

# Cache for processed images
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

@cache.memoize(timeout=300)  # Cache for 5 minutes
def cached_image_process(file_path):
    return process_image(file_path)

@app.route("/analyze", methods=["POST"])
def analyze_food():
    if "image" not in request.files:
        return jsonify({"error": "No image file part in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Upload an image (PNG, JPG, JPEG, GIF, BMP, WEBP)."}), 400

    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)
    
    logger.info(f"Image {filename} uploaded successfully.")
    
    # Process the image
    image_data = process_image(file_path)
    
    if not image_data:
        return jsonify({"error": "Failed to process the image."}), 500

    # Get analysis type from request
    analysis_type = request.form.get("analysis_option", "Food Detection")
    
    # Analyze the food image
    analysis_result = get_food_analysis(image_data, analysis_type)
    
    # Clean up temporary file
    try:
        os.remove(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary files: {e}")
    
    return jsonify({
        "analysis": analysis_result,
        "filename": filename,
        "analysis_type": analysis_type
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "Food Detection API"})

@app.route("/")
def home():
    return """
    <h1>Food Detection API</h1>
    <p>Upload an image to detect and analyze food items.</p>
    <p>Available analysis types:</p>
    <ul>
        <li>Food Detection - Basic food identification</li>
        <li>Nutritional Analysis - Detailed nutrition information</li>
        <li>Recipe Suggestion - Recipe and cooking information</li>
    </ul>
    """

if __name__ == "__main__":
    app.run(debug=True)
