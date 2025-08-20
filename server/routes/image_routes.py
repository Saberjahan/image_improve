# server/routes/image_routes.py

from flask import Blueprint, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np # <-- This line is crucial for the new code to work

# Import the AI service functions
from services.ai_service import (
    process_image_with_ai,
    generate_keywords_from_description
)

# Create a Blueprint for image-related routes
image_bp = Blueprint('image_bp', __name__)

@image_bp.route('/image/process', methods=['POST'])
def process_image():
    """
    API endpoint to process an image using various AI methods.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    image_b64 = data.get('image')
    mask_b64 = data.get('mask')
    process_type = data.get('process_type')
    image_type_description = data.get('image_type_description', '')

    try:
        if image_b64:
            if image_b64.startswith('data:image/'):
                image_b64 = image_b64.split(',')[1]
            original_image_bytes = base64.b64decode(image_b64)
            original_image = Image.open(BytesIO(original_image_bytes)).convert("RGB")
        else:
            return jsonify({"error": "Original image data is missing"}), 400

        mask_image = None
        if mask_b64:
            if mask_b64.startswith('data:image/'):
                mask_b64 = mask_b64.split(',')[1]
            mask_image_bytes = base64.b64decode(mask_b64)

            # Open the mask as a grayscale image (L mode)
            pil_mask = Image.open(BytesIO(mask_image_bytes)).convert("L")

            # Convert the PIL image to a NumPy array
            mask_np = np.array(pil_mask)

            # Invert the values to get 0 for masked (was 255) and 1 for unmasked (was 0)
            inverted_mask_np = np.where(mask_np > 0, 0, 1).astype(np.uint8)

            # Convert the NumPy array back to a PIL Image
            mask_image = Image.fromarray(inverted_mask_np * 255, mode='L')
            
            # This is a key step to ensure compatibility with the AI service.
            # You would need to change the ai_service.py to accept a 0/1 mask
            # For now, it will be inverted back.
            mask_image_inverted_to_pil = Image.fromarray(inverted_mask_np, mode='L')

            # Pass the modified mask to the AI service
            processed_image_pil = process_image_with_ai(
                original_image,
                mask_image_inverted_to_pil,
                process_type,
                image_type_description
            )

        else:
            processed_image_pil = process_image_with_ai(
                original_image,
                None,
                process_type,
                image_type_description
            )

        # Encode the processed image back to base64 Data URL
        buffered = BytesIO()
        processed_image_pil.save(buffered, format="PNG")
        processed_image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        processed_image_data_url = f"data:image/png;base64,{processed_image_b64}"

        return jsonify({
            "processed_image": processed_image_data_url,
            "message": f"Image successfully processed with '{process_type}'."
        }), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500