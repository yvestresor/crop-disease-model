from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Added secret key for flash messages

# =============================================================================
# PARAMETERS
# =============================================================================
IMG_HEIGHT, IMG_WIDTH = 224, 224

# =============================================================================
# LOAD THE TRAINED MODEL
# =============================================================================
# Wrap model loading in a try-except to handle errors gracefully
try:
    model = tf.keras.models.load_model("final_crop_disease_model.keras")
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# =============================================================================
# LABEL MAPPING
# =============================================================================
int_to_label = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    5: "Corn_(maize)___Common_rust_",
    6: "Corn_(maize)___Northern_Leaf_Blight",
    7: "Corn_(maize)___healthy",
    8: "Grape___Black_rot",
    9: "Grape___Esca_(Black_Measles)",
    10: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    11: "Grape___healthy",
    12: "Potato___Early_blight",
    13: "Potato___Late_blight",
    14: "Potato___healthy", 
    15: "Tomato___Bacterial_spot",
    16: "Tomato___Early_blight",
    17: "Tomato___Late_blight",
    18: "Tomato___Leaf_Mold",
    19: "Tomato___Septoria_leaf_spot",
    20: "Tomato___Spider_mites Two-spotted_spider_mite",
    21: "Tomato___Target_Spot",
    22: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    23: "Tomato___Tomato_mosaic_virus",
    24: "Tomato___healthy"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def load_and_preprocess_image(image_bytes):
    """
    Load an image from raw bytes, decode it, resize, and normalize.
    """
    try:
        image = tf.image.decode_image(image_bytes, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = image / 255.0  # Normalize pixel values to [0, 1]
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def parse_prediction(label_str):
    """
    Given a predicted label string in the format "Plant___Disease" or "Plant___healthy",
    parse and return the plant type, a boolean indicating infection, and the disease.
    """
    parts = label_str.split("___")
    plant = parts[0].replace("_", " ")
    
    if len(parts) > 1:
        disease = parts[1].replace("_", " ")
    else:
        disease = "Unknown"
        
    infected = False if disease.lower() == "healthy" else True
    return plant, infected, disease

def get_disease_info(disease):
    """
    Return information about the disease for display purposes.
    """
    disease_info = {
        "Apple scab": {
            "description": "A fungal disease that causes dark, scabby lesions on leaves and fruit.",
            "treatment": "Apply fungicides during the growing season, practice good sanitation by removing fallen leaves."
        },
        "Black rot": {
            "description": "A fungal disease causing circular lesions on leaves and fruit rot.",
            "treatment": "Prune infected branches, apply fungicides, and maintain good air circulation."
        },
        "Cedar apple rust": {
            "description": "A fungal disease causing bright orange spots on leaves and fruit.",
            "treatment": "Remove nearby cedar trees if possible, apply protective fungicides."
        },
        "Cercospora leaf spot Gray leaf spot": {
            "description": "A fungal disease causing gray to tan spots with dark borders on corn leaves.",
            "treatment": "Rotate crops, use resistant varieties, and apply fungicides when necessary."
        },
        "Common rust": {
            "description": "Fungal disease appearing as small, reddish-brown pustules on corn leaves.",
            "treatment": "Plant resistant varieties, apply fungicides early in the growing season."
        },
        "Northern Leaf Blight": {
            "description": "A fungal disease causing large, cigar-shaped lesions on corn leaves.",
            "treatment": "Use crop rotation, plant resistant varieties, and apply fungicides."
        },
        "Esca (Black Measles)": {
            "description": "A complex fungal disease causing leaf discoloration and wood decay in grapevines.",
            "treatment": "Remove infected vines, avoid stress on plants, no effective chemical controls."
        },
        "Leaf blight (Isariopsis Leaf Spot)": {
            "description": "Fungal disease causing brown spots with yellowish margins on grape leaves.",
            "treatment": "Apply fungicides, ensure good air circulation, manage canopy."
        },
        "Early blight": {
            "description": "Fungal disease causing dark, concentric rings on lower leaves of potato/tomato plants.",
            "treatment": "Rotate crops, water at the base, apply fungicides preventatively."
        },
        "Late blight": {
            "description": "A destructive disease causing water-soaked spots and white fungal growth.",
            "treatment": "Use resistant varieties, apply fungicides before symptoms appear, destroy infected plants."
        },
        "Bacterial spot": {
            "description": "Bacterial disease causing small, dark spots on leaves, stems, and fruit.",
            "treatment": "Rotate crops, use copper-based sprays, avoid overhead irrigation."
        },
        "Leaf Mold": {
            "description": "Fungal disease causing yellow spots on leaf surfaces and olive-green mold underneath.",
            "treatment": "Improve air circulation, reduce humidity, apply fungicides."
        },
        "Septoria leaf spot": {
            "description": "Fungal disease causing small, circular spots with dark borders on tomato leaves.",
            "treatment": "Rotate crops, remove infected leaves, apply fungicides."
        },
        "Spider mites Two-spotted spider mite": {
            "description": "Tiny pests causing stippling on leaves and fine webbing.",
            "treatment": "Introduce predatory mites, spray with water, apply miticides if necessary."
        },
        "Target Spot": {
            "description": "Fungal disease causing concentric rings of brown spots on tomato leaves.",
            "treatment": "Improve air circulation, apply fungicides, rotate crops."
        },
        "Tomato Yellow Leaf Curl Virus": {
            "description": "Viral disease causing yellowing and curling of leaves, stunted growth.",
            "treatment": "Control whitefly vectors, use resistant varieties, remove infected plants."
        },
        "Tomato mosaic virus": {
            "description": "Viral disease causing mottled and distorted leaves, stunted growth.",
            "treatment": "No cure available, remove and destroy infected plants, control aphids."
        }
    }
    
    # Clean up disease name for lookup
    clean_disease = disease.replace("_", " ")
    for key in disease_info:
        if key.lower() in clean_disease.lower():
            return disease_info[key]
    
    return {
        "description": "Information not available for this condition.",
        "treatment": "Consult with a local agricultural extension for specific advice."
    }

# =============================================================================
# ROUTES
# =============================================================================
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if not model_loaded:
        flash("Model could not be loaded. Please check the server logs.", "error")
        return render_template("upload.html", model_error=True)
        
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)
            
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)
            
        if file:
            try:
                # Read file bytes and convert to base64 for display
                image_bytes = file.read()
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                session['image'] = encoded_image
                
                # Process image for prediction
                image = load_and_preprocess_image(image_bytes)
                if image is None:
                    flash("Could not process the image. Please try another one.", "error")
                    return redirect(request.url)
                    
                # Expand dims to create batch of 1
                image_expanded = tf.expand_dims(image, axis=0)
                
                # Predict using the model
                predictions = model.predict(image_expanded)
                pred_index = np.argmax(predictions, axis=1)[0]
                confidence = float(predictions[0][pred_index]) * 100  # Convert to percentage
                
                predicted_label = int_to_label.get(pred_index, "Unknown")
                plant, infected, disease = parse_prediction(predicted_label)
                
                # Get additional information about the disease
                disease_info = get_disease_info(disease) if infected else {
                    "description": "The plant appears to be healthy.",
                    "treatment": "Continue with regular care and monitoring."
                }
                
                return render_template(
                    "result.html", 
                    plant=plant,
                    infected=infected,
                    disease=disease,
                    confidence=confidence,
                    image=encoded_image,
                    description=disease_info["description"],
                    treatment=disease_info["treatment"]
                )
                
            except Exception as e:
                flash(f"Error processing request: {str(e)}", "error")
                return redirect(request.url)
                
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)