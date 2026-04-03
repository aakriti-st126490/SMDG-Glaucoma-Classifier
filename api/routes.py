from flask import Blueprint, request, jsonify
from flask import render_template
from utils.preprocess import preprocess_image
from utils.gradcam import generate_gradcam

def create_routes(models):
    bp = Blueprint("routes", __name__)

    @bp.route("/", methods=["GET"])
    def home():
        return "Glaucoma API is running!"

    @bp.route("/", methods=["GET"])
    def ui():
        return render_template("index.html")
        
    @bp.route("/predict", methods=["POST"])
    def predict():
        file = request.files["image"]
        img = preprocess_image(file)

        results = {}

        for name, model in models.items():
            output = model.predict(img)
            print("output",output)
            # Handle dict output (TFSMLayer)
            if isinstance(output, dict):
                output = list(output.values())[0]
            
            # Convert tensor → numpy
            if hasattr(output, "numpy"):
                output = output.numpy()
            
            # Now safely extract value
            pred = float(output.squeeze())
    
            label = "Glaucoma" if pred > 0.5 else "Normal"
    
            heatmap = generate_gradcam(model, img, name)
    
            results[name] = {
                "prediction": label,
                "confidence": float(pred),
                "heatmap": heatmap.tolist()
            }
    
        return jsonify(results)

    @bp.route("/predict_ui", methods=["POST"])
    def predict_ui():
        file = request.files["image"]
        img = preprocess_image(file)
    
        results = {}
    
        for name, model in models.items():
            output = model.predict(img)
    
            if isinstance(output, dict):
                output = list(output.values())[0]
    
            if hasattr(output, "numpy"):
                output = output.numpy()
    
            pred = float(output.squeeze())
    
            label = "Glaucoma" if pred > 0.5 else "Normal"
    
            results[name] = {
                "prediction": label,
                "confidence": round(pred, 3)
            }
    
        return render_template("index.html", results=results)
    
    return bp