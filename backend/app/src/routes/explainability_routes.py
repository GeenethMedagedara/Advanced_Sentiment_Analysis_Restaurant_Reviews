from flask import Blueprint, request, jsonify
from src.models.model_loader import explainer  # Import the ABSAExplainability instance

explain_bp = Blueprint("explain", __name__)

@explain_bp.route("/explain_shap", methods=["POST"])
def explain_shap():
    """Generate SHAP explanation and return as base64."""
    try:
        data = request.get_json()
        base64_image = explainer.explain_spacy_similarity(data["aspect"], data["review"])
        return jsonify({"explanation": base64_image})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@explain_bp.route("/explain_lime", methods=["POST"])
def explain_lime():
    """Generate LIME explanation and return as base64."""
    try:
        data = request.get_json()
        base64_image = explainer.explain_lime(data["aspect"], data["review"])
        return jsonify({"explanation": base64_image})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
