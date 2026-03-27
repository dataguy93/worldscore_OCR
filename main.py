import base64
import json
import os
import tempfile

from flask import Flask, jsonify, request

from ocr_engine import ocr_scorecard

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/ocr", methods=["OPTIONS"])
def ocr_options():
    return ("", 204)


def _decode_base64_image(image_base64):
    if not image_base64:
        return None
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    return base64.b64decode(image_base64)


def _create_temp_image_file(image_file=None, image_base64=None):
    if image_file is None and not image_base64:
        raise ValueError("No image content was provided")

    suffix = ".jpg"
    if image_file is not None:
        _, ext = os.path.splitext(image_file.filename or "")
        if ext:
            suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_file.save(tmp)
            return tmp.name

    image_bytes = _decode_base64_image(image_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        return tmp.name


@app.get("/")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/ready")
def ready_check():
    return jsonify({"ready": True, "service": "worldscore-ocr"})


@app.post("/ocr")
def ocr_endpoint():
    image_file = request.files.get("image")
    body = request.get_json(silent=True) or {}
    image_base64 = body.get("image_base64")

    if image_file is None and not image_base64:
        return jsonify({"error": "Provide an image file via multipart/form-data field 'image' or JSON field 'image_base64'."}), 400

    tmp_path = None

    try:
        tmp_path = _create_temp_image_file(image_file=image_file, image_base64=image_base64)

        result = ocr_scorecard(tmp_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"OCR request failed: {exc}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
