import base64
import os
import tempfile

from flask import Flask, jsonify, request

from ocr_engine import ocr_scorecard, process_ocr_for_round

app = Flask(__name__)


@app.get("/")
def health_check():
    return jsonify({"status": "ok"})


@app.post("/ocr")
def ocr_endpoint():
    image_file = request.files.get("image")
    image_base64 = request.json.get("image_base64") if request.is_json else None

    if image_file is None and not image_base64:
        return jsonify({"error": "Provide an image file via multipart/form-data field 'image' or JSON field 'image_base64'."}), 400

    suffix = ".jpg"
    tmp_path = None

    try:
        if image_file is not None:
            _, ext = os.path.splitext(image_file.filename or "")
            if ext:
                suffix = ext
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                image_file.save(tmp)
                tmp_path = tmp.name
        else:
            image_bytes = base64.b64decode(image_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name

        result = ocr_scorecard(tmp_path)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"OCR request failed: {exc}"}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/ocr/round")
def ocr_round_endpoint():
    image_file = request.files.get("image")
    round_data = request.form.get("round_data")

    if image_file is None:
        return jsonify({"error": "Provide an image file via multipart/form-data field 'image'."}), 400

    try:
        parsed_round_data = {}
        if round_data:
            import json
            parsed_round_data = json.loads(round_data)

        _, ext = os.path.splitext(image_file.filename or "")
        suffix = ext or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_file.save(tmp)
            tmp_path = tmp.name

        result = process_ocr_for_round(tmp_path, parsed_round_data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Round OCR request failed: {exc}"}), 500
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
