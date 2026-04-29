from flask import Blueprint, request, jsonify
import config
from utils.file_helpers import generate_file_id, get_extension, build_filepath

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"}), 400

    ext = get_extension(file.filename)

    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": "Only PDF allowed"}), 400

    file_id = generate_file_id()
    filepath = build_filepath(config.UPLOAD_FOLDER, file_id, ext)

    file.save(filepath)

    return jsonify({
        "success": True,
        "file_id": file_id,
        "filename": file.filename
    })