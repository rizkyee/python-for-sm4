from flask import Blueprint, send_file
import os
import config

download_bp = Blueprint('download', __name__)

@download_bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    for file in os.listdir(config.OUTPUT_FOLDER):
        if file.startswith(file_id):
            path = os.path.join(config.OUTPUT_FOLDER, file)
            return send_file(path, as_attachment=True)

    return {"error": "File not found"}, 404