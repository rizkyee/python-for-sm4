from flask import Blueprint, request, jsonify
from services import pdf_to_word

convert_bp = Blueprint('convert', __name__)

@convert_bp.route('/convert', methods=['POST'])
def convert_file():
    data = request.json

    file_id = data.get('file_id')

    try:
        result = pdf_to_word.convert(file_id)

        return jsonify({
            "success": True,
            "output_file_id": result['file_id'],
            "filename": result['filename']
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500