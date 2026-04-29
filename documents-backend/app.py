from flask import Flask
from flask_cors import CORS
import config

from api.upload import upload_bp
from api.convert import convert_bp
from api.download import download_bp

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = config.OUTPUT_FOLDER

app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(convert_bp, url_prefix='/api')
app.register_blueprint(download_bp, url_prefix='/api')

@app.route('/')
def home():
    return {"message": "API PDF to Word Running"}

if __name__ == '__main__':
    app.run(debug=True)