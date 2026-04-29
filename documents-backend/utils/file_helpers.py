import uuid
import os

def generate_file_id():
    return str(uuid.uuid4())

def get_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

def build_filepath(folder, file_id, ext):
    return os.path.join(folder, f"{file_id}.{ext}")