#!/usr/bin/env python3.7
"""
Frontend to download models
How to download:
1. Open http://localhost:5000/files to get list of files
2. To download, open http://localhost:5000/files/put_filename_here

Usage:
    python3.7 app.py
"""

from flask import Flask, jsonify, send_from_directory
import os

UPLOAD_DIRECTORY = "/apps/models"
api = Flask(__name__)


@api.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in sorted(os.listdir(UPLOAD_DIRECTORY)):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)


@api.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)


if __name__ == "__main__":
    api.run(host='0.0.0.0', port=5000)
