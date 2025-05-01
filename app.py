from flask import Flask, render_template, request, jsonify
from bot_logic import get_bot_response
from werkzeug.utils import secure_filename
import os
from flask import jsonify, request
import pandas as pd
import os
from werkzeug.utils import secure_filename

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"response": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"response": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            preview_html = df.head(5).to_html(classes="preview-table", index=False)
            return jsonify({
                "response": f"Thanks! Your dataset <strong>{filename}</strong> has been received and is ready for analysis.",
                "preview": preview_html
            })
        except Exception as e:
            return jsonify({"response": f"File saved, but failed to preview the data: {str(e)}"}), 500

    else:
        return jsonify({"response": "Please upload a CSV or Excel file (.csv, .xls, .xlsx)."}), 400

app = Flask(__name__)

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    response = get_bot_response(user_input)
    return jsonify({"response": response})

@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"response": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"response": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"response": "Thanks for uploading your data! It's saved and ready for analysis."})
    else:
        return jsonify({"response": "Please upload a CSV or Excel file (.csv, .xls, .xlsx)."}), 400

if __name__ == "__main__":
    app.run(debug=True)
