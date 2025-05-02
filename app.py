from flask import Flask, render_template, request, jsonify
from bot_logic import get_bot_response
from werkzeug.utils import secure_filename
import os
import pandas as pd

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

# Define a global variable to track cleaning confirmation state
cleaning_confirmation_pending = False
uploaded_df = None  # Store the uploaded dataframe globally (or use session)

def get_response(intents_list, intents_json, user_message=None):
    global cleaning_confirmation_pending, uploaded_df

    if cleaning_confirmation_pending and uploaded_df is not None:
        message = (user_message or "").strip().lower()
        if "yes" in message:
            cleaning_confirmation_pending = False
            return "Awesome! Let's move on to the next step."
        elif "no" in message:
            cleaned_df, explanation = clean_data(uploaded_df)
            uploaded_df = cleaned_df
            cleaning_confirmation_pending = False
            return f"I’ve cleaned your data. {explanation} Let’s proceed."
        elif "not sure" in message or "unsure" in message:
            explanation = explain_data_cleanliness(uploaded_df)
            cleaned_df, cleaning_notes = clean_data(uploaded_df)
            uploaded_df = cleaned_df
            cleaning_confirmation_pending = False
            return f"No worries! {explanation} I’ve also cleaned the data for you. {cleaning_notes} Let’s continue."

    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

    return "I'm not sure how to respond to that. Can you rephrase?"


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

            # Generate HTML preview
            preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)

            # Cleanliness analysis
            total_rows = len(df)
            total_columns = len(df.columns)
            missing_values = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            total_cells = total_rows * total_columns
            missing_percent = (missing_values / total_cells) * 100 if total_cells else 0

            cleanliness_summary = (
                f"✅ <strong>{filename}</strong> has been uploaded and previewed.<br><br>"
                f"🔍 Here's a quick data quality check:<br>"
                f"- Rows: <strong>{total_rows}</strong><br>"
                f"- Columns: <strong>{total_columns}</strong><br>"
                f"- Missing values: <strong>{missing_values}</strong> ({missing_percent:.2f}%)<br>"
                f"- Duplicate rows: <strong>{duplicate_rows}</strong><br><br>"
                f"Is the data you provided clean?"
            )

            return jsonify({
                "response": cleanliness_summary,
                "preview": preview_html,
                "suggestions": [
                    "Yes, it is clean",
                    "No, I need help cleaning it",
                    "I'm not sure"
                ]
            })

        except Exception as e:
            return jsonify({"response": f"File saved, but failed to preview or analyze the data: {str(e)}"}), 500

    return jsonify({"response": "Please upload a CSV or Excel file (.csv, .xls, .xlsx)."}), 400

if __name__ == "__main__":
    app.run(debug=True)
