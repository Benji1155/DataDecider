from flask import Flask, render_template, request, jsonify, session # Added session
from bot_logic import get_bot_response as nlu_get_bot_response # aliased original bot_logic function
from werkzeug.utils import secure_filename
import os
import pandas as pd

app = Flask(__name__)
# IMPORTANT: Set a strong, random secret key in a real application, ideally from an environment variable or config file.
# For development, os.urandom(24) is fine. For production, use a fixed key.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    # Clear any previous visualization state when user goes to home or refreshes
    session.pop('visualization_questions_state', None)
    session.pop('uploaded_filepath', None)
    session.pop('uploaded_filename', None)
    session.pop('df_columns', None)
    session.pop('user_answer_variable_types', None)
    session.pop('user_answer_visualization_message', None)
    session.pop('user_answer_variable_count', None)
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    response_data = {}
    bot_reply = "" 

    current_viz_state = session.get('visualization_questions_state')

    if current_viz_state == 'asking_variable_types':
        session['user_answer_variable_types'] = user_input
        bot_reply = (
            f"Understood. You're working with: \"{user_input}\".<br><br>"
            f"<strong>2. What message or insight do you want your visualization to communicate?</strong>"
            f"<br>(e.g., compare values across categories, show data distribution, identify relationships, track trends over time)"
        )
        session['visualization_questions_state'] = 'asking_visualization_message'
        response_data = {
            "suggestions": [
                "Compare values/categories",
                "Show data distribution",
                "Identify relationships",
                "Track trends over time"
            ]
        }
    elif current_viz_state == 'asking_visualization_message':
        session['user_answer_visualization_message'] = user_input
        df_columns = session.get('df_columns', [])
        
        columns_list_str = ""
        if df_columns:
            columns_list_str = "<ul>" + "".join([f"<li>{col}</li>" for col in df_columns]) + "</ul>"
            columns_reminder = f"For reference, columns in <strong>{session.get('uploaded_filename', 'your file')}</strong> include: {columns_list_str}"
        else:
            columns_reminder = "<p>(Could not retrieve column list from the uploaded file.)</p>"


        bot_reply = (
            f"Great! The goal is to: \"{user_input}\".<br><br>"
            f"<strong>3. How many variables would you typically like to visualize in a single chart?</strong>"
            f"<br>(e.g., one for a histogram, two for a scatter plot, etc.)<br><br>"
            f"{columns_reminder}"
        )
        session['visualization_questions_state'] = 'asking_variable_count'
        response_data = {
            "suggestions": ["One variable", "Two variables", "Three variables", "More than three"]
        }
    elif current_viz_state == 'asking_variable_count':
        session['user_answer_variable_count'] = user_input
        bot_reply = (
            f"Excellent! Here’s a summary of your preferences:<br>"
            f"- <strong>Variable Types:</strong> \"{session.get('user_answer_variable_types', 'N/A')}\"<br>"
            f"- <strong>Desired Message/Insight:</strong> \"{session.get('user_answer_visualization_message', 'N/A')}\"<br>"
            f"- <strong>Variables per Chart:</strong> \"{user_input}\"<br><br>"
            f"What would you like to do next with this information?"
        )
        session['visualization_questions_state'] = 'visualization_info_gathered'
        response_data = {
            "suggestions": [
                "Suggest chart types for me",
                "Let me choose columns to plot",
                "Restart these visualization questions"
            ]
        }
    elif current_viz_state == 'visualization_info_gathered':
        if "suggest chart types" in user_input.lower():
            # Placeholder for actual chart suggestion logic
            # This logic would analyze session['user_answer_*'], session['df_columns']
            bot_reply = ("Okay! Based on your answers, I can help pick some suitable chart types. "
                         "(Actual chart suggestion logic is the next step to implement!)")
            session.pop('visualization_questions_state', None) # End this specific flow
            response_data = {"suggestions": ["Upload new data", "Thanks, that's all!"]}
        elif "choose columns" in user_input.lower():
            df_columns = session.get('df_columns', [])
            if df_columns:
                bot_reply = (f"Sure! Which columns from <strong>{session.get('uploaded_filename')}</strong> "
                             f"would you like to use for the visualization? Please list them or click below.")
                suggestions = [f"Use: {col}" for col in df_columns[:min(len(df_columns), 4)]] 
                suggestions.append("Finished selecting columns")
            else:
                bot_reply = "It seems I don't have the column list for your file. Could you perhaps upload the file again?"
                suggestions = ["Upload Data"]
            session['visualization_questions_state'] = 'awaiting_column_selection' # New state
            response_data = {"suggestions": suggestions}
        elif "restart" in user_input.lower():
            # Reset answers and go back to the first question
            session.pop('user_answer_variable_types', None)
            session.pop('user_answer_visualization_message', None)
            session.pop('user_answer_variable_count', None)
            bot_reply = ("No problem, let's start the visualization questions again.<br><br>"
                         f"<strong>1. What types of variables are you working with primarily?</strong>"
                         f"<br>(e.g., categories, numbers, dates/times)")
            session['visualization_questions_state'] = 'asking_variable_types'
            response_data = {
                "suggestions": ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of types"]
            }
        else: 
            bot_reply = "Please select one of the provided options, or type 'Restart visualization questions'."
            response_data = { "suggestions": session.get('last_suggestions', ["Suggest chart types for me", "Let me choose columns to plot", "Restart these visualization questions"]) }
    elif current_viz_state == 'awaiting_column_selection':
        # Basic placeholder for handling column selection by the user
        if "finished selecting" in user_input.lower():
            bot_reply = (f"Great, you've indicated you're done selecting columns. "
                         f"(The next step would be to generate a plot with these columns - this feature is pending!)")
            session.pop('visualization_questions_state', None) 
            response_data = {"suggestions": ["Upload new data", "Thanks!"]}
        else:
            # This part would ideally parse column names from user_input
            bot_reply = f"Okay, noted: \"{user_input}\". You can list more columns or click 'Finished selecting columns'."
            df_columns = session.get('df_columns', [])
            suggestions = [f"Use: {col}" for col in df_columns[:min(len(df_columns), 4)] if col.lower() not in user_input.lower()] # Avoid re-suggesting what was typed
            suggestions.append("Finished selecting columns")
            response_data = {"suggestions": suggestions}

    else:
        # Not in a specific visualization question flow, use the NLU
        nlu_response_output = nlu_get_bot_response(user_input) # Assuming this returns a dict or can be adapted
        
        if isinstance(nlu_response_output, dict):
            bot_reply = nlu_response_output.get("response", "Sorry, I had trouble understanding that.")
            suggestions = nlu_response_output.get("suggestions", [])
        else: # Assuming it's a string response from the original bot_logic.py
            bot_reply = nlu_response_output
            suggestions = []

        if not suggestions: # Provide generic suggestions if NLU doesn't
            suggestions = ["Help", "Upload Data", "What can you do?"]
        response_data = {"suggestions": suggestions}

    response_data["response"] = bot_reply
    session['last_suggestions'] = response_data.get("suggestions", []) # Store last suggestions for fallback
    return jsonify(response_data)


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"response": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"response": "No file selected for upload."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Clear previous file's session data before processing new one
        session.pop('uploaded_filepath', None)
        session.pop('uploaded_filename', None)
        session.pop('df_columns', None)
        session.pop('user_answer_variable_types', None)
        session.pop('user_answer_visualization_message', None)
        session.pop('user_answer_variable_count', None)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            session['uploaded_filepath'] = filepath
            session['uploaded_filename'] = filename

            if filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            session['df_columns'] = list(df.columns) 

            preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)

            total_rows = len(df)
            total_columns = len(df.columns)
            missing_values = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            total_cells = total_rows * total_columns
            missing_percent = (missing_values / total_cells) * 100 if total_cells else 0

            initial_bot_message = (
                f"✅ <strong>{filename}</strong> has been uploaded and previewed successfully.<br><br>"
                f"🔍 Here's a quick data quality check:<br>"
                f"- Rows: <strong>{total_rows}</strong><br>"
                f"- Columns: <strong>{total_columns}</strong><br>"
                f"- Missing values: <strong>{missing_values}</strong> ({missing_percent:.2f}%)<br>"
                f"- Duplicate rows: <strong>{duplicate_rows}</strong><br><br>"
                f"Great! To help me suggest the best visualization for your data, I have a few questions:<br><br>"
                f"<strong>1. What types of variables are you working with primarily?</strong>"
                f"<br>(e.g., categories/text, numbers, dates/times)"
            )
            
            session['visualization_questions_state'] = 'asking_variable_types' 

            current_suggestions = [
                    "Categorical (text, groups)",
                    "Numerical (numbers, counts)",
                    "Time-series (dates/times)",
                    "A mix of these types"
                ]
            session['last_suggestions'] = current_suggestions


            return jsonify({
                "response": initial_bot_message,
                "preview": preview_html,
                "suggestions": current_suggestions
            })

        except Exception as e:
            session.pop('visualization_questions_state', None) # Clear state on error
            return jsonify({"response": f"File '{filename}' was uploaded, but there was an error processing it: {str(e)}"}), 500
    else:
        return jsonify({"response": "This file type isn't allowed. Please upload a CSV or Excel file (e.g., .csv, .xls, .xlsx)."}), 400

if __name__ == "__main__":
    app.run(debug=True)