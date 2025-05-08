import base64
import io
import os
from typing import List, Optional  # Added Typing for hints

import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from pandas.api.types import (is_datetime64_any_dtype, is_numeric_dtype,
                              is_string_dtype)
from werkzeug.utils import secure_filename

# Matplotlib and Seaborn setup for plotting
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend, crucial for web servers
import matplotlib.pyplot as plt
import seaborn as sns

# Import your bot logic (ensure bot_logic.py is in the same directory)
from bot_logic import get_bot_response as nlu_get_bot_response

app = Flask(__name__)
# IMPORTANT: Use a persistent, strong secret key from environment or config in production
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_simplified_column_types(df):
    """Analyzes DataFrame columns and returns simplified types."""
    simplified_types = {}
    if df is None or df.empty:
        return simplified_types
    for col in df.columns:
        try: # Add try-except for robustness against weird column data
            if is_numeric_dtype(df[col]):
                # Heuristic: treat as categorical if few unique numeric values
                unique_count = df[col].nunique()
                if unique_count < 10 and (unique_count < len(df) / 10 or len(df) < 50):
                    simplified_types[col] = 'categorical_numeric'
                else:
                    simplified_types[col] = 'numerical'
            elif is_datetime64_any_dtype(df[col]):
                simplified_types[col] = 'datetime'
            elif is_string_dtype(df[col]) or df[col].dtype == 'object':
                 # Heuristic: treat as categorical if not too many unique values
                 unique_count = df[col].nunique()
                 if unique_count <= 1: # Treat single-value 'object' cols potentially differently if needed
                      simplified_types[col] = 'categorical' # or 'constant' ?
                 elif unique_count < len(df) * 0.8 and unique_count < 100:
                     simplified_types[col] = 'categorical'
                 else:
                     simplified_types[col] = 'id_like_text' # High cardinality text
            else:
                simplified_types[col] = 'other'
        except Exception as e:
             print(f"Warning: Could not determine type for column '{col}': {e}")
             simplified_types[col] = 'other' # Fallback on error
    return simplified_types

def suggest_charts_based_on_answers(user_answers, df_sample):
    """Suggests chart types based on user preferences and data sample."""
    suggestions = []
    if df_sample is None or df_sample.empty:
        return [{"name": "Cannot suggest charts without data insight.", "type": "Info", "reason": "Data sample is empty.", "required_cols_specific": []}]

    col_types = get_simplified_column_types(df_sample)
    numerical_cols = [col for col, typ in col_types.items() if typ == 'numerical']
    categorical_cols = [col for col, typ in col_types.items() if typ in ['categorical', 'categorical_numeric']]
    datetime_cols = [col for col, typ in col_types.items() if typ == 'datetime']

    ua_var_count_str = user_answers.get('variable_count', '').lower()
    ua_var_types = user_answers.get('variable_types', '').lower()
    ua_message = user_answers.get('message_insight', '').lower()

    # --- Suggestion Rules (refined based on previous logic) ---
    # Rule: Distribution of ONE NUMERICAL variable
    if ("one" in ua_var_count_str or "1" in ua_var_count_str) and \
       ("distribut" in ua_message or "spread" in ua_message or "summary" in ua_message) and \
       ("numer" in ua_var_types or "any" in ua_var_types or not ua_var_types) and numerical_cols:
        for col in numerical_cols:
            suggestions.append({"name": "Histogram", "for_col": col, "type": "Univariate Numerical", "reason": f"Shows distribution of '{col}'.", "required_cols_specific": [col]})
            suggestions.append({"name": "Box Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Summarizes '{col}' (median, quartiles, outliers).", "required_cols_specific": [col]})
            suggestions.append({"name": "Density Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Smoothly shows distribution of '{col}'.", "required_cols_specific": [col]})

    # Rule: Frequency/Proportion of ONE CATEGORICAL variable
    if ("one" in ua_var_count_str or "1" in ua_var_count_str) and \
       ("proportion" in ua_message or "share" in ua_message or "frequency" in ua_message or "count" in ua_message or "values" in ua_message) and \
       ("categor" in ua_var_types or "any" in ua_var_types or not ua_var_types) and categorical_cols:
        for col in categorical_cols:
            suggestions.append({"name": "Bar Chart (Counts)", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows counts for each category in '{col}'.", "required_cols_specific": [col]})
            # Check uniqueness again just in case df_sample was small
            try:
                if df_sample[col].nunique() < 8 and df_sample[col].nunique() > 1:
                     suggestions.append({"name": "Pie Chart", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows proportions for '{col}'. Best for few categories.", "required_cols_specific": [col]})
            except: pass # Ignore error if nunique fails on sample

    # Rule: TWO NUMERICAL variables (Relationship)
    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
       ("relationship" in ua_message or "correlat" in ua_message or "scatter" in ua_message) and \
       ("numer" in ua_var_types or "any" in ua_var_types or not ua_var_types) and len(numerical_cols) >= 2:
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                suggestions.append({"name": "Scatter Plot", "for_cols": f"{col1} & {col2}", "type": "Bivariate Numerical-Numerical", "reason": f"Shows relationship between '{col1}' and '{col2}'.", "required_cols_specific": [col1, col2]})

    # Rule: ONE NUMERICAL vs ONE CATEGORICAL (Comparison)
    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
       ("compare" in ua_message or "across categories" in ua_message or "group by" in ua_message or "distribution" in ua_message) and \
       ("mix" in ua_var_types or "categor" in ua_var_types or "numer" in ua_var_types or "any" in ua_var_types or not ua_var_types) and numerical_cols and categorical_cols:
        for num_col in numerical_cols:
            for cat_col in categorical_cols:
                suggestions.append({"name": "Box Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares distribution of '{num_col}' across '{cat_col}' categories.", "required_cols_specific": [cat_col, num_col]})
                suggestions.append({"name": "Violin Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares distribution (with density) of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})
                suggestions.append({"name": "Bar Chart (Aggregated)", "for_cols": f"Avg/Sum of {num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares average/sum of '{num_col}' for each category in '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})

    # Rule: TWO CATEGORICAL variables
    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
        ("relationship" in ua_message or "compare" in ua_message or "contingency" in ua_message or "joint" in ua_message) and \
        ("categor" in ua_var_types or "any" in ua_var_types or not ua_var_types) and len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                suggestions.append({"name": "Grouped Bar Chart", "for_cols": f"{col1} & {col2}", "type": "Bivariate Categorical-Categorical", "reason": f"Shows counts of '{col1}' grouped by '{col2}'.", "required_cols_specific": [col1, col2]})
                suggestions.append({"name": "Heatmap (Counts)", "for_cols": f"{col1} & {col2}", "type": "Bivariate Categorical-Categorical", "reason": f"Shows co-occurrence frequency of '{col1}' and '{col2}'.", "required_cols_specific": [col1, col2]})

    # Rule: TIME SERIES (DATETIME vs NUMERICAL)
    if (("two" in ua_var_count_str or "2" in ua_var_count_str) or "time" in ua_var_types or "trend" in ua_message) and \
       datetime_cols and numerical_cols:
        for dt_col in datetime_cols:
            for num_col in numerical_cols:
                suggestions.append({"name": "Line Chart", "for_cols": f"{num_col} over {dt_col}", "type": "Time Series", "reason": f"Shows trend of '{num_col}' over '{dt_col}'.", "required_cols_specific": [dt_col, num_col]})
                suggestions.append({"name": "Area Chart", "for_cols": f"{num_col} over {dt_col}", "type": "Time Series", "reason": f"Shows cumulative trend/magnitude of '{num_col}' over '{dt_col}'.", "required_cols_specific": [dt_col, num_col]})

    # Rule: MULTIPLE VARIABLES
    if ("more" in ua_var_count_str or "multiple" in ua_var_count_str or "pair plot" in ua_message or "heatmap" in ua_message or "parallel" in ua_message) or \
       ((ua_var_count_str not in ["one", "1", "two", "2"]) and (len(numerical_cols) > 2 or len(categorical_cols) > 2) ):
        if len(numerical_cols) >= 3:
            suggestions.append({"name": "Pair Plot", "type": "Multivariate", "reason": "Shows pairwise relationships between numerical variables.", "required_cols_specific": numerical_cols[:min(4, len(numerical_cols))] })
            suggestions.append({"name": "Correlation Heatmap", "type": "Multivariate", "reason": "Shows correlation matrix for numerical variables.", "required_cols_specific": numerical_cols })
        if len(numerical_cols) >=3:
            suggestions.append({"name": "Parallel Coordinates Plot", "type": "Multivariate", "reason": "Compares multiple numerical variables across records.", "required_cols_specific": numerical_cols[:min(6, len(numerical_cols))]})

    final_suggestions_dict = {}
    for s in suggestions:
        s_key = s["name"] + ("_" + "_".join(sorted(s.get("required_cols_specific",[]))) if s.get("required_cols_specific") else "") # Handle missing key
        if s_key not in final_suggestions_dict:
            final_suggestions_dict[s_key] = s
    final_suggestions = list(final_suggestions_dict.values())

    if not final_suggestions:
        final_suggestions.append({"name": "No specific chart matched well", "type": "Info", "reason": "Your criteria didn't closely match common chart types. You can try picking columns manually.", "required_cols_specific": []})

    manual_pick_exists = any(s['name'] == "Pick columns manually" for s in final_suggestions)
    if not manual_pick_exists:
        final_suggestions.append({"name": "Pick columns manually", "type": "Action", "reason": "If you have a specific chart in mind or want to explore freely.", "required_cols_specific": []})
    return final_suggestions

# --- Validation Function ---
def validate_columns_for_chart(chart_type: str, columns: List[str], df: pd.DataFrame) -> Optional[str]:
    """Validates if columns are suitable for chart type. Returns error message or None."""
    if not columns: return "No columns were selected."
    if not all(col in df.columns for col in columns):
        missing = [col for col in columns if col not in df.columns]
        return f"Column(s) not found: {', '.join(missing)}."

    col_types = get_simplified_column_types(df[columns])
    num_numerical = sum(1 for typ in col_types.values() if typ == 'numerical')
    num_categorical = sum(1 for typ in col_types.values() if typ in ['categorical', 'categorical_numeric'])
    num_datetime = sum(1 for typ in col_types.values() if typ == 'datetime')
    num_selected = len(columns)

    requirements = {
        "Histogram": {'exact_cols': 1, 'numerical': 1}, "Box Plot": {'exact_cols': 1, 'numerical': 1},
        "Density Plot": {'exact_cols': 1, 'numerical': 1}, "Bar Chart (Counts)": {'exact_cols': 1, 'categorical': 1},
        "Pie Chart": {'exact_cols': 1, 'categorical': 1}, "Scatter Plot": {'exact_cols': 2, 'numerical': 2},
        "Line Chart": {'exact_cols': 2, 'numerical': (1, 2)}, # Needs at least 1 numerical
        "Box Plots (by Category)": {'exact_cols': 2, 'categorical': 1, 'numerical': 1},
        "Violin Plots (by Category)": {'exact_cols': 2, 'categorical': 1, 'numerical': 1},
        "Bar Chart (Aggregated)": {'exact_cols': 2, 'categorical': 1, 'numerical': 1},
        "Grouped Bar Chart": {'exact_cols': 2, 'categorical': 2}, "Heatmap (Counts)": {'exact_cols': 2, 'categorical': 2},
        "Area Chart": {'exact_cols': 2, 'numerical': (1, 2)}, # Needs at least 1 numerical
        "Pair Plot": {'min_cols': 3, 'numerical': 3}, # Min 3 numerical
        "Correlation Heatmap": {'min_cols': 2, 'numerical': 2}, # Min 2 numerical
         "Parallel Coordinates Plot": {'min_cols': 3, 'numerical': 3} # Min 3 numerical
    }

    if chart_type not in requirements: return None # Skip validation if unknown

    req = requirements[chart_type]

    if 'exact_cols' in req and num_selected != req['exact_cols']: return f"needs exactly {req['exact_cols']} column(s), you chose {num_selected}."
    if 'min_cols' in req and num_selected < req['min_cols']: return f"needs at least {req['min_cols']} columns, you chose {num_selected}."
    if isinstance(req.get('numerical'), int) and req['numerical'] > num_numerical: return f"needs {req['numerical']} numerical column(s), you chose {num_numerical}."
    if isinstance(req.get('categorical'), int) and req['categorical'] > num_categorical: return f"needs {req['categorical']} categorical column(s), you chose {num_categorical}."
    if isinstance(req.get('datetime'), int) and req['datetime'] > num_datetime: return f"needs {req['datetime']} datetime column(s), you chose {num_datetime}."
    if isinstance(req.get('numerical'), tuple) and not (req['numerical'][0] <= num_numerical <= req['numerical'][1]): return f"needs {req['numerical'][0]}-{req['numerical'][1]} numerical cols, you chose {num_numerical}."
    if isinstance(req.get('datetime'), tuple) and not (req['datetime'][0] <= num_datetime <= req['datetime'][1]): return f"needs {req['datetime'][0]}-{req['datetime'][1]} datetime cols, you chose {num_datetime}."
    if chart_type in ["Line Chart", "Area Chart"] and num_numerical == 0: return "needs at least one numerical column."
    if chart_type in ["Box Plots (by Category)", "Violin Plots (by Category)", "Bar Chart (Aggregated)"] and (num_categorical == 0 or num_numerical == 0): return "needs one categorical and one numerical column."

    return None # Passed


# --- Plotting Function ---
def generate_plot_and_get_uri(filepath, chart_type, columns):
    """Generates plot and returns base64 URI or (None, error_msg)."""
    if not filepath: return None, "File path missing."
    try:
        # Read only necessary columns if possible
        try:
             df_full = pd.read_csv(filepath, usecols=columns) if filepath.endswith(".csv") else pd.read_excel(filepath, usecols=columns)
        except ValueError: # If usecols fails (e.g., column missing), read all to validate cols later
             print(f"Warning: Could not read only specific columns {columns}. Reading full file for validation.")
             df_full = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading full dataframe ('{filepath}') for plotting: {e}")
        return None, "Error reading the data file."

    # *** Run Validation ***
    validation_error = validate_columns_for_chart(chart_type, columns, df_full)
    if validation_error:
        print(f"Plotting Validation Error for {chart_type}: {validation_error}")
        return None, f"Invalid columns for {chart_type}: {validation_error}"

    # Subset dataframe AFTER validation if needed (though reading with usecols is better)
    if not all(col in df_full.columns for col in columns):
         missing_cols = [col for col in columns if col not in df_full.columns]
         return None, f"Column(s) not found: {', '.join(missing_cols)}"
    df_plot = df_full[columns].copy() # Work with a copy of needed cols

    img = io.BytesIO()
    plt.figure(figsize=(7, 4.5))
    plt.style.use('seaborn-v0_8-whitegrid')

    try:
        print(f"Attempting to generate plot: {chart_type} with columns: {columns}")
        if chart_type == "Histogram": sns.histplot(data=df_plot, x=columns[0], kde=True); plt.title(f"Histogram of {columns[0]}", fontsize=12)
        elif chart_type == "Box Plot": sns.boxplot(data=df_plot, y=columns[0]); plt.title(f"Box Plot of {columns[0]}", fontsize=12)
        elif chart_type == "Density Plot": sns.kdeplot(data=df_plot, x=columns[0], fill=True); plt.title(f"Density Plot of {columns[0]}", fontsize=12)
        elif chart_type == "Bar Chart (Counts)":
            counts = df_plot[columns[0]].value_counts().nlargest(15); sns.barplot(x=counts.index, y=counts.values); plt.title(f"Counts for {columns[0]}", fontsize=12); plt.ylabel("Count", fontsize=10); plt.xlabel(columns[0], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Pie Chart":
            counts = df_plot[columns[0]].value_counts(); effective_counts = counts.nlargest(6)
            if len(counts) > 6: effective_counts.loc['Other'] = counts.iloc[6:].sum()
            plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85); plt.title(f"Pie Chart of {columns[0]}", fontsize=12); plt.axis('equal')
        elif chart_type == "Scatter Plot":
            sns.scatterplot(data=df_plot, x=columns[0], y=columns[1]); plt.title(f"Scatter: {columns[0]} vs {columns[1]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10)
        elif chart_type == "Line Chart":
            df_to_plot = df_plot.copy()
            if pd.api.types.is_datetime64_any_dtype(df_to_plot[columns[0]]): df_to_plot = df_to_plot.sort_values(by=columns[0])
            elif pd.api.types.is_numeric_dtype(df_to_plot[columns[0]]): df_to_plot = df_to_plot.sort_values(by=columns[0])
            sns.lineplot(data=df_to_plot, x=columns[0], y=columns[1]); plt.title(f"Line: {columns[1]} over {columns[0]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Box Plots (by Category)":
            sns.boxplot(data=df_plot, x=columns[0], y=columns[1]); plt.title(f"Box Plots: {columns[1]} by {columns[0]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Violin Plots (by Category)":
            sns.violinplot(data=df_plot, x=columns[0], y=columns[1]); plt.title(f"Violin Plots: {columns[1]} by {columns[0]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Bar Chart (Aggregated)":
             col_types_specific = get_simplified_column_types(df_plot)
             cat_col = columns[0] if col_types_specific.get(columns[0]) in ['categorical', 'categorical_numeric'] else columns[1]
             num_col = columns[1] if col_types_specific.get(columns[1]) == 'numerical' else columns[0]
             agg_data = df_plot.groupby(cat_col)[num_col].mean().nlargest(15); sns.barplot(x=agg_data.index, y=agg_data.values); plt.title(f"Mean of {num_col} by {cat_col}", fontsize=12); plt.xlabel(cat_col, fontsize=10); plt.ylabel(f"Mean of {num_col}", fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        # --- Add more plotting types here ---
        else:
            print(f"Plot type '{chart_type}' is not implemented in generate_plot_and_get_uri.")
            return None, f"Plot type '{chart_type}' is not implemented."

        plt.tight_layout(pad=1.0)
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        print(f"Successfully generated plot: {chart_type}")
        return f"data:image/png;base64,{plot_url}", None

    except Exception as e:
        print(f"!!! Error during plot generation execution for '{chart_type}' with {columns}: {e}")
        if 'plt' in locals() and plt.get_fignums(): plt.close('all')
        error_message = f"Failed to generate {chart_type}. Check data types ({str(e)[:100]}...)." # Include part of error
        return None, error_message

# --- Flask Routes ---

@app.route("/")
def home():
    """Clears session and renders the main page."""
    session.pop('visualization_questions_state', None); session.pop('uploaded_filepath', None)
    session.pop('uploaded_filename', None); session.pop('df_columns', None)
    session.pop('user_answer_variable_types', None); session.pop('user_answer_visualization_message', None)
    session.pop('user_answer_variable_count', None); session.pop('chart_suggestions_list', None)
    session.pop('selected_chart_for_plotting', None); session.pop('plotting_columns', None)
    session.pop('manual_columns_selected', None); session.pop('last_suggestions', None)
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    """Handles user messages, state transitions, suggestions, and plotting calls."""
    user_input = request.json.get("message")
    response_data = {}
    bot_reply = ""

    current_viz_state = session.get('visualization_questions_state')
    df_columns = session.get('df_columns', [])
    uploaded_filepath = session.get('uploaded_filepath')
    user_answers = {
        'variable_types': session.get('user_answer_variable_types', ''),
        'message_insight': session.get('user_answer_visualization_message', ''),
        'variable_count': session.get('user_answer_variable_count', '')
    }

    # --- State Machine Logic ---
    if current_viz_state == 'asking_variable_types':
        session['user_answer_variable_types'] = user_input
        bot_reply = (f"Understood. You're working with: \"{user_input}\".<br><br>"
                     f"<strong>2. What message or insight do you want your visualization to communicate?</strong>"
                     f"<br>(e.g., compare values, show distribution, identify relationships, track trends)")
        session['visualization_questions_state'] = 'asking_visualization_message'
        response_data = {"suggestions": ["Compare values/categories", "Show data distribution", "Identify relationships", "Track trends over time"]}

    elif current_viz_state == 'asking_visualization_message':
        session['user_answer_visualization_message'] = user_input
        columns_reminder = "<p>(Column list unavailable. Please upload data again if needed.)</p>"
        if df_columns:
            columns_list_html = "<ul>" + "".join([f"<li>{col}</li>" for col in df_columns]) + "</ul>"
            columns_reminder = f"For reference, columns in <strong>{session.get('uploaded_filename', 'your file')}</strong> include: {columns_list_html}"
        bot_reply = (f"Great! The goal is to: \"{user_input}\".<br><br>"
                     f"<strong>3. How many variables would you typically like to visualize in a single chart?</strong>"
                     f"<br>(e.g., one, two, three, or 'more' for multivariate)<br><br>{columns_reminder}")
        session['visualization_questions_state'] = 'asking_variable_count'
        response_data = {"suggestions": ["One variable", "Two variables", "Three variables", "More than three"]}

    elif current_viz_state == 'asking_variable_count':
        session['user_answer_variable_count'] = user_input
        bot_reply = (f"Excellent! Here’s a summary of your preferences:<br>"
                     f"- Variable Types: \"{user_answers['variable_types']}\"<br>"
                     f"- Desired Message/Insight: \"{user_answers['message_insight']}\"<br>"
                     f"- Variables per Chart: \"{user_input}\"<br><br>"
                     f"What would you like to do next?")
        session['visualization_questions_state'] = 'visualization_info_gathered'
        response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot", "Restart these visualization questions"]}

    elif current_viz_state == 'visualization_info_gathered':
        if "suggest chart types" in user_input.lower():
            df_sample = None
            if uploaded_filepath:
                try:
                    df_sample = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e: print(f"Error reading df_sample for suggestions: {e}")
            chart_suggestions = suggest_charts_based_on_answers(user_answers, df_sample)
            session['chart_suggestions_list'] = chart_suggestions
            if chart_suggestions and chart_suggestions[0].get("type") != "Info":
                bot_reply = "Okay, based on your preferences and data, consider these chart types:<br>"
                suggestions_for_user_options = []
                count = 0
                for chart_sugg in chart_suggestions:
                    if chart_sugg.get("type") == "Action": continue
                    if count >= 4 : break
                    count += 1
                    bot_reply += f"<br><strong>{count}. {chart_sugg['name']}</strong>"
                    if chart_sugg.get("for_col"): bot_reply += f" (for '{chart_sugg['for_col']}')"
                    elif chart_sugg.get("for_cols"): bot_reply += f" (e.g., for '{chart_sugg['for_cols']}')"
                    bot_reply += f": {chart_sugg.get('reason', '')}"
                    suggestions_for_user_options.append(f"Select: {chart_sugg['name']}")
                bot_reply += "<br><br>Which one sounds interesting, or do you want to pick columns manually?"
                manual_pick_option = "Pick columns manually"
                if not any(sugg_opt == manual_pick_option for sugg_opt in suggestions_for_user_options):
                    if any(s['name'] == manual_pick_option for s in chart_suggestions): suggestions_for_user_options.append(manual_pick_option)
                response_data_suggestions = suggestions_for_user_options
                if manual_pick_option not in response_data_suggestions: response_data_suggestions.append(manual_pick_option)
                response_data_suggestions.append("Restart visualization questions")
                response_data = {"suggestions": response_data_suggestions[:5]}
                session['visualization_questions_state'] = 'awaiting_chart_type_selection'
            else:
                bot_reply = "I couldn't come up with specific chart suggestions. You can try picking columns manually."
                response_data = {"suggestions": ["Let me pick columns", "Restart visualization questions"]}
                session['visualization_questions_state'] = 'awaiting_column_selection_general'

        elif "choose columns" in user_input.lower() or "pick columns" in user_input.lower():
             if not df_columns:
                 bot_reply = "I need the column list for that. Please upload the data file again."
                 response_data = {"suggestions": ["Upload Data"]}
                 session['visualization_questions_state'] = None
             else:
                 bot_reply = "Sure! Which columns? Available:<br><ul>" + "".join([f"<li>{col}</li>" for col in df_columns]) + "</ul>Please list them (e.g., 'Age, Salary')."
                 session['visualization_questions_state'] = 'awaiting_column_selection_general'
                 session['manual_columns_selected'] = []
                 response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}

        elif "restart" in user_input.lower():
             home() # Clear session
             bot_reply = "Okay, let's start over.<br><br><strong>1. What types of variables...</strong>"
             session['visualization_questions_state'] = 'asking_variable_types' # Need to re-set after home()
             response_data = {"suggestions": ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of types"]}
        else:
             bot_reply = "Sorry, I didn't catch that. What next?"
             response_data = {"suggestions": session.get('last_suggestions', ["Suggest chart types for me", "Let me choose columns to plot", "Restart these visualization questions"])}

    elif current_viz_state == 'awaiting_chart_type_selection':
        user_choice_str = user_input.replace("Select: ", "").strip()
        chart_suggestions_list = session.get('chart_suggestions_list', [])
        selected_chart_info = next((sugg for sugg in chart_suggestions_list if sugg['name'] == user_choice_str), None)

        if user_choice_str == "Pick columns manually":
            if not df_columns:
                bot_reply = "I need the column list for that. Please upload data again."
                response_data = {"suggestions": ["Upload Data"]}
                session['visualization_questions_state'] = None
            else:
                 bot_reply = "Okay! Which columns? Available: (<strong>" + ", ".join(df_columns) + "</strong>)"
                 session['visualization_questions_state'] = 'awaiting_column_selection_general'
                 session['manual_columns_selected'] = []
                 response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}
        elif selected_chart_info:
            session['selected_chart_for_plotting'] = selected_chart_info
            chart_name = selected_chart_info['name']
            required_cols_specific = selected_chart_info.get('required_cols_specific')
            bot_reply = f"Great choice: <strong>{chart_name}</strong>. "
            if required_cols_specific and isinstance(required_cols_specific, list) and len(required_cols_specific) > 0 :
                cols_to_use_str = ", ".join(required_cols_specific)
                validation_msg = None
                if uploaded_filepath:
                    try:
                        df_val = pd.read_csv(uploaded_filepath, usecols=required_cols_specific, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=required_cols_specific, nrows=5)
                        validation_msg = validate_columns_for_chart(chart_name, required_cols_specific, df_val)
                    except Exception: validation_msg = "Could not validate columns automatically."
                if validation_msg is None:
                    bot_reply += f"Based on the data, I suggest columns: <strong>{cols_to_use_str}</strong>. Proceed?"
                    session['plotting_columns'] = required_cols_specific
                    response_data = {"suggestions": [f"Yes, plot {chart_name} with these", "Choose other columns", "Back to chart list"]}
                    session['visualization_questions_state'] = 'confirm_plot_details'
                else:
                     bot_reply += f"The columns suggested ({cols_to_use_str}) don't seem ideal for a {chart_name}. ({validation_msg}) <br>Please select appropriate columns. Available: {', '.join(df_columns)}"
                     session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'
                     response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns", "Back to chart list"]}
            else:
                 bot_reply += f"Which columns for the {chart_name}? Available: {', '.join(df_columns)}"
                 session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'
                 response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns", "Back to chart list"]}
        else:
            bot_reply = "I didn't recognize that selection. Please choose again."
            response_data = {"suggestions": session.get('last_suggestions', [])}

    elif current_viz_state == 'confirm_plot_details':
        chart_to_plot_info = session.get('selected_chart_for_plotting')
        cols_for_plot = session.get('plotting_columns')

        if user_input.startswith("Yes, plot"):
            if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                chart_name = chart_to_plot_info['name']
                bot_reply = f"Okay, generating <strong>{chart_name}</strong> with columns: {', '.join(cols_for_plot)}."
                plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_name, cols_for_plot)
                if plot_image_uri: response_data["plot_image"] = plot_image_uri
                else: bot_reply += f"<br><strong>Plotting Error:</strong> {error_msg or 'Unknown error.'}"
                session['visualization_questions_state'] = None
                response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart visualization questions", "Upload new data"])
            else:
                bot_reply = "Missing plot details/file path. Let's retry."
                session['visualization_questions_state'] = 'visualization_info_gathered'
                response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot"]}
        elif "choose other columns" in user_input.lower() or "change columns" in user_input.lower():
             chart_name = chart_to_plot_info['name'] if chart_to_plot_info else 'chart'
             bot_reply = f"Okay, for the <strong>{chart_name}</strong>, which columns instead? Available: {', '.join(df_columns)}"
             session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'
             response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns"]}
        else: # Back to chart list or other
            bot_reply = "Okay, let's go back to the chart list."
            # Need to reconstruct the chart list message and suggestions
            # For simplicity, go back to gathered state
            session['visualization_questions_state'] = 'visualization_info_gathered'
            response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot", "Restart visualization questions"]}


    elif current_viz_state == 'awaiting_columns_for_selected_chart':
        potential_cols_str = user_input.replace("Use:", "").strip()
        user_selected_cols = [col.strip() for col in potential_cols_str.split(',') if col.strip() and col.strip() in df_columns]
        chart_to_plot_info = session.get('selected_chart_for_plotting')
        chart_name = chart_to_plot_info.get('name', 'chart') if chart_to_plot_info else 'chart'

        if user_selected_cols:
            validation_msg = None
            if uploaded_filepath:
                 try:
                     df_val = pd.read_csv(uploaded_filepath, usecols=user_selected_cols, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=user_selected_cols, nrows=5)
                     validation_msg = validate_columns_for_chart(chart_name, user_selected_cols, df_val)
                 except Exception: validation_msg = "Could not validate columns automatically."
            if validation_msg is None:
                session['plotting_columns'] = user_selected_cols
                bot_reply = f"Using columns: <strong>{', '.join(user_selected_cols)}</strong> for the <strong>{chart_name}</strong>. Ready to plot?"
                response_data = {"suggestions": ["Yes, generate this plot", "Add/Change columns", "Back to chart list"]}
                session['visualization_questions_state'] = 'confirm_plot_details'
            else:
                 bot_reply = f"The columns <strong>{', '.join(user_selected_cols)}</strong> might not work for a <strong>{chart_name}</strong>. ({validation_msg})<br>Please select different columns from: {', '.join(df_columns)}"
                 response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns", "Back to chart list"]}
        elif "list all columns" in user_input.lower():
             bot_reply = "Available columns: " + ", ".join(df_columns) + f"<br>Which ones for the {chart_name}?"
             response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        elif "back to chart list" in user_input.lower():
            session['visualization_questions_state'] = 'awaiting_chart_type_selection' # Go back to chart selection
            bot_reply = "Okay, going back. Which chart type were you interested in?"
             # Reshow previous chart suggestions (simple version)
            chart_suggestions_list = session.get('chart_suggestions_list', [])
            temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]
            temp_suggs.append("Pick columns manually")
            response_data = {"suggestions": temp_suggs}
        else:
             bot_reply = f"Invalid column names. For <strong>{chart_name}</strong>, choose from: {', '.join(df_columns)}"
             response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns", "Back to chart list"]}

    elif current_viz_state == 'awaiting_column_selection_general':
        if "finished selecting columns" in user_input.lower():
            selected_cols = session.get('manual_columns_selected', [])
            if selected_cols:
                bot_reply = f"Okay, you selected: <strong>{', '.join(selected_cols)}</strong>. What kind of chart?"
                session['plotting_columns'] = selected_cols
                session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'
                chart_type_suggestions = ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Table View"]
                response_data = {"suggestions": chart_type_suggestions}
            else:
                bot_reply = "No columns selected. Please list columns or cancel."
                response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}
        elif "cancel selection" in user_input.lower():
            session.pop('manual_columns_selected', None)
            session['visualization_questions_state'] = 'visualization_info_gathered'
            bot_reply = "Column selection cancelled."
            response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot"]}
        else:
            potential_col = user_input.replace("Use:","").strip()
            current_selection = session.get('manual_columns_selected', [])
            if potential_col in df_columns:
                if potential_col not in current_selection: current_selection.append(potential_col)
                session['manual_columns_selected'] = current_selection
                bot_reply = f"Added '<strong>{potential_col}</strong>'. Selected: <strong>{', '.join(current_selection) if current_selection else 'None'}</strong>.<br>Add more, or 'Finished selecting columns'."
                remaining_cols_suggestions = [f"Use: {col}" for col in df_columns if col not in current_selection][:2]
                response_data = {"suggestions": remaining_cols_suggestions + ["Finished selecting columns", "Cancel selection"]}
            else:
                bot_reply = f"'{potential_col}' is not valid. Choose from: {', '.join(df_columns)}."
                response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}

    elif current_viz_state == 'awaiting_chart_type_for_manual_cols':
        chart_type_from_user = user_input.strip()
        cols_for_plot = session.get('plotting_columns', [])
        if cols_for_plot and uploaded_filepath:
            bot_reply = f"Okay, attempting <strong>{chart_type_from_user}</strong> with: {', '.join(cols_for_plot)}."
            plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_type_from_user, cols_for_plot)
            if plot_image_uri: response_data["plot_image"] = plot_image_uri
            else: bot_reply += f"<br><strong>Plotting Error:</strong> {error_msg or 'Unknown error.'}"
        else:
            bot_reply = "Missing column/file details. Let's retry."
        session['visualization_questions_state'] = None
        response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart visualization questions", "Upload new data"])

    else: # Fallback to NLU
        nlu_output = nlu_get_bot_response(user_input)
        if isinstance(nlu_output, dict):
            bot_reply = nlu_output.get("response", "Sorry, I had trouble understanding.")
            temp_suggestions = nlu_output.get("suggestions", [])
        else:
            bot_reply = nlu_output; temp_suggestions = []
        if not temp_suggestions: temp_suggestions = ["Help", "Upload Data", "What can you do?"]
        response_data = {"suggestions": temp_suggestions}

    response_data["response"] = bot_reply
    session['last_suggestions'] = response_data.get("suggestions", [])
    return jsonify(response_data)

@app.route("/upload_file", methods=["POST"])
def upload_file():
    """Handles file upload, validation, and starts the question flow."""
    if "file" not in request.files: return jsonify({"response": "No file part in request."}), 400
    file = request.files["file"]
    if file.filename == "": return jsonify({"response": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        home() # Clear session before processing new file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            session['uploaded_filepath'] = filepath
            session['uploaded_filename'] = filename

            # Read file to get columns and preview
            df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
            session['df_columns'] = list(df.columns)
            preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)

            # Data Quality Check
            total_rows, total_columns = len(df), len(df.columns)
            missing_values = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            total_cells = total_rows * total_columns
            missing_percent = (missing_values / total_cells) * 100 if total_cells else 0

            initial_bot_message = (
                f"✅ <strong>{filename}</strong> uploaded.<br><br>"
                f"🔍 Quality Check: {total_rows} rows, {total_columns} cols; "
                f"{missing_values} missing ({missing_percent:.1f}%); {duplicate_rows} duplicates.<br><br>"
                f"Let's choose a visualization. I need a bit more info:<br><br>"
                f"<strong>1. What types of variables?</strong> (e.g., text, numbers, dates)"
            )
            session['visualization_questions_state'] = 'asking_variable_types'
            current_suggestions = ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of these types", "Not sure / Any"]
            session['last_suggestions'] = current_suggestions
            return jsonify({"response": initial_bot_message, "preview": preview_html, "suggestions": current_suggestions})

        except Exception as e:
            home() # Clear session on error
            print(f"Error processing uploaded file {filename}: {e}")
            return jsonify({"response": f"Error processing '{filename}': {str(e)[:100]}..."}), 500 # Show part of error
    else:
        return jsonify({"response": "Invalid file type. Use CSV or Excel."}), 400

if __name__ == "__main__":
    # Set seaborn style globally once
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'figure.autolayout': True}) # Helps with layout
    app.run(debug=True) # debug=False for production