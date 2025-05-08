import base64
import io
import os
from typing import List, Optional # Added Typing for hints

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
        try:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            non_null_count = df[col].count() # Count non-null values

            if non_null_count == 0: # Handle fully empty columns
                 simplified_types[col] = 'empty'
                 continue

            if is_numeric_dtype(dtype):
                # Refined Heuristic for categorical numeric:
                # Low unique count relative to non-nulls OR very few unique values overall
                if unique_count < 15 and (unique_count < non_null_count * 0.1 or unique_count < 7):
                    simplified_types[col] = 'categorical_numeric'
                else:
                    simplified_types[col] = 'numerical'
            elif is_datetime64_any_dtype(dtype):
                simplified_types[col] = 'datetime'
            elif is_string_dtype(dtype) or dtype == 'object':
                 # Refined Heuristic for categorical text:
                 # Low unique count relative to non-nulls AND not excessively high absolute count
                 if unique_count <= 1:
                      simplified_types[col] = 'categorical' # Treat constant as categorical for plotting counts
                 elif unique_count < non_null_count * 0.8 and unique_count < 150:
                     simplified_types[col] = 'categorical'
                 else:
                     simplified_types[col] = 'id_like_text' # High cardinality text
            else:
                simplified_types[col] = 'other'
        except Exception as e:
             print(f"Warning: Could not determine type for column '{col}': {e}")
             simplified_types[col] = 'other'
    # print(f"DEBUG Simplified Types: {simplified_types}") # Optional: uncomment for deep debug
    return simplified_types

def suggest_charts_based_on_answers(user_answers, df_sample):
    """Suggests chart types based on user preferences and data sample."""
    suggestions = []
    if df_sample is None or df_sample.empty:
        return [{"name": "Cannot suggest charts without data insight.", "type": "Info", "reason": "Data sample is empty.", "required_cols_specific": []}]

    col_types = get_simplified_column_types(df_sample)
    numerical_cols = [col for col, typ in col_types.items() if typ == 'numerical']
    # Treat categorical_numeric as suitable for *some* categorical roles AND distribution plots
    categorical_cols = [col for col, typ in col_types.items() if typ in ['categorical', 'categorical_numeric']]
    distributable_numeric_cols = [col for col, typ in col_types.items() if typ in ['numerical', 'categorical_numeric']]
    datetime_cols = [col for col, typ in col_types.items() if typ == 'datetime']

    ua_var_count_str = user_answers.get('variable_count', '').lower()
    ua_var_types = user_answers.get('variable_types', '').lower()
    ua_message = user_answers.get('message_insight', '').lower()

    # --- Suggestion Rules ---

    # Rule: Distribution of ONE NUMERICAL/RATING variable
    if ("one" in ua_var_count_str or "1" in ua_var_count_str) and \
       ("distribut" in ua_message or "spread" in ua_message or "summary" in ua_message) and \
       ("numer" in ua_var_types or "categor" in ua_var_types or "any" in ua_var_types or not ua_var_types) and distributable_numeric_cols:
        for col in distributable_numeric_cols:
            suggestions.append({"name": "Histogram", "for_col": col, "type": "Univariate Numerical", "reason": f"Shows distribution of '{col}'.", "required_cols_specific": [col]})
            suggestions.append({"name": "Box Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Summarizes '{col}' (median, quartiles, outliers).", "required_cols_specific": [col]})
            suggestions.append({"name": "Density Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Smoothly shows distribution of '{col}'.", "required_cols_specific": [col]})

    # Rule: Frequency/Proportion of ONE CATEGORICAL variable (includes categorical_numeric)
    if ("one" in ua_var_count_str or "1" in ua_var_count_str) and \
       ("proportion" in ua_message or "share" in ua_message or "frequency" in ua_message or "count" in ua_message or "values" in ua_message) and \
       ("categor" in ua_var_types or "any" in ua_var_types or not ua_var_types) and categorical_cols:
        for col in categorical_cols:
            suggestions.append({"name": "Bar Chart (Counts)", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows counts for each category in '{col}'.", "required_cols_specific": [col]})
            try:
                # Allow pie for categorical_numeric too if few unique values
                if df_sample[col].nunique() < 8 and df_sample[col].nunique() > 1 :
                     suggestions.append({"name": "Pie Chart", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows proportions for '{col}'. Best for few categories.", "required_cols_specific": [col]})
            except: pass

    # Rule: TWO NUMERICAL variables (Relationship)
    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
       ("relationship" in ua_message or "correlat" in ua_message or "scatter" in ua_message) and \
       ("numer" in ua_var_types or "any" in ua_var_types or not ua_var_types) and len(numerical_cols) >= 2:
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                suggestions.append({"name": "Scatter Plot", "for_cols": f"{col1} & {col2}", "type": "Bivariate Numerical-Numerical", "reason": f"Shows relationship between '{col1}' and '{col2}'.", "required_cols_specific": [col1, col2]})

    # Rule: ONE NUMERICAL/DISTRIBUTABLE vs ONE CATEGORICAL (Comparison)
    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
       ("compare" in ua_message or "across categories" in ua_message or "group by" in ua_message or "distribution" in ua_message) and \
       ("mix" in ua_var_types or "categor" in ua_var_types or "numer" in ua_var_types or "any" in ua_var_types or not ua_var_types) and distributable_numeric_cols and categorical_cols:
        for num_col in distributable_numeric_cols: # Use distributable here for Box/Violin
            for cat_col in categorical_cols:
                # Check they are not the same column if one was classified as both (unlikely with current logic but safe)
                if num_col == cat_col: continue
                suggestions.append({"name": "Box Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares distribution of '{num_col}' across '{cat_col}' categories.", "required_cols_specific": [cat_col, num_col]})
                suggestions.append({"name": "Violin Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares distribution (with density) of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})
        # Bar chart aggregation still needs strictly numerical
        if numerical_cols and categorical_cols:
            for num_col in numerical_cols:
                 for cat_col in categorical_cols:
                      if num_col == cat_col: continue
                      suggestions.append({"name": "Bar Chart (Aggregated)", "for_cols": f"Avg/Sum of {num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares average/sum of '{num_col}' for each category in '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})


    # --- (Keep other rules for Cat-Cat, Time Series, Multivariate as before) ---
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

    # --- (Keep final suggestion formatting and deduplication logic) ---
    final_suggestions_dict = {}
    for s in suggestions:
        s_key = s["name"] + ("_" + "_".join(sorted(s.get("required_cols_specific",[]))) if s.get("required_cols_specific") else "")
        if s_key not in final_suggestions_dict:
            final_suggestions_dict[s_key] = s
    final_suggestions = list(final_suggestions_dict.values())

    if not final_suggestions:
        final_suggestions.append({"name": "No specific chart matched well", "type": "Info", "reason": "Your criteria didn't closely match common chart types. You can try picking columns manually.", "required_cols_specific": []})

    manual_pick_exists = any(s['name'] == "Pick columns manually" for s in final_suggestions)
    if not manual_pick_exists:
        final_suggestions.append({"name": "Pick columns manually", "type": "Action", "reason": "If you have a specific chart in mind or want to explore freely.", "required_cols_specific": []})
    return final_suggestions

# --- Validation Function (UPDATED VERSION) ---
def validate_columns_for_chart(chart_type: str, columns: List[str], df: pd.DataFrame) -> Optional[str]:
    """Validates if columns are suitable for chart type. Returns error message or None."""
    if not columns: return "No columns were selected."
    if not all(col in df.columns for col in columns):
        missing = [col for col in columns if col not in df.columns]
        return f"Column(s) not found: {', '.join(missing)}."

    col_types = get_simplified_column_types(df[columns])

    # --- Count relevant types ---
    num_numerical = sum(1 for typ in col_types.values() if typ == 'numerical')
    num_categorical = sum(1 for typ in col_types.values() if typ in ['categorical', 'categorical_numeric'])
    num_distributable_numeric = sum(1 for typ in col_types.values() if typ in ['numerical', 'categorical_numeric']) # Counts both types for distribution charts
    num_datetime = sum(1 for typ in col_types.values() if typ == 'datetime')
    num_selected = len(columns)

    # --- Define requirements (Using 'distributable_numeric' where appropriate) ---
    requirements = {
        "Histogram": {'exact_cols': 1, 'distributable_numeric': 1}, # Use combined type
        "Box Plot": {'exact_cols': 1, 'distributable_numeric': 1}, # Use combined type
        "Density Plot": {'exact_cols': 1, 'distributable_numeric': 1}, # Use combined type
        "Bar Chart (Counts)": {'exact_cols': 1, 'categorical': 1},
        "Pie Chart": {'exact_cols': 1, 'categorical': 1},
        "Scatter Plot": {'exact_cols': 2, 'numerical': 2}, # Scatter needs strictly numerical
        "Line Chart": {'exact_cols': 2, 'numerical': (1, 2)}, # Needs at least 1 numerical (Y axis)
        "Box Plots (by Category)": {'exact_cols': 2, 'categorical': 1, 'distributable_numeric': 1}, # Use combined type for the value axis
        "Violin Plots (by Category)": {'exact_cols': 2, 'categorical': 1, 'distributable_numeric': 1}, # Use combined type for the value axis
        "Bar Chart (Aggregated)": {'exact_cols': 2, 'categorical': 1, 'numerical': 1}, # Aggregation needs strictly numerical
        "Grouped Bar Chart": {'exact_cols': 2, 'categorical': 2},
        "Heatmap (Counts)": {'exact_cols': 2, 'categorical': 2},
        "Area Chart": {'exact_cols': 2, 'numerical': (1, 2)}, # Y-axis must be numerical
        "Pair Plot": {'min_cols': 3, 'numerical': 3},
        "Correlation Heatmap": {'min_cols': 2, 'numerical': 2},
        "Parallel Coordinates Plot": {'min_cols': 3, 'numerical': 3}
    }

    if chart_type not in requirements:
         print(f"Warning: Validation rules not defined for chart type '{chart_type}'. Skipping validation.")
         return None

    req = requirements[chart_type]

    # --- Perform checks ---
    if 'exact_cols' in req and num_selected != req['exact_cols']: return f"needs exactly {req['exact_cols']} column(s), you chose {num_selected}"
    if 'min_cols' in req and num_selected < req['min_cols']: return f"needs at least {req['min_cols']} columns, you chose {num_selected}"

    err_msg = ""
    # Check counts using the appropriate counter variable based on requirements key
    if 'numerical' in req:
        if isinstance(req['numerical'], int) and num_numerical < req['numerical']: err_msg += f" Needs {req['numerical']} numerical (found {num_numerical})."
        elif isinstance(req['numerical'], tuple) and not (req['numerical'][0] <= num_numerical <= req['numerical'][1]): err_msg += f" Needs {req['numerical'][0]}-{req['numerical'][1]} numerical (found {num_numerical})."
    if 'categorical' in req:
        if isinstance(req['categorical'], int) and num_categorical < req['categorical']: err_msg += f" Needs {req['categorical']} categorical (found {num_categorical})."
    if 'distributable_numeric' in req:
         if isinstance(req['distributable_numeric'], int) and num_distributable_numeric < req['distributable_numeric']: err_msg += f" Needs {req['distributable_numeric']} numerical/rating-like (found {num_distributable_numeric})." # Use correct count
    if 'datetime' in req:
         if isinstance(req['datetime'], int) and num_datetime < req['datetime']: err_msg += f" Needs {req['datetime']} datetime (found {num_datetime})."

    # Specific combination checks (using refined counts)
    if chart_type in ["Line Chart", "Area Chart"] and num_numerical == 0: err_msg += " Needs at least one numerical column."
    if chart_type in ["Box Plots (by Category)", "Violin Plots (by Category)"] and (num_categorical == 0 or num_distributable_numeric == 0): err_msg += " Needs one categorical and one numerical/rating-like column."
    if chart_type == "Bar Chart (Aggregated)" and (num_categorical == 0 or num_numerical == 0): err_msg += " Needs one categorical and one numerical column."
    if chart_type in ["Grouped Bar Chart", "Heatmap (Counts)"] and num_categorical < 2: err_msg += " Needs two categorical columns."
    if chart_type == "Scatter Plot" and num_numerical < 2: err_msg += " Needs two numerical columns."

    return err_msg.strip() if err_msg else None # Return combined error message or None


# --- (generate_plot_and_get_uri function - keep as is from previous step) ---
def generate_plot_and_get_uri(filepath, chart_type, columns):
    """Generates plot and returns base64 URI or (None, error_msg)."""
    # ... (Keep the implementation including the call to validate_columns_for_chart) ...
    if not filepath: return None, "File path missing."
    try:
        try: df_full = pd.read_csv(filepath, usecols=columns) if filepath.endswith(".csv") else pd.read_excel(filepath, usecols=columns)
        except ValueError: df_full = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath) # Read full if usecols fails
    except Exception as e: return None, f"Error reading data file: {str(e)[:100]}"

    validation_error = validate_columns_for_chart(chart_type, columns, df_full)
    if validation_error: return None, validation_error

    if not all(col in df_full.columns for col in columns):
        missing_cols = [col for col in columns if col not in df_full.columns]
        return None, f"Column(s) not found: {', '.join(missing_cols)}"
    df_plot = df_full[columns].copy()

    img = io.BytesIO()
    plt.figure(figsize=(7, 4.5))
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        # --- Plotting logic ---
        if chart_type == "Histogram": sns.histplot(data=df_plot, x=columns[0], kde=True); plt.title(f"Histogram of {columns[0]}", fontsize=12)
        elif chart_type == "Box Plot": sns.boxplot(data=df_plot, y=columns[0]); plt.title(f"Box Plot of {columns[0]}", fontsize=12)
        elif chart_type == "Density Plot": sns.kdeplot(data=df_plot, x=columns[0], fill=True); plt.title(f"Density Plot of {columns[0]}", fontsize=12)
        elif chart_type == "Bar Chart (Counts)": counts = df_plot[columns[0]].value_counts().nlargest(15); sns.barplot(x=counts.index, y=counts.values); plt.title(f"Counts for {columns[0]}", fontsize=12); plt.ylabel("Count", fontsize=10); plt.xlabel(columns[0], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Pie Chart": counts = df_plot[columns[0]].value_counts(); effective_counts = counts.nlargest(6); plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85); plt.title(f"Pie Chart of {columns[0]}", fontsize=12); plt.axis('equal') # No donut for simplicity here
        elif chart_type == "Scatter Plot": sns.scatterplot(data=df_plot, x=columns[0], y=columns[1]); plt.title(f"Scatter: {columns[0]} vs {columns[1]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10)
        elif chart_type == "Line Chart":
            df_to_plot = df_plot.copy(); sort_col = columns[0]
            if pd.api.types.is_datetime64_any_dtype(df_to_plot[sort_col]): df_to_plot = df_to_plot.sort_values(by=sort_col)
            elif pd.api.types.is_numeric_dtype(df_to_plot[sort_col]): df_to_plot = df_to_plot.sort_values(by=sort_col)
            sns.lineplot(data=df_to_plot, x=columns[0], y=columns[1]); plt.title(f"Line: {columns[1]} over {columns[0]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Box Plots (by Category)": sns.boxplot(data=df_plot, x=columns[0], y=columns[1]); plt.title(f"Box Plots: {columns[1]} by {columns[0]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Violin Plots (by Category)": sns.violinplot(data=df_plot, x=columns[0], y=columns[1]); plt.title(f"Violin Plots: {columns[1]} by {columns[0]}", fontsize=12); plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Bar Chart (Aggregated)":
             col_types_specific = get_simplified_column_types(df_plot); cat_col = columns[0] if col_types_specific.get(columns[0]) in ['categorical', 'categorical_numeric'] else columns[1]; num_col = columns[1] if col_types_specific.get(columns[1]) == 'numerical' else columns[0]
             agg_data = df_plot.groupby(cat_col)[num_col].mean().nlargest(15); sns.barplot(x=agg_data.index, y=agg_data.values); plt.title(f"Mean of {num_col} by {cat_col}", fontsize=12); plt.xlabel(cat_col, fontsize=10); plt.ylabel(f"Mean of {num_col}", fontsize=10); plt.xticks(rotation=45, ha='right', fontsize=9)
        else: raise NotImplementedError(f"Plot type '{chart_type}' not implemented.") # Raise error if not implemented

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
        error_message = f"Failed to generate {chart_type}. ({str(e)[:100]}...)."
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

    # --- State Machine Logic (with validation integrated) ---
    if current_viz_state == 'asking_variable_types':
        session['user_answer_variable_types'] = user_input
        bot_reply = (f"Understood ({user_input}).<br><br>"
                     f"<strong>2. What message/insight for the visualization?</strong>"
                     f"<br>(e.g., compare values, show distribution, relationships, trends)")
        session['visualization_questions_state'] = 'asking_visualization_message'
        response_data = {"suggestions": ["Compare values/categories", "Show data distribution", "Identify relationships", "Track trends over time"]}

    elif current_viz_state == 'asking_visualization_message':
        session['user_answer_visualization_message'] = user_input
        columns_reminder = "<p>(Column list unavailable.)</p>"
        if df_columns: columns_reminder = f"Columns: {', '.join(df_columns[:5])}..." # Show first few
        bot_reply = (f"Goal: \"{user_input}\".<br><br>"
                     f"<strong>3. How many variables per chart?</strong>"
                     f"<br>(e.g., one, two, more)<br><br>{columns_reminder}")
        session['visualization_questions_state'] = 'asking_variable_count'
        response_data = {"suggestions": ["One variable", "Two variables", "More than two"]} # Simplified options

    elif current_viz_state == 'asking_variable_count':
        session['user_answer_variable_count'] = user_input # Store simplified count
        bot_reply = (f"Preferences Summary:<br>"
                     f"- Types: \"{user_answers['variable_types']}\"<br>"
                     f"- Goal: \"{user_answers['message_insight']}\"<br>"
                     f"- Vars: \"{user_input}\"<br><br>"
                     f"What next?")
        session['visualization_questions_state'] = 'visualization_info_gathered'
        response_data = {"suggestions": ["Suggest charts for me", "Let me choose columns", "Restart questions"]}

    elif current_viz_state == 'visualization_info_gathered':
        if "suggest chart" in user_input.lower():
            df_sample = None
            if uploaded_filepath:
                try: df_sample = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e: print(f"Error reading df_sample: {e}")
            chart_suggestions = suggest_charts_based_on_answers(user_answers, df_sample)
            session['chart_suggestions_list'] = chart_suggestions
            if chart_suggestions and chart_suggestions[0].get("type") != "Info":
                bot_reply = "Based on your info, consider these:<br>"; suggestions_for_user_options = []
                count = 0
                for chart_sugg in chart_suggestions:
                    if chart_sugg.get("type") == "Action": continue
                    if count >= 4 : break; count += 1
                    bot_reply += f"<br><strong>{count}. {chart_sugg['name']}</strong>"; sugg_suffix = ""
                    if chart_sugg.get("for_col"): sugg_suffix = f" for '{chart_sugg['for_col']}'"
                    elif chart_sugg.get("for_cols"): sugg_suffix = f" for '{chart_sugg['for_cols']}'"
                    bot_reply += f"{sugg_suffix}: {chart_sugg.get('reason', '')}"
                    suggestions_for_user_options.append(f"Select: {chart_sugg['name']}")
                bot_reply += "<br><br>Choose one, or pick columns manually."
                manual_pick_option = "Pick columns manually"
                if not any(s['name'] == manual_pick_option for s in chart_suggestions): chart_suggestions.append({"name": manual_pick_option, "type": "Action"}) # Ensure manual pick is possible
                response_data_suggestions = suggestions_for_user_options + [s['name'] for s in chart_suggestions if s.get("type") == "Action"]
                response_data_suggestions.append("Restart questions")
                response_data = {"suggestions": response_data_suggestions[:5]}
                session['visualization_questions_state'] = 'awaiting_chart_type_selection'
            else:
                bot_reply = "Couldn't find specific suggestions. Try picking columns."; response_data = {"suggestions": ["Let me pick columns", "Restart questions"]}; session['visualization_questions_state'] = 'awaiting_column_selection_general'
        elif "choose columns" in user_input.lower() or "pick columns" in user_input.lower():
             if not df_columns: bot_reply = "Need columns list. Upload data again."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
             else: bot_reply = f"Sure! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
        elif "restart" in user_input.lower(): home(); bot_reply = "Let's start over.<br><br><strong>1. Variable types?</strong>"; session['visualization_questions_state'] = 'asking_variable_types'; response_data = {"suggestions": ["Categorical", "Numerical", "Time-series", "Mix", "Any"]}
        else: bot_reply = "What next?"; response_data = {"suggestions": session.get('last_suggestions', ["Suggest charts for me", "Let me choose columns", "Restart questions"])}

    elif current_viz_state == 'awaiting_chart_type_selection':
        user_choice_str = user_input.replace("Select: ", "").strip()
        chart_suggestions_list = session.get('chart_suggestions_list', [])
        selected_chart_info = next((sugg for sugg in chart_suggestions_list if sugg['name'] == user_choice_str), None)
        if user_choice_str == "Pick columns manually":
             if not df_columns: bot_reply = "Need columns list. Upload data again."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
             else: bot_reply = f"Okay! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
        elif selected_chart_info:
            session['selected_chart_for_plotting'] = selected_chart_info; chart_name = selected_chart_info['name']
            required_cols_specific = selected_chart_info.get('required_cols_specific')
            bot_reply = f"Okay: <strong>{chart_name}</strong>. "
            if required_cols_specific: # If suggestion pinpointed columns
                cols_to_use_str = ", ".join(required_cols_specific); validation_msg = None
                if uploaded_filepath:
                    try: df_val = pd.read_csv(uploaded_filepath, usecols=required_cols_specific, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=required_cols_specific, nrows=5); validation_msg = validate_columns_for_chart(chart_name, required_cols_specific, df_val)
                    except Exception as e: validation_msg = f"Couldn't validate ({e})."
                    print(f"DEBUG Pre-validation for {chart_name}, cols {required_cols_specific}: {validation_msg or 'Passed'}") # <<< DEBUG PRINT
                if validation_msg is None:
                    bot_reply += f"Suggest using: <strong>{cols_to_use_str}</strong>. Proceed?"; session['plotting_columns'] = required_cols_specific; response_data = {"suggestions": [f"Yes, plot {chart_name}", "Choose other columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
                else:
                     bot_reply += f"Suggested columns '{cols_to_use_str}' may not work ({validation_msg}).<br>Please select columns for {chart_name}. Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
            else:
                 bot_reply += f"Which columns for {chart_name}? Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        else: bot_reply = "Didn't recognize that chart. Please choose again."; response_data = {"suggestions": session.get('last_suggestions', [])}

    elif current_viz_state == 'confirm_plot_details':
        chart_to_plot_info = session.get('selected_chart_for_plotting')
        cols_for_plot = session.get('plotting_columns')
        if user_input.startswith("Yes, plot"):
            if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                chart_name = chart_to_plot_info['name']
                bot_reply = f"Generating <strong>{chart_name}</strong> for: {', '.join(cols_for_plot)}."
                plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_name, cols_for_plot)
                if plot_image_uri: response_data["plot_image"] = plot_image_uri
                else: bot_reply += f"<br><strong>Plot Error:</strong> {error_msg or 'Unknown.'}"
                session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])
            else: bot_reply = "Missing plot details/file path."; session['visualization_questions_state'] = 'visualization_info_gathered'; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
        elif "choose other columns" in user_input.lower() or "change columns" in user_input.lower():
             chart_name = chart_to_plot_info['name'] if chart_to_plot_info else 'chart'; bot_reply = f"Okay, for the <strong>{chart_name}</strong>, which columns? Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        else: # Back to chart list or other
            bot_reply = "Okay, going back."; session['visualization_questions_state'] = 'awaiting_chart_type_selection'; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}


    elif current_viz_state == 'awaiting_columns_for_selected_chart':
        potential_cols_str = user_input.replace("Use:", "").strip()
        user_selected_cols = [col.strip() for col in potential_cols_str.split(',') if col.strip() and col.strip() in df_columns]
        chart_to_plot_info = session.get('selected_chart_for_plotting')
        chart_name = chart_to_plot_info.get('name', 'chart') if chart_to_plot_info else 'chart'
        if user_selected_cols:
            validation_msg = None
            if uploaded_filepath:
                 try: df_val = pd.read_csv(uploaded_filepath, usecols=user_selected_cols, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=user_selected_cols, nrows=5); validation_msg = validate_columns_for_chart(chart_name, user_selected_cols, df_val)
                 except Exception as e: validation_msg = f"Could not validate ({e})."
                 print(f"DEBUG Post-validation for {chart_name}, cols {user_selected_cols}: {validation_msg or 'Passed'}") # <<< DEBUG PRINT
            if validation_msg is None:
                session['plotting_columns'] = user_selected_cols; bot_reply = f"Using: <strong>{', '.join(user_selected_cols)}</strong> for <strong>{chart_name}</strong>. Plot?"; response_data = {"suggestions": ["Yes, generate plot", "Change columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
            else: bot_reply = f"Columns <strong>{', '.join(user_selected_cols)}</strong> might not work for <strong>{chart_name}</strong>. ({validation_msg})<br>Select others from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        elif "list all columns" in user_input.lower(): bot_reply = f"Available columns: {', '.join(df_columns)}.<br>Which for the {chart_name}?"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        elif "back to chart list" in user_input.lower():
            session['visualization_questions_state'] = 'awaiting_chart_type_selection'; bot_reply = "Okay, which chart type?"; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}
        else: bot_reply = f"Invalid columns. For <strong>{chart_name}</strong>, choose from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}

    elif current_viz_state == 'awaiting_column_selection_general':
        if "finished selecting" in user_input.lower():
            selected_cols = session.get('manual_columns_selected', [])
            if selected_cols: bot_reply = f"Selected: <strong>{', '.join(selected_cols)}</strong>. What kind of chart?"; session['plotting_columns'] = selected_cols; session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'; response_data = {"suggestions": ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]}
            else: bot_reply = "No columns selected. List columns or cancel."; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Cancel selection"]}
        elif "cancel selection" in user_input.lower(): session.pop('manual_columns_selected', None); session['visualization_questions_state'] = 'visualization_info_gathered'; bot_reply = "Selection cancelled."; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
        else:
            potential_col = user_input.replace("Use:","").strip(); current_selection = session.get('manual_columns_selected', [])
            if potential_col in df_columns:
                if potential_col not in current_selection: current_selection.append(potential_col)
                session['manual_columns_selected'] = current_selection; bot_reply = f"Added '<strong>{potential_col}</strong>'. Selected: <strong>{', '.join(current_selection) if current_selection else 'None'}</strong>.<br>Add more, or 'Finished selecting'." ; remaining_cols_suggestions = [f"Use: {col}" for col in df_columns if col not in current_selection][:2]; response_data = {"suggestions": remaining_cols_suggestions + ["Finished selecting", "Cancel selection"]}
            else: bot_reply = f"'{potential_col}' not valid. Choose from: {', '.join(df_columns)}."; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}

    elif current_viz_state == 'awaiting_chart_type_for_manual_cols':
        chart_type_from_user = user_input.strip(); cols_for_plot = session.get('plotting_columns', [])
        if cols_for_plot and uploaded_filepath:
            bot_reply = f"Attempting <strong>{chart_type_from_user}</strong> with: {', '.join(cols_for_plot)}."
            plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_type_from_user, cols_for_plot)
            if plot_image_uri: response_data["plot_image"] = plot_image_uri
            else: bot_reply += f"<br><strong>Plot Error:</strong> {error_msg or 'Unknown.'}"
        else: bot_reply = "Missing column/file details."
        session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])

    else: # Fallback to NLU
        nlu_output = nlu_get_bot_response(user_input)
        if isinstance(nlu_output, dict): bot_reply = nlu_output.get("response", "Sorry..."); temp_suggestions = nlu_output.get("suggestions", [])
        else: bot_reply = nlu_output; temp_suggestions = []
        if not temp_suggestions: temp_suggestions = ["Help", "Upload Data", "What can you do?"]
        response_data = {"suggestions": temp_suggestions}

    # --- Final response packaging ---
    response_data["response"] = bot_reply
    session['last_suggestions'] = response_data.get("suggestions", [])
    return jsonify(response_data)

# --- Upload Route ---
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
            df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
            session['df_columns'] = list(df.columns)
            preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)
            total_rows, total_columns = len(df), len(df.columns)
            missing_values = df.isnull().sum().sum(); duplicate_rows = df.duplicated().sum()
            total_cells = total_rows * total_columns; missing_percent = (missing_values / total_cells) * 100 if total_cells else 0
            initial_bot_message = (f"✅ <strong>{filename}</strong> uploaded.<br><br>"
                                 f"🔍 Quality Check: {total_rows} rows, {total_columns} cols; "
                                 f"{missing_values} missing ({missing_percent:.1f}%); {duplicate_rows} duplicates.<br><br>"
                                 f"Let's choose a visualization. I need info:<br><br>"
                                 f"<strong>1. Variable types?</strong> (e.g., text, numbers, dates)")
            session['visualization_questions_state'] = 'asking_variable_types'
            current_suggestions = ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of these types", "Not sure / Any"]
            session['last_suggestions'] = current_suggestions
            return jsonify({"response": initial_bot_message, "preview": preview_html, "suggestions": current_suggestions})
        except Exception as e:
            home(); print(f"Error processing uploaded file {filename}: {e}")
            return jsonify({"response": f"Error processing '{filename}': {str(e)[:100]}..."}), 500
    else: return jsonify({"response": "Invalid file type. Use CSV or Excel."}), 400

# --- Main Execution ---
if __name__ == "__main__":
    sns.set_theme(style="whitegrid", palette="muted") # Apply seaborn style globally
    plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 90}) # Auto layout and DPI for web
    app.run(debug=True)