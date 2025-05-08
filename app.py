import base64
import io
import os
import re # Import regex for cleaning
from typing import List, Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from pandas.api.types import (is_datetime64_any_dtype, is_numeric_dtype,
                              is_string_dtype)
from werkzeug.utils import secure_filename

# Matplotlib and Seaborn setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import bot logic
from bot_logic import get_bot_response as nlu_get_bot_response

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_numeric_column(series):
    """Attempt to clean a pandas Series to be numeric (handles $, ,, % )."""
    if series is None: return None
    if is_numeric_dtype(series): return series # Already numeric
    if series.dtype == 'object':
        try:
            # Remove $, ,, leading/trailing whitespace, and %
            # Convert empty strings resulting from cleaning to NaN
            cleaned_series = series.astype(str).str.replace(r'[$,%]', '', regex=True).str.strip()
            cleaned_series = cleaned_series.replace('', pd.NA) # Replace empty strings with NaN before conversion
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            # Check if successful (at least one number resulted)
            if numeric_series.notna().any():
                print(f"Cleaned column '{series.name}' to numeric.")
                return numeric_series
        except Exception as e:
            print(f"Cleaning failed for column '{series.name}': {e}")
            pass
    return series # Return original if cleaning failed or wasn't applicable

def get_simplified_column_types(df):
    """Analyzes DataFrame columns (after potential cleaning) and returns simplified types."""
    simplified_types = {}
    if df is None or df.empty: return simplified_types
    print("DEBUG: Inferring types for columns:", df.columns.tolist()) # Debug
    for col in df.columns:
        original_dtype_str = str(df[col].dtype) # Store original for reference
        try:
            # --- Attempt Cleaning First ---
            # Only apply to object columns that don't look like dates based on name
            temp_series = df[col]
            if temp_series.dtype == 'object' and not any(substr in col.lower() for substr in ['date', 'time', 'yr', 'year']):
                 temp_series = clean_numeric_column(temp_series)
            # --- End Cleaning Attempt ---

            dtype = temp_series.dtype # Use dtype of potentially cleaned series
            unique_count = temp_series.nunique(dropna=True)
            non_null_count = temp_series.count()

            if non_null_count == 0: simplified_types[col] = 'empty'
            elif is_numeric_dtype(dtype):
                 is_integer = pd.api.types.is_integer_dtype(dtype)
                 # Looser heuristic for categorical numeric - primarily check unique count
                 if unique_count < 20 and is_integer : # Treat low-unique integers as categorical potential
                      simplified_types[col] = 'categorical_numeric'
                 else:
                      simplified_types[col] = 'numerical' # Includes floats and high-unique ints
            elif is_datetime64_any_dtype(dtype) or any(substr in col.lower() for substr in ['date', 'time', 'yr', 'year']):
                 # Also try converting object columns named like dates
                 if not is_datetime64_any_dtype(dtype):
                      try:
                          pd.to_datetime(temp_series, errors='raise') # Test conversion
                          simplified_types[col] = 'datetime'
                      except: # If conversion fails, treat as categorical/other
                           simplified_types[col] = 'categorical' if unique_count < 150 else 'id_like_text'
                 else:
                      simplified_types[col] = 'datetime'
            elif is_string_dtype(dtype) or dtype == 'object':
                 if unique_count <= 1: simplified_types[col] = 'categorical'
                 # Use a higher threshold for uniqueness before calling it ID-like
                 elif unique_count < max(2, non_null_count * 0.7) and unique_count < 250:
                     simplified_types[col] = 'categorical'
                 else: simplified_types[col] = 'id_like_text'
            else: simplified_types[col] = 'other'
        except Exception as e: print(f"Warning: Type check failed for '{col}' (Original dtype: {original_dtype_str}): {e}"); simplified_types[col] = 'other'
    print(f"DEBUG Simplified Types: {simplified_types}") # Keep this debug print
    return simplified_types

# --- Validation Function (Returns User-Friendly Error String) ---
def validate_columns_for_chart(chart_type: str, columns: List[str], df: pd.DataFrame) -> Optional[str]:
    """Validates columns for chart type. Returns USER-FRIENDLY error message or None."""
    if not columns: return "No columns selected."
    missing = [col for col in columns if col not in df.columns]
    if missing: return f"Column(s) not found: {', '.join(missing)}. Check spelling?"

    df_subset = df[columns].copy()
    # Clean again *before* validation for consistency
    for col in df_subset.columns: df_subset[col] = clean_numeric_column(df_subset[col])
    col_types = get_simplified_column_types(df_subset)

    num_numerical = sum(1 for t in col_types.values() if t == 'numerical')
    num_categorical = sum(1 for t in col_types.values() if t in ['categorical', 'categorical_numeric'])
    num_distributable = sum(1 for t in col_types.values() if t in ['numerical', 'categorical_numeric'])
    num_datetime = sum(1 for t in col_types.values() if t == 'datetime')
    num_id_like = sum(1 for t in col_types.values() if t == 'id_like_text')
    num_selected = len(columns)
    col_details = ", ".join([f"'{c}' ({col_types.get(c, '?')})" for c in columns]) # Show inferred type

    error = None
    # --- More User-Friendly Error Messages ---
    if chart_type == "Histogram":
        if num_selected != 1: error = f"needs 1 column, but you selected {num_selected}."
        elif num_distributable < 1: error = f"needs a numerical column (like Amount, Boxes Shipped, Rating) to show its distribution. ({col_details} was selected)."
    elif chart_type in ["Box Plot", "Density Plot"]:
        if num_selected != 1: error = f"needs 1 column, but you selected {num_selected}."
        elif num_distributable < 1: error = f"needs a numerical column (like Amount, Boxes Shipped, Rating). ({col_details} was selected)."
    elif chart_type == "Bar Chart (Counts)":
        if num_selected != 1: error = f"needs 1 column, but you selected {num_selected}."
        elif num_categorical < 1: error = f"needs a categorical column with distinct groups (like Country, Product). The selected column ({col_details}) doesn't seem suitable, possibly because it has too many unique text values (like Sales Person)."
    elif chart_type == "Pie Chart":
        if num_selected != 1: error = f"needs 1 column, but you selected {num_selected}."
        elif num_categorical < 1: error = f"needs a categorical column with distinct groups (like Country, Product). ({col_details} was selected)."
        else: # Check slice count
            try: nunique = df_subset[columns[0]].nunique(dropna=True); if nunique > 10: error = f"'{columns[0]}' has {nunique} categories, which is too many for a clear Pie Chart. Try a Bar Chart instead."
            except: pass # Ignore potential errors here, focus on type first
    elif chart_type == "Scatter Plot":
        if num_selected != 2: error = f"needs exactly 2 columns, but you selected {num_selected}."
        elif num_numerical < 2: error = f"needs 2 numerical columns (e.g., Amount vs Boxes Shipped) to show their relationship, but found {num_numerical} numerical in ({col_details})."
    elif chart_type in ["Line Chart", "Area Chart"]:
         if num_selected != 2: error = f"needs exactly 2 columns, but you selected {num_selected}."
         elif num_numerical == 0 : error = f"needs at least one numerical column (e.g., Amount) for the values over time or sequence. ({col_details} was selected)."
         # Could add check for datetime if required
    elif chart_type in ["Box Plots (by Category)", "Violin Plots (by Category)"]:
        if num_selected != 2: error = f"needs exactly 2 columns, but you selected {num_selected}."
        elif num_categorical < 1 or num_distributable < 1: error = f"needs one categorical column (e.g., Country) and one numerical column (e.g., Amount). You selected ({col_details})."
    elif chart_type == "Bar Chart (Aggregated)":
        if num_selected != 2: error = f"needs exactly 2 columns, but you selected {num_selected}."
        elif num_categorical < 1 or num_numerical < 1: error = f"needs one categorical column (e.g., Country) and one strictly numerical column (e.g., Amount) for averaging. You selected ({col_details})."
    elif chart_type in ["Grouped Bar Chart", "Heatmap (Counts)"]:
        if num_selected != 2: error = f"needs exactly 2 columns, but you selected {num_selected}."
        elif num_categorical < 2: error = f"needs two categorical columns (e.g., Country and Product). You selected ({col_details})."
    # Add validation for multivariate charts if implemented

    if error: return f"{chart_type} " + error # Prepend chart type to error
    return None # Passed

# --- Plotting Function (Includes Cleaning and Validation) ---
def generate_plot_and_get_uri(filepath, chart_type, columns):
    """Generates plot and returns base64 URI or (None, error_msg)."""
    if not filepath: return None, "File path missing."
    try:
        # Read full data first
        df_full = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
        # Check if columns exist
        if not all(col in df_full.columns for col in columns):
            missing_cols = [c for c in columns if c not in df_full.columns]
            return None, f"Column(s) not found: {', '.join(missing_cols)}."

        # --- Data Cleaning Step ---
        df_plot = df_full[columns].copy() # Subset BEFORE cleaning
        print(f"DEBUG: Columns before cleaning for {chart_type}: {df_plot.dtypes}")
        for col in plot_columns:
            # Apply cleaning, especially to potential numeric/datetime columns
            # Attempt date conversion more broadly based on name
            if any(substr in col.lower() for substr in ['date', 'time', 'yr', 'year']) and not is_datetime64_any_dtype(df_plot[col]):
                try: df_plot[col] = pd.to_datetime(df_plot[col]) ; print(f"Cleaned '{col}' to datetime.")
                except Exception as e: print(f"Warn: Failed datetime conversion for '{col}': {e}")
            # Apply numeric cleaning
            df_plot[col] = clean_numeric_column(df_plot[col])
        print(f"DEBUG: Columns after cleaning for {chart_type}: {df_plot.dtypes}")
        # --- End Cleaning ---

        # *** Run Validation ON CLEANED DATA ***
        validation_error = validate_columns_for_chart(chart_type, columns, df_plot)
        if validation_error:
            # Return the user-friendly validation error
            return None, f"Invalid columns for {chart_type}: {validation_error}"

    except Exception as e:
        print(f"Error reading/cleaning/validating dataframe ('{filepath}'): {e}")
        return None, f"Error preparing data: {str(e)[:100]}"

    # --- Proceed with Plotting if Validation Passed ---
    img = io.BytesIO(); plt.figure(figsize=(7.5, 5)); plt.style.use('seaborn-v0_8-whitegrid')
    original_chart_type = chart_type; plot_title_detail = ""
    mapped_chart_type = chart_type; plot_columns = list(columns);

    try:
        print(f"Attempting plot generation: {original_chart_type} with {plot_columns}")
        col_types_specific = get_simplified_column_types(df_plot) # Use types from cleaned df_plot

        # --- Handle Generic "Bar Chart" Request ---
        if original_chart_type == "Bar Chart": # Map based on original request
             if len(plot_columns)==1 and plot_columns[0] in col_types_specific and col_types_specific[plot_columns[0]] in ['categorical','categorical_numeric']: mapped_chart_type="Bar Chart (Counts)"; plot_title_detail=f" for {plot_columns[0]}"
             elif len(plot_columns)==2:
                  cat_cols=[c for c in plot_columns if col_types_specific.get(c) in ['categorical','categorical_numeric']]; num_cols=[c for c in plot_columns if col_types_specific.get(c)=='numerical']
                  if len(cat_cols)==1 and len(num_cols)==1: mapped_chart_type="Bar Chart (Aggregated)"; plot_columns=[cat_cols[0],num_cols[0]]; plot_title_detail=f" of {num_cols[0]} by {cat_cols[0]}"
                  elif len(cat_cols)==2: mapped_chart_type="Grouped Bar Chart"; plot_title_detail=f" for {plot_columns[0]} by {plot_columns[1]}"
                  else: raise ValueError(f"Cannot determine Bar Chart type for ({', '.join(plot_columns)}).")
             else: raise ValueError("Bar Chart needs 1 or 2 columns.")
             print(f"--> Handling '{original_chart_type}' as '{mapped_chart_type}' with columns {plot_columns}")
        # --- End Handling Generic "Bar Chart" ---

        plot_title=f"{mapped_chart_type}{plot_title_detail}"

        # --- Plotting Logic (uses mapped_chart_type, plot_columns, and cleaned df_plot) ---
        if mapped_chart_type=="Histogram": sns.histplot(data=df_plot, x=plot_columns[0], kde=True); plot_title=f"Histogram of {plot_columns[0]}"
        elif mapped_chart_type=="Box Plot": sns.boxplot(data=df_plot, y=plot_columns[0]); plot_title=f"Box Plot of {plot_columns[0]}"
        elif mapped_chart_type=="Density Plot": sns.kdeplot(data=df_plot, x=plot_columns[0], fill=True); plot_title=f"Density Plot of {plot_columns[0]}"
        elif mapped_chart_type=="Bar Chart (Counts)": counts=df_plot[plot_columns[0]].value_counts().nlargest(20); sns.barplot(x=counts.index.astype(str), y=counts.values); plot_title=f"Top Counts for {plot_columns[0]}"; plt.ylabel("Count"); plt.xlabel(plot_columns[0]); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type == "Pie Chart": counts = df_plot[plot_columns[0]].value_counts(); effective_counts = counts.nlargest(7); if len(counts) > 7: effective_counts.loc['Other'] = counts.iloc[7:].sum(); plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85); plot_title = f"Pie Chart of {plot_columns[0]}"; plt.axis('equal')
        elif mapped_chart_type=="Scatter Plot": sns.scatterplot(data=df_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Scatter: {plot_columns[0]} vs {plot_columns[1]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1])
        elif mapped_chart_type=="Line Chart":
            df_to_plot=df_plot.copy(); sort_col=plot_columns[0]
            try: # Attempt sorting for line chart
                if pd.api.types.is_datetime64_any_dtype(df_to_plot[sort_col]): df_to_plot=df_to_plot.sort_values(by=sort_col)
                elif pd.api.types.is_numeric_dtype(df_to_plot[sort_col]): df_to_plot=df_to_plot.sort_values(by=sort_col)
            except Exception as sort_e: print(f"Note: Could not sort for Line Chart: {sort_e}")
            sns.lineplot(data=df_to_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Line: {plot_columns[1]} over {plot_columns[0]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1]); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif mapped_chart_type=="Box Plots (by Category)": sns.boxplot(data=df_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Box Plots: {plot_columns[1]} by {plot_columns[0]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1]); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Violin Plots (by Category)": sns.violinplot(data=df_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Violin Plots: {plot_columns[1]} by {plot_columns[0]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1]); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Bar Chart (Aggregated)": cat_col,num_col=plot_columns[0],plot_columns[1]; agg_data=df_plot.groupby(cat_col)[num_col].mean().nlargest(20); sns.barplot(x=agg_data.index.astype(str), y=agg_data.values); plot_title=f"Mean of {num_col} by {cat_col}"; plt.xlabel(cat_col); plt.ylabel(f"Mean of {num_col}"); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Grouped Bar Chart": col1_tc=df_plot[plot_columns[0]].value_counts().nlargest(10).index; col2_tc=df_plot[plot_columns[1]].value_counts().nlargest(5).index; df_f=df_plot[df_plot[plot_columns[0]].isin(col1_tc) & df_plot[plot_columns[1]].isin(col2_tc)]; sns.countplot(data=df_f, x=plot_columns[0], hue=plot_columns[1]); plot_title=f"Counts: {plot_columns[0]} by {plot_columns[1]}"; plt.xlabel(plot_columns[0]); plt.ylabel("Count"); plt.xticks(rotation=65, ha='right', fontsize=9); plt.legend(title=plot_columns[1], fontsize='x-small', title_fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')
        else: raise NotImplementedError(f"Plot type '{mapped_chart_type}' is not explicitly implemented.")

        plt.title(plot_title, fontsize=12); plt.tight_layout(pad=1.0); plt.savefig(img, format='png', bbox_inches='tight'); plt.close(); img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8'); print(f"Success: {original_chart_type} (as {mapped_chart_type})"); return f"data:image/png;base64,{plot_url}", None
    except Exception as e:
        error_info = f"{type(e).__name__}: {str(e)}"
        print(f"!!! Error during plot generation execution for '{mapped_chart_type}' with {plot_columns}: {error_info}")
        error_message = f"Failed to generate {original_chart_type}. ({error_info[:100]}...). Check if data types match the chart."
        if 'plt' in locals() and plt.get_fignums(): plt.close('all')
        return None, error_message

# --- Flask Routes ---

@app.route("/")
def home():
    """Clears session and renders the main page."""
    keys_to_clear = ['visualization_questions_state', 'uploaded_filepath', 'uploaded_filename', 'df_columns', 'user_answer_variable_types', 'user_answer_visualization_message', 'user_answer_variable_count', 'chart_suggestions_list', 'selected_chart_for_plotting', 'plotting_columns', 'manual_columns_selected', 'last_suggestions']
    for key in keys_to_clear: session.pop(key, None)
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    """Handles user messages, state transitions, suggestions, and plotting calls."""
    user_input = request.json.get("message")
    response_data = {}; bot_reply = ""
    user_input_lower = user_input.lower() if user_input else ""

    # --- Explicit command checks BEFORE state machine ---
    if "restart questions" in user_input_lower or "restart" == user_input_lower:
        home(); bot_reply = "Okay, restarting questions.<br><br><strong>1. Variable types?</strong>"; session['visualization_questions_state'] = 'asking_variable_types'; response_data = {"suggestions": ["Categorical", "Numerical", "Time-series", "Mix", "Any"]}
        response_data["response"] = bot_reply; session['last_suggestions'] = response_data.get("suggestions", []); return jsonify(response_data)

    if "suggest another chart" in user_input_lower:
        if session.get('uploaded_filepath') and session.get('user_answer_variable_count') is not None:
            session['visualization_questions_state'] = 'visualization_info_gathered'; bot_reply = "Okay, let's look for other visualizations. How proceed?"
            response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns", "Restart questions"]}
            response_data["response"] = bot_reply; session['last_suggestions'] = response_data.get("suggestions", []); return jsonify(response_data)
        else: bot_reply = "Let's start over. Please upload data."; home(); response_data = {"suggestions": ["Upload Data", "Help"]}
        # Fall through to normal packaging if context lost

    # --- State Machine Logic ---
    current_viz_state = session.get('visualization_questions_state')
    df_columns = session.get('df_columns', [])
    uploaded_filepath = session.get('uploaded_filepath')
    user_answers = {'variable_types': session.get('user_answer_variable_types', ''), 'message_insight': session.get('user_answer_visualization_message', ''), 'variable_count': session.get('user_answer_variable_count', '')}

    # --- State Handling with Improved Validation Feedback ---
    if current_viz_state == 'asking_variable_types':
        session['user_answer_variable_types'] = user_input
        bot_reply = (f"Understood ({user_input}).<br><br>"
                     f"<strong>2. What's the main message/insight?</strong>"
                     f"<br>(e.g., compare, distribution, relationship, trend)")
        session['visualization_questions_state'] = 'asking_visualization_message'
        response_data = {"suggestions": ["Compare values/categories", "Show data distribution", "Identify relationships", "Track trends over time"]}

    elif current_viz_state == 'asking_visualization_message':
        session['user_answer_visualization_message'] = user_input
        columns_reminder = ""
        if df_columns: columns_reminder = f"(Cols: {', '.join(df_columns[:3])}...)"
        bot_reply = (f"Goal: \"{user_input}\".<br><br>"
                     f"<strong>3. How many variables per chart?</strong> {columns_reminder}"
                     f"<br>(e.g., one, two, more)")
        session['visualization_questions_state'] = 'asking_variable_count'
        response_data = {"suggestions": ["One variable", "Two variables", "More than two"]}

    elif current_viz_state == 'asking_variable_count':
        session['user_answer_variable_count'] = user_input
        bot_reply = (f"Preferences:<br>"
                     f"- Types: \"{user_answers['variable_types']}\"<br>"
                     f"- Goal: \"{user_answers['message_insight']}\"<br>"
                     f"- Vars: \"{user_input}\"<br><br>"
                     f"What next?")
        session['visualization_questions_state'] = 'visualization_info_gathered'
        response_data = {"suggestions": ["Suggest charts for me", "Let me choose columns", "Restart questions"]}

    elif current_viz_state == 'visualization_info_gathered':
        if "suggest chart" in user_input_lower:
            df_sample = None; bot_reply = ""; response_data_suggestions = []
            if uploaded_filepath:
                try: df_sample = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e: print(f"Error reading df_sample: {e}")
            chart_suggestions = suggest_charts_based_on_answers(user_answers, df_sample); session['chart_suggestions_list'] = chart_suggestions
            if chart_suggestions and chart_suggestions[0].get("type") not in ["Info", None]:
                 bot_reply = "Based on your info, consider:<br>"; suggestions_for_user_options = []
                 count = 0
                 for chart_sugg in chart_suggestions:
                     if chart_sugg.get("type") == "Action": continue
                     if count >= 4 : break; count += 1
                     bot_reply += f"<br><strong>{count}. {chart_sugg['name']}</strong>"; sugg_suffix = ""
                     if chart_sugg.get("for_col"): sugg_suffix = f" for '{chart_sugg['for_col']}'"
                     elif chart_sugg.get("for_cols"): sugg_suffix = f" (e.g., for '{chart_sugg['for_cols']}')"
                     bot_reply += f"{sugg_suffix}: {chart_sugg.get('reason', '')}"
                     suggestions_for_user_options.append(f"Select: {chart_sugg['name']}")
                 bot_reply += "<br><br>Choose one, or pick columns?"
                 manual_pick_option = "Pick columns manually"; if not any(s['name']==manual_pick_option for s in chart_suggestions): chart_suggestions.append({"name": manual_pick_option, "type": "Action"})
                 response_data_suggestions = suggestions_for_user_options + [s['name'] for s in chart_suggestions if s.get("type") == "Action"]
                 if "Restart questions" not in response_data_suggestions: response_data_suggestions.append("Restart questions")
                 response_data = {"suggestions": response_data_suggestions[:5]}; session['visualization_questions_state'] = 'awaiting_chart_type_selection'
            else: bot_reply = "Couldn't find specific suggestions based on your input and data. You could try picking columns manually."; response_data = {"suggestions": ["Let me pick columns", "Restart questions"]}; session['visualization_questions_state'] = 'awaiting_column_selection_general'
        elif "choose columns" in user_input_lower or "pick columns" in user_input_lower:
             if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
             else: bot_reply = f"Sure! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
        else: bot_reply = "What next?"; response_data = {"suggestions": session.get('last_suggestions', ["Suggest charts", "Pick columns", "Restart"])}

    elif current_viz_state == 'awaiting_chart_type_selection':
        user_choice_str = user_input.replace("Select: ", "").strip(); chart_suggestions_list = session.get('chart_suggestions_list', []); selected_chart_info = next((sugg for sugg in chart_suggestions_list if sugg['name'] == user_choice_str), None)
        if user_choice_str == "Pick columns manually":
             if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
             else: bot_reply = f"Okay! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
        elif selected_chart_info:
            session['selected_chart_for_plotting'] = selected_chart_info; chart_name = selected_chart_info['name']; required_cols_specific = selected_chart_info.get('required_cols_specific')
            bot_reply = f"Okay: <strong>{chart_name}</strong>. "
            if required_cols_specific:
                cols_to_use_str = ", ".join(required_cols_specific); validation_msg = None
                if uploaded_filepath:
                    try: # Perform Pre-validation
                        df_val = pd.read_csv(uploaded_filepath, usecols=required_cols_specific, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=required_cols_specific, nrows=5)
                        # Clean subset before validation
                        for col in df_val.columns: df_val[col] = clean_numeric_column(df_val[col])
                        validation_msg = validate_columns_for_chart(chart_name, required_cols_specific, df_val)
                        print(f"DEBUG Pre-validation for {chart_name}, cols {required_cols_specific}: {validation_msg or 'Passed'}")
                    except Exception as e: validation_msg = f"Couldn't pre-validate ({str(e)[:50]}...)."; print(f"DEBUG Validation Read Error: {e}")
                if validation_msg is None: # If valid, confirm with user
                    bot_reply += f"Using suggested columns: <strong>{cols_to_use_str}</strong>. Plot?"; session['plotting_columns'] = required_cols_specific; response_data = {"suggestions": [f"Yes, plot {chart_name}", "Choose other columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
                else: # If suggested cols are invalid (e.g., due to actual data types), prompt user
                     bot_reply += f"The suggested columns '{cols_to_use_str}' might not be suitable for a {chart_name}. Reason: {validation_msg}<br>Please select appropriate columns below. Available: {', '.join(df_columns)}"
                     session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
            else: # If no specific columns were suggested by the logic, ask user
                 bot_reply += f"Which columns for the {chart_name}? Available: {', '.join(df_columns)}"
                 # Add hints based on chart type
                 if chart_name == "Histogram": bot_reply += "<br><i>Hint: Choose one numerical column.</i>"
                 elif chart_name == "Scatter Plot": bot_reply += "<br><i>Hint: Choose two numerical columns.</i>"
                 elif chart_name == "Box Plots (by Category)": bot_reply += "<br><i>Hint: Choose one categorical and one numerical column.</i>"
                 # Add more hints
                 session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        else: bot_reply = "Didn't recognize that chart. Choose again."; response_data = {"suggestions": session.get('last_suggestions', [])}

    elif current_viz_state == 'confirm_plot_details':
        chart_to_plot_info = session.get('selected_chart_for_plotting'); cols_for_plot = session.get('plotting_columns')
        if user_input.startswith("Yes, plot"):
            if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                chart_name = chart_to_plot_info['name']; bot_reply = f"Generating <strong>{chart_name}</strong>..."
                plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_name, cols_for_plot)
                if plot_image_uri: response_data["plot_image"] = plot_image_uri; bot_reply = f"Here's the <strong>{chart_name}</strong> for: {', '.join(cols_for_plot)}." # Shorter success message
                else: bot_reply = f"Sorry, couldn't generate the <strong>{chart_name}</strong>.<br><strong>Reason:</strong> {error_msg or 'Unknown error.'}<br>Try suggesting another chart or picking different columns." # Show error
                session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])
            else: bot_reply = "Missing details/file path."; session['visualization_questions_state'] = 'visualization_info_gathered'; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
        elif "choose other columns" in user_input_lower or "change columns" in user_input_lower:
             chart_name = chart_to_plot_info['name'] if chart_to_plot_info else 'chart'; bot_reply = f"Okay, for <strong>{chart_name}</strong>, which columns? Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        else: # Back to chart list
            bot_reply = "Okay, back to chart list."; session['visualization_questions_state'] = 'awaiting_chart_type_selection'; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}

    elif current_viz_state == 'awaiting_columns_for_selected_chart':
        potential_cols_str = user_input.replace("Use:", "").strip(); user_selected_cols = [col.strip() for col in potential_cols_str.split(',') if col.strip() and col.strip() in df_columns]
        chart_to_plot_info = session.get('selected_chart_for_plotting'); chart_name = chart_to_plot_info.get('name', 'chart') if chart_to_plot_info else 'chart'
        if user_selected_cols:
            validation_msg = None
            if uploaded_filepath: # Perform validation on user selected columns
                 try: df_val = pd.read_csv(uploaded_filepath, usecols=user_selected_cols, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=user_selected_cols, nrows=5);
                      # Clean before validation
                      for col in df_val.columns: df_val[col] = clean_numeric_column(df_val[col])
                      validation_msg = validate_columns_for_chart(chart_name, user_selected_cols, df_val)
                 except Exception as e: validation_msg = f"Couldn't validate ({str(e)[:50]}...)."; print(f"DEBUG Post-Validation Read Error: {e}")
                 print(f"DEBUG Post-validation for {chart_name}, cols {user_selected_cols}: {validation_msg or 'Passed'}")
            if validation_msg is None: # If valid, confirm
                session['plotting_columns'] = user_selected_cols; bot_reply = f"Using: <strong>{', '.join(user_selected_cols)}</strong> for <strong>{chart_name}</strong>. Plot?"; response_data = {"suggestions": ["Yes, generate plot", "Change columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
            else: # If invalid, explain and re-prompt
                 bot_reply = f"The columns <strong>{', '.join(user_selected_cols)}</strong> might not work for <strong>{chart_name}</strong>. <br><strong>Reason:</strong> {validation_msg}<br>Please select others from: {', '.join(df_columns)}"
                 response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        elif "list all columns" in user_input_lower: bot_reply = f"Available: {', '.join(df_columns)}.<br>Which for {chart_name}?"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        elif "back to chart list" in user_input_lower:
            session['visualization_questions_state'] = 'awaiting_chart_type_selection'; bot_reply = "Okay, which chart type?"; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}
        else: bot_reply = f"Invalid columns for <strong>{chart_name}</strong>. Choose from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}

    elif current_viz_state == 'awaiting_column_selection_general':
        # Restart handled globally
        if "finished selecting" in user_input_lower:
            selected_cols = session.get('manual_columns_selected', [])
            if selected_cols: bot_reply = f"Selected: <strong>{', '.join(selected_cols)}</strong>. What kind of chart?"; session['plotting_columns'] = selected_cols; session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'; response_data = {"suggestions": ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]}
            else: bot_reply = "No columns selected. List columns or cancel."; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Cancel selection"]}
        elif "cancel selection" in user_input_lower: session.pop('manual_columns_selected', None); session['visualization_questions_state'] = 'visualization_info_gathered'; bot_reply = "Selection cancelled."; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
        else:
            potential_col = user_input.replace("Use:","").strip(); current_selection = session.get('manual_columns_selected', [])
            if potential_col in df_columns:
                if potential_col not in current_selection: current_selection.append(potential_col)
                session['manual_columns_selected'] = current_selection; bot_reply = f"Added '<strong>{potential_col}</strong>'. Selected: <strong>{', '.join(current_selection) if current_selection else 'None'}</strong>.<br>Add more, or 'Finished selecting'." ; remaining_cols_suggestions = [f"Use: {col}" for col in df_columns if col not in current_selection][:2]; response_data = {"suggestions": remaining_cols_suggestions + ["Finished selecting", "Cancel selection"]}
            else: bot_reply = f"'{potential_col}' not valid. Choose from: {', '.join(df_columns)}."; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}

    elif current_viz_state == 'awaiting_chart_type_for_manual_cols':
        # Restart handled globally
        chart_type_from_user = user_input.strip(); cols_for_plot = session.get('plotting_columns', [])
        if cols_for_plot and uploaded_filepath:
            bot_reply = f"Attempting <strong>{chart_type_from_user}</strong> with: {', '.join(cols_for_plot)}."
            # Call plot generation with validation included
            plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_type_from_user, cols_for_plot)
            if plot_image_uri: response_data["plot_image"] = plot_image_uri
            else: bot_reply += f"<br><strong>Plot Error:</strong> {error_msg or 'Unknown.'}"
        else: bot_reply = "Missing column/file details."
        session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])

    else: # Fallback to NLU / No specific state
        if not bot_reply: # Only call NLU if no other state handled it and no global command returned early
            nlu_output = nlu_get_bot_response(user_input);
            if isinstance(nlu_output, dict): bot_reply = nlu_output.get("response", "Sorry..."); temp_suggestions = nlu_output.get("suggestions", [])
            else: bot_reply = nlu_output; temp_suggestions = []
            if not temp_suggestions: temp_suggestions = ["Help", "Upload Data", "What can you do?"]
            response_data = {"suggestions": temp_suggestions}
        elif 'suggestions' not in response_data: response_data = {"suggestions": ["Help", "Upload Data"]}


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
        filename = secure_filename(file.filename); home(); filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath); session['uploaded_filepath'] = filepath; session['uploaded_filename'] = filename
            read_engine = 'openpyxl' if filename.endswith(('.xlsx', '.xls')) else None
            # Attempt to read, handle potential parsing errors early
            try:
                df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath, engine=read_engine)
            except Exception as read_err:
                 print(f"Initial read error for {filename}: {read_err}")
                 # Try with a different approach for CSV if it failed
                 if filename.endswith(".csv"):
                      try: df = pd.read_csv(filepath, encoding='latin1') # Try different encoding
                      except Exception as read_err_2: raise read_err_2 # Re-raise if still fails
                 else: raise read_err # Re-raise non-csv error

            session['df_columns'] = list(df.columns);
            preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)
            total_rows,total_columns=len(df),len(df.columns); missing_values=df.isnull().sum().sum(); duplicate_rows=df.duplicated().sum(); total_cells=total_rows*total_columns; missing_percent=(missing_values/total_cells)*100 if total_cells else 0
            initial_bot_message = (f"✅ <strong>{filename}</strong> uploaded.<br><br>"
                                 f"🔍 Quality Check: {total_rows} R, {total_columns} C; {missing_values} missing ({missing_percent:.1f}%); {duplicate_rows} duplicates.<br><br>"
                                 f"Let's choose a visualization. I need info:<br><br>"
                                 f"<strong>1. Variable types?</strong> (e.g., text, numbers, dates)")
            session['visualization_questions_state'] = 'asking_variable_types'
            current_suggestions = ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of these types", "Not sure / Any"]
            session['last_suggestions'] = current_suggestions
            return jsonify({"response": initial_bot_message, "preview": preview_html, "suggestions": current_suggestions})
        except Exception as e: home(); print(f"Error processing uploaded file {filename}: {e}"); return jsonify({"response": f"Error processing '{filename}': {str(e)[:100]}..."}), 500
    else: return jsonify({"response": "Invalid file type. Use CSV or Excel."}), 400

# --- Main Execution ---
if __name__ == "__main__":
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 90})
    app.run(debug=True) # Set debug=False for production