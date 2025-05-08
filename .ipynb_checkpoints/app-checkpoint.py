import base64
import io
import os
from typing import List, Optional

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
    if df is None or df.empty: return simplified_types
    for col in df.columns:
        try:
            dtype = df[col].dtype; unique_count = df[col].nunique(dropna=True); non_null_count = df[col].count()
            if non_null_count == 0: simplified_types[col] = 'empty'; continue
            if is_numeric_dtype(dtype): simplified_types[col] = 'categorical_numeric' if unique_count < 15 and (unique_count < non_null_count * 0.1 or unique_count < 7) else 'numerical'
            elif is_datetime64_any_dtype(dtype): simplified_types[col] = 'datetime'
            elif is_string_dtype(dtype) or dtype == 'object': simplified_types[col] = 'categorical' if unique_count <= 1 or (unique_count < non_null_count * 0.8 and unique_count < 150) else 'id_like_text'
            else: simplified_types[col] = 'other'
        except Exception as e: print(f"Warning: Type check failed for '{col}': {e}"); simplified_types[col] = 'other'
    return simplified_types

def suggest_charts_based_on_answers(user_answers, df_sample):
    """Suggests chart types based on user preferences and data sample."""
    suggestions = []
    if df_sample is None or df_sample.empty: return [{"name": "Cannot suggest: Data sample missing.", "type": "Info"}]
    col_types = get_simplified_column_types(df_sample)
    numerical_cols = [c for c,t in col_types.items() if t=='numerical']; categorical_cols = [c for c,t in col_types.items() if t in ['categorical', 'categorical_numeric']]; distributable_numeric_cols = [c for c,t in col_types.items() if t in ['numerical', 'categorical_numeric']]; datetime_cols = [c for c,t in col_types.items() if t=='datetime']
    ua_count, ua_types, ua_msg = user_answers.get('variable_count','').lower(), user_answers.get('variable_types','').lower(), user_answers.get('message_insight','').lower()

    # --- Suggestion Rules ---
    if ("one" in ua_count or "1" in ua_count) and ("dist" in ua_msg or "spread" in ua_msg or "summ" in ua_msg) and ("num" in ua_types or "cat" in ua_types or "any" in ua_types or not ua_types) and distributable_numeric_cols:
        for col in distributable_numeric_cols: suggestions.extend([{"name": "Histogram", "for_col": col, "reason": f"Distribution of '{col}'.", "required_cols_specific": [col]}, {"name": "Box Plot", "for_col": col, "reason": f"Summary of '{col}'.", "required_cols_specific": [col]}, {"name": "Density Plot", "for_col": col, "reason": f"Smooth distribution of '{col}'.", "required_cols_specific": [col]}])
    if ("one" in ua_count or "1" in ua_count) and ("prop" in ua_msg or "share" in ua_msg or "freq" in ua_msg or "count" in ua_msg or "val" in ua_msg) and ("cat" in ua_types or "any" in ua_types or not ua_types) and categorical_cols:
        for col in categorical_cols: suggestions.append({"name": "Bar Chart (Counts)", "for_col": col, "reason": f"Counts for categories in '{col}'.", "required_cols_specific": [col]}); try: nunique=df_sample[col].nunique(); if 1 < nunique < 8: suggestions.append({"name": "Pie Chart", "for_col": col, "reason": f"Proportions for '{col}'.", "required_cols_specific": [col]}) except: pass
    if ("two" in ua_count or "2" in ua_count) and ("relat" in ua_msg or "corr" in ua_msg or "scat" in ua_msg) and ("num" in ua_types or "any" in ua_types or not ua_types) and len(numerical_cols)>=2:
        for i,c1 in enumerate(numerical_cols): suggestions.extend({"name": "Scatter Plot", "for_cols": f"{c1} & {c2}", "reason": f"Relationship: '{c1}' vs '{c2}'.", "required_cols_specific": [c1, c2]} for j,c2 in enumerate(numerical_cols) if j>i)
    if ("two" in ua_count or "2" in ua_count) and ("comp" in ua_msg or "across" in ua_msg or "group" in ua_msg or "dist" in ua_msg) and ("mix" in ua_types or "cat" in ua_types or "num" in ua_types or "any" in ua_types or not ua_types) and distributable_numeric_cols and categorical_cols:
        for num_col in distributable_numeric_cols: suggestions.extend([{"name": "Box Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "reason": f"Distribution of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]}, {"name": "Violin Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "reason": f"Density/distribution of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]}] for cat_col in categorical_cols if num_col != cat_col)
        if numerical_cols and categorical_cols: suggestions.extend([{"name": "Bar Chart (Aggregated)", "for_cols": f"Avg of {num_col} by {cat_col}", "reason": f"Average '{num_col}' for each category in '{cat_col}'.", "required_cols_specific": [cat_col, num_col]} for num_col in numerical_cols for cat_col in categorical_cols if num_col != cat_col])
    if ("two" in ua_count or "2" in ua_count) and ("relat" in ua_msg or "comp" in ua_msg or "cont" in ua_msg or "joint" in ua_msg) and ("cat" in ua_types or "any" in ua_types or not ua_types) and len(categorical_cols)>=2:
        for i,c1 in enumerate(categorical_cols): suggestions.extend([{"name": "Grouped Bar Chart", "for_cols": f"{c1} & {c2}", "reason": f"Counts of '{c1}' grouped by '{c2}'.", "required_cols_specific": [c1, c2]}, {"name": "Heatmap (Counts)", "for_cols": f"{c1} & {c2}", "reason": f"Co-occurrence frequency of '{c1}' & '{c2}'.", "required_cols_specific": [c1, c2]}] for j,c2 in enumerate(categorical_cols) if j>i)
    if (("two" in ua_count or "2" in ua_count) or "time" in ua_types or "trend" in ua_msg) and datetime_cols and numerical_cols:
        for dt in datetime_cols: suggestions.extend([{"name": "Line Chart", "for_cols": f"{num} over {dt}", "reason": f"Trend of '{num}' over '{dt}'.", "required_cols_specific": [dt, num]}, {"name": "Area Chart", "for_cols": f"{num} over {dt}", "reason": f"Cumulative trend of '{num}' over '{dt}'.", "required_cols_specific": [dt, num]}] for num in numerical_cols)
    if ("more" in ua_count or "mult" in ua_count or "pair" in ua_msg or "heat" in ua_msg or "para" in ua_msg) or ((ua_count not in ["one","1","two","2"]) and (len(numerical_cols)>2 or len(categorical_cols)>2)):
        if len(numerical_cols)>=3: suggestions.extend([{"name": "Pair Plot", "reason": "Pairwise relationships (numerical).", "required_cols_specific": numerical_cols[:min(4,len(numerical_cols))]}, {"name": "Correlation Heatmap", "reason": "Correlation matrix (numerical).", "required_cols_specific": numerical_cols}, {"name": "Parallel Coordinates Plot", "reason": "Compare multiple numerical variables.", "required_cols_specific": numerical_cols[:min(6,len(numerical_cols))]}] )

    final_suggestions_dict = {}; suggestions_order = []
    for s in suggestions: s_key = s["name"] + ("_" + "_".join(sorted(s.get("required_cols_specific",[]))) if s.get("required_cols_specific") else ""); if s_key not in final_suggestions_dict: final_suggestions_dict[s_key] = s; suggestions_order.append(s_key)
    final_suggestions = [final_suggestions_dict[key] for key in suggestions_order]

    if not final_suggestions: final_suggestions.append({"name": "No specific chart matched well", "type": "Info", "reason": "Criteria didn't match. Pick columns manually?", "required_cols_specific": []})
    if not any(s['name']=="Pick columns manually" for s in final_suggestions): final_suggestions.append({"name": "Pick columns manually", "type": "Action", "reason": "Choose columns yourself.", "required_cols_specific": []})
    return final_suggestions


# --- Validation Function (UPDATED VERSION) ---
def validate_columns_for_chart(chart_type: str, columns: List[str], df: pd.DataFrame) -> Optional[str]:
    """Validates if columns are suitable for chart type. Returns error message or None."""
    if not columns: return "No columns selected."
    if not all(col in df.columns for col in columns): missing = [col for col in columns if col not in df.columns]; return f"Column(s) not found: {', '.join(missing)}."
    col_types = get_simplified_column_types(df[columns]); num_numerical = sum(1 for t in col_types.values() if t == 'numerical'); num_categorical = sum(1 for t in col_types.values() if t in ['categorical', 'categorical_numeric']); num_distributable_numeric = sum(1 for t in col_types.values() if t in ['numerical', 'categorical_numeric']); num_datetime = sum(1 for t in col_types.values() if t == 'datetime'); num_selected = len(columns)
    requirements = {"Histogram":{'exact_cols':1,'distributable_numeric':1},"Box Plot":{'exact_cols':1,'distributable_numeric':1},"Density Plot":{'exact_cols':1,'distributable_numeric':1},"Bar Chart (Counts)":{'exact_cols':1,'categorical':1},"Pie Chart":{'exact_cols':1,'categorical':1},"Scatter Plot":{'exact_cols':2,'numerical':2},"Line Chart":{'exact_cols':2,'numerical':(1,2)},"Box Plots (by Category)":{'exact_cols':2,'categorical':1,'distributable_numeric':1},"Violin Plots (by Category)":{'exact_cols':2,'categorical':1,'distributable_numeric':1},"Bar Chart (Aggregated)":{'exact_cols':2,'categorical':1,'numerical':1},"Grouped Bar Chart":{'exact_cols':2,'categorical':2},"Heatmap (Counts)":{'exact_cols':2,'categorical':2},"Area Chart":{'exact_cols':2,'numerical':(1,2)},"Pair Plot":{'min_cols':3,'numerical':3},"Correlation Heatmap":{'min_cols':2,'numerical':2},"Parallel Coordinates Plot":{'min_cols':3,'numerical':3}}
    if chart_type == "Bar Chart":
         if num_selected == 1: req = requirements.get("Bar Chart (Counts)",{}); req['exact_cols']=1
         elif num_selected == 2:
             if num_categorical==2: req = requirements.get("Grouped Bar Chart",{}); req['exact_cols']=2
             elif num_categorical==1 and num_numerical==1: req = requirements.get("Bar Chart (Aggregated)",{}); req['exact_cols']=2
             else: return "Bar Chart(2 cols) needs 2 categorical, or 1 cat & 1 num."
         else: return "Bar Chart needs 1 or 2 columns."
    elif chart_type in requirements: req = requirements[chart_type]
    else: return None # Skip validation if unknown
    if 'exact_cols' in req and num_selected != req['exact_cols']: return f"needs {req['exact_cols']} col(s), you chose {num_selected}"
    if 'min_cols' in req and num_selected < req['min_cols']: return f"needs at least {req['min_cols']} cols, you chose {num_selected}"
    err_msg = ""; type_error = False
    if 'numerical' in req and isinstance(req['numerical'],int) and num_numerical<req['numerical']: err_msg+=f" {req['numerical']} numerical(found {num_numerical})."; type_error=True
    elif 'numerical' in req and isinstance(req['numerical'],tuple) and not (req['numerical'][0]<=num_numerical<=req['numerical'][1]): err_msg+=f" {req['numerical'][0]}-{req['numerical'][1]} numerical(found {num_numerical})."; type_error=True
    if 'categorical' in req and isinstance(req['categorical'],int) and num_categorical<req['categorical']: err_msg+=f" {req['categorical']} categorical(found {num_categorical})."; type_error=True
    if 'distributable_numeric' in req and isinstance(req['distributable_numeric'],int) and num_distributable_numeric<req['distributable_numeric']: err_msg+=f" {req['distributable_numeric']} numerical/rating(found {num_distributable_numeric})."; type_error=True
    if 'datetime' in req and isinstance(req['datetime'],int) and num_datetime<req['datetime']: err_msg+=f" {req['datetime']} datetime(found {num_datetime})."; type_error=True
    if type_error: return f"needs specific column types:{err_msg}"
    return None

# --- Plotting Function ---
def generate_plot_and_get_uri(filepath, chart_type, columns):
    """Generates plot and returns base64 URI or (None, error_msg)."""
    if not filepath: return None, "File path missing."
    try:
        df_full = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
        if not all(col in df_full.columns for col in columns): missing_cols = [c for c in columns if c not in df_full.columns]; return None, f"Column(s) not found: {', '.join(missing_cols)}."
        # Validate using the full dataframe's column types for accuracy
        validation_error = validate_columns_for_chart(chart_type, columns, df_full[columns].copy()) # Pass subset to validation too
        if validation_error: return None, f"Invalid columns for {chart_type}: {validation_error}"
        df_plot = df_full[columns].copy() # Now subset for plotting
    except Exception as e: print(f"Error reading/validating dataframe ('{filepath}'): {e}"); return None, f"Error reading/validating data: {str(e)[:100]}"

    img = io.BytesIO(); plt.figure(figsize=(7.5, 5)); plt.style.use('seaborn-v0_8-whitegrid'); original_chart_type = chart_type; plot_title_detail = ""
    try:
        print(f"Attempting plot: {chart_type} with {columns}"); col_types_specific = get_simplified_column_types(df_plot); mapped_chart_type = chart_type; plot_columns = columns[:]
        if chart_type == "Bar Chart":
             if len(plot_columns)==1 and plot_columns[0] in col_types_specific and col_types_specific[plot_columns[0]] in ['categorical','categorical_numeric']: mapped_chart_type="Bar Chart (Counts)"; plot_title_detail=f" for {plot_columns[0]}"
             elif len(plot_columns)==2:
                  cat_cols=[c for c in plot_columns if col_types_specific.get(c) in ['categorical','categorical_numeric']]; num_cols=[c for c in plot_columns if col_types_specific.get(c)=='numerical']
                  if len(cat_cols)==1 and len(num_cols)==1: mapped_chart_type="Bar Chart (Aggregated)"; plot_columns=[cat_cols[0],num_cols[0]]; plot_title_detail=f" for {num_cols[0]} by {cat_cols[0]}"
                  elif len(cat_cols)==2: mapped_chart_type="Grouped Bar Chart"; plot_title_detail=f" for {plot_columns[0]} by {plot_columns[1]}"
                  else: raise ValueError("Cannot determine Bar Chart type.")
             else: raise ValueError("Bar Chart needs 1 or 2 columns.")
             print(f"--> Handling '{original_chart_type}' as '{mapped_chart_type}'")
        plot_title=f"{mapped_chart_type}{plot_title_detail}"
        if mapped_chart_type=="Histogram": sns.histplot(data=df_plot, x=plot_columns[0], kde=True); plot_title=f"Histogram of {plot_columns[0]}"
        elif mapped_chart_type=="Box Plot": sns.boxplot(data=df_plot, y=plot_columns[0]); plot_title=f"Box Plot of {plot_columns[0]}"
        elif mapped_chart_type=="Density Plot": sns.kdeplot(data=df_plot, x=plot_columns[0], fill=True); plot_title=f"Density Plot of {plot_columns[0]}"
        elif mapped_chart_type=="Bar Chart (Counts)": counts=df_plot[plot_columns[0]].value_counts().nlargest(20); sns.barplot(x=counts.index, y=counts.values); plot_title=f"Top Counts for {plot_columns[0]}"; plt.ylabel("Count"); plt.xlabel(plot_columns[0]); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Pie Chart": counts=df_plot[plot_columns[0]].value_counts(); effective_counts=counts.nlargest(7); if len(counts)>7: effective_counts.loc['Other']=counts.iloc[7:].sum(); plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85); plot_title=f"Pie Chart of {plot_columns[0]}"; plt.axis('equal')
        elif mapped_chart_type=="Scatter Plot": sns.scatterplot(data=df_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Scatter: {plot_columns[0]} vs {plot_columns[1]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1])
        elif mapped_chart_type=="Line Chart":
            df_to_plot=df_plot.copy(); sort_col=plot_columns[0]
            try:
                 if not pd.api.types.is_datetime64_any_dtype(df_to_plot[sort_col]): df_to_plot[sort_col]=pd.to_datetime(df_to_plot[sort_col])
                 df_to_plot=df_to_plot.sort_values(by=sort_col)
            except: pass # Ignore conversion errors, still try to sort
            if pd.api.types.is_numeric_dtype(df_to_plot[sort_col]): df_to_plot=df_to_plot.sort_values(by=sort_col)
            sns.lineplot(data=df_to_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Line: {plot_columns[1]} over {plot_columns[0]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1]); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif mapped_chart_type=="Box Plots (by Category)": sns.boxplot(data=df_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Box Plots: {plot_columns[1]} by {plot_columns[0]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1]); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Violin Plots (by Category)": sns.violinplot(data=df_plot, x=plot_columns[0], y=plot_columns[1]); plot_title=f"Violin Plots: {plot_columns[1]} by {plot_columns[0]}"; plt.xlabel(plot_columns[0]); plt.ylabel(plot_columns[1]); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Bar Chart (Aggregated)": cat_col,num_col=plot_columns[0],plot_columns[1]; agg_data=df_plot.groupby(cat_col)[num_col].mean().nlargest(20); sns.barplot(x=agg_data.index, y=agg_data.values); plot_title=f"Mean of {num_col} by {cat_col}"; plt.xlabel(cat_col); plt.ylabel(f"Mean of {num_col}"); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Grouped Bar Chart": col1_tc=df_plot[plot_columns[0]].value_counts().nlargest(10).index; col2_tc=df_plot[plot_columns[1]].value_counts().nlargest(5).index; df_f=df_plot[df_plot[plot_columns[0]].isin(col1_tc) & df_plot[plot_columns[1]].isin(col2_tc)]; sns.countplot(data=df_f, x=plot_columns[0], hue=plot_columns[1]); plot_title=f"Counts: {plot_columns[0]} by {plot_columns[1]}"; plt.xlabel(plot_columns[0]); plt.ylabel("Count"); plt.xticks(rotation=65, ha='right', fontsize=9); plt.legend(title=plot_columns[1], fontsize='x-small', title_fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')
        # Add other specific plot types here
        else: raise NotImplementedError(f"Plot type '{mapped_chart_type}' is not explicitly implemented.")
        plt.title(plot_title, fontsize=12); plt.tight_layout(pad=1.0); plt.savefig(img, format='png', bbox_inches='tight'); plt.close(); img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8'); print(f"Success: {original_chart_type} (as {mapped_chart_type})"); return f"data:image/png;base64,{plot_url}", None
    except Exception as e: error_info = f"{type(e).__name__}: {str(e)}"; print(f"!!! Plot Error: '{mapped_chart_type}' with {plot_columns}: {error_info}"); error_message = f"Failed: {original_chart_type} ({error_info[:100]}...)."; if 'plt' in locals() and plt.get_fignums(): plt.close('all'); return None, error_message

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

    # --- Explicit command checks BEFORE state machine ---
    # Use lower() for case-insensitive matching
    user_input_lower = user_input.lower() if user_input else ""

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

    # --- State Machine Logic (with validation integrated) ---
    current_viz_state = session.get('visualization_questions_state')
    df_columns = session.get('df_columns', [])
    uploaded_filepath = session.get('uploaded_filepath')
    user_answers = {'variable_types': session.get('user_answer_variable_types', ''), 'message_insight': session.get('user_answer_visualization_message', ''), 'variable_count': session.get('user_answer_variable_count', '')}

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
        if "suggest chart" in user_input_lower: # Changed to lower
            df_sample = None; bot_reply = ""; response_data_suggestions = []
            if uploaded_filepath:
                try: df_sample = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e: print(f"Error reading df_sample: {e}")
            # --- Debug Suggester Inputs ---
            print(f"\n--- DEBUG Suggester ---"); print(f"User Answers: {user_answers}")
            col_types_sample = {};
            if df_sample is not None and not df_sample.empty: col_types_sample = get_simplified_column_types(df_sample); print(f"Sample Col Types: {col_types_sample}")
            else: print("Sample Col Types: (Could not read or empty sample)")
            # --- End Debug ---
            chart_suggestions = suggest_charts_based_on_answers(user_answers, df_sample); session['chart_suggestions_list'] = chart_suggestions
            print(f"Chart Suggestions Returned: {[s.get('name') for s in chart_suggestions]}")
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
            else: bot_reply = "Couldn't find specific suggestions. Try picking columns."; response_data = {"suggestions": ["Let me pick columns", "Restart questions"]}; session['visualization_questions_state'] = 'awaiting_column_selection_general'
        elif "choose columns" in user_input_lower or "pick columns" in user_input_lower:
             if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
             else: bot_reply = f"Sure! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
        # Restart is handled globally now
        # elif "restart" in user_input_lower: ...
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
                    try: df_val = pd.read_csv(uploaded_filepath, usecols=required_cols_specific, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=required_cols_specific, nrows=5); validation_msg = validate_columns_for_chart(chart_name, required_cols_specific, df_val)
                    except Exception as e: validation_msg = f"Couldn't validate ({str(e)[:50]}...)."; print(f"DEBUG Validation Read Error: {e}")
                    print(f"DEBUG Pre-validation for {chart_name}, cols {required_cols_specific}: {validation_msg or 'Passed'}") # <<< DEBUG PRINT
                if validation_msg is None: bot_reply += f"Suggest using: <strong>{cols_to_use_str}</strong>. Plot?"; session['plotting_columns'] = required_cols_specific; response_data = {"suggestions": [f"Yes, plot {chart_name}", "Choose other columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
                else: bot_reply += f"Suggested cols '{cols_to_use_str}' may not work ({validation_msg}).<br>Select columns for {chart_name}. Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
            else: bot_reply += f"Which columns for {chart_name}? Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        else: bot_reply = "Didn't recognize that chart. Choose again."; response_data = {"suggestions": session.get('last_suggestions', [])}

    elif current_viz_state == 'confirm_plot_details':
        chart_to_plot_info = session.get('selected_chart_for_plotting'); cols_for_plot = session.get('plotting_columns')
        if user_input.startswith("Yes, plot"):
            if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                chart_name = chart_to_plot_info['name']; bot_reply = f"Generating <strong>{chart_name}</strong> for: {', '.join(cols_for_plot)}."
                plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_name, cols_for_plot)
                if plot_image_uri: response_data["plot_image"] = plot_image_uri
                else: bot_reply += f"<br><strong>Plot Error:</strong> {error_msg or 'Unknown.'}"
                session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])
            else: bot_reply = "Missing details/file path."; session['visualization_questions_state'] = 'visualization_info_gathered'; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
        elif "choose other columns" in user_input_lower or "change columns" in user_input_lower:
             chart_name = chart_to_plot_info['name'] if chart_to_plot_info else 'chart'; bot_reply = f"Okay, for <strong>{chart_name}</strong>, which columns? Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        else: bot_reply = "Okay, back to chart list."; session['visualization_questions_state'] = 'awaiting_chart_type_selection'; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}

    elif current_viz_state == 'awaiting_columns_for_selected_chart':
        # Restart check is handled globally, no need here
        potential_cols_str = user_input.replace("Use:", "").strip(); user_selected_cols = [col.strip() for col in potential_cols_str.split(',') if col.strip() and col.strip() in df_columns]
        chart_to_plot_info = session.get('selected_chart_for_plotting'); chart_name = chart_to_plot_info.get('name', 'chart') if chart_to_plot_info else 'chart'
        if user_selected_cols:
            validation_msg = None
            if uploaded_filepath:
                 try: df_val = pd.read_csv(uploaded_filepath, usecols=user_selected_cols, nrows=5) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=user_selected_cols, nrows=5); validation_msg = validate_columns_for_chart(chart_name, user_selected_cols, df_val)
                 except Exception as e: validation_msg = f"Couldn't validate ({str(e)[:50]}...)."; print(f"DEBUG Post-Validation Read Error: {e}")
                 print(f"DEBUG Post-validation for {chart_name}, cols {user_selected_cols}: {validation_msg or 'Passed'}")
            if validation_msg is None:
                session['plotting_columns'] = user_selected_cols; bot_reply = f"Using: <strong>{', '.join(user_selected_cols)}</strong> for <strong>{chart_name}</strong>. Plot?"; response_data = {"suggestions": ["Yes, generate plot", "Change columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
            else: bot_reply = f"Columns <strong>{', '.join(user_selected_cols)}</strong> might not work for <strong>{chart_name}</strong>. ({validation_msg})<br>Select others from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        elif "list all columns" in user_input_lower: bot_reply = f"Available: {', '.join(df_columns)}.<br>Which for {chart_name}?"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        elif "back to chart list" in user_input_lower:
            session['visualization_questions_state'] = 'awaiting_chart_type_selection'; bot_reply = "Okay, which chart type?"; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}
        else: bot_reply = f"Invalid columns for <strong>{chart_name}</strong>. Choose from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}

    elif current_viz_state == 'awaiting_column_selection_general':
         # Restart check handled globally
        if "finished selecting" in user_input_lower:
            selected_cols = session.get('manual_columns_selected', [])
            if selected_cols: bot_reply = f"Selected: <strong>{', '.join(selected_cols)}</strong>. What chart?"; session['plotting_columns'] = selected_cols; session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'; response_data = {"suggestions": ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]}
            else: bot_reply = "No columns selected. List columns or cancel."; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Cancel selection"]} # Removed Finished selecting here
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
            plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_type_from_user, cols_for_plot)
            if plot_image_uri: response_data["plot_image"] = plot_image_uri
            else: bot_reply += f"<br><strong>Plot Error:</strong> {error_msg or 'Unknown.'}"
        else: bot_reply = "Missing column/file details."
        session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])

    else: # Fallback to NLU / No specific state OR if global commands didn't return early
        if not bot_reply: # Only call NLU if no other state handled it and no global command returned early
            nlu_output = nlu_get_bot_response(user_input)
            if isinstance(nlu_output, dict): bot_reply = nlu_output.get("response", "Sorry..."); temp_suggestions = nlu_output.get("suggestions", [])
            else: bot_reply = nlu_output; temp_suggestions = []
            if not temp_suggestions: temp_suggestions = ["Help", "Upload Data", "What can you do?"]
            response_data = {"suggestions": temp_suggestions}
        # If bot_reply was set by a global command handler (like lost context for 'suggest another'), package it now
        elif not response_data: # Ensure response_data is initialized if only bot_reply was set
             response_data = {"suggestions": ["Help", "Upload Data"]}


    # --- Final response packaging ---
    response_data["response"] = bot_reply
    session['last_suggestions'] = response_data.get("suggestions", [])
    return jsonify(response_data)

# --- Upload Route ---
@app.route("/upload_file", methods=["POST"])
def upload_file():
    """Handles file upload, validation, and starts the question flow."""
    # ... (Keep upload logic as is) ...
    if "file" not in request.files: return jsonify({"response": "No file part in request."}), 400
    file = request.files["file"]
    if file.filename == "": return jsonify({"response": "No file selected."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename); home(); filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath); session['uploaded_filepath'] = filepath; session['uploaded_filename'] = filename
            df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
            session['df_columns'] = list(df.columns); preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)
            total_rows,total_columns=len(df),len(df.columns); missing_values=df.isnull().sum().sum(); duplicate_rows=df.duplicated().sum(); total_cells=total_rows*total_columns; missing_percent=(missing_values/total_cells)*100 if total_cells else 0
            initial_bot_message = (f"✅ <strong>{filename}</strong> uploaded.<br><br>"
                                 f"🔍 Quality Check: {total_rows} R, {total_columns} C; "
                                 f"{missing_values} missing ({missing_percent:.1f}%); {duplicate_rows} duplicates.<br><br>"
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
    app.run(debug=True)