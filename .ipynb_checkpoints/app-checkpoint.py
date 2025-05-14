import base64
import io
import os
import re # Import regex for cleaning
from typing import List, Optional

import pandas as pd
import numpy as np # Ensure NumPy is imported
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

def clean_numeric_column(series):
    """Attempt to clean a pandas Series to be numeric (handles $, ,, % )."""
    if series is None: return None
    if is_numeric_dtype(series): return series # Already numeric
    if series.dtype == 'object':
        try:
            cleaned_series = series.astype(str).str.replace(r'[$,%]', '', regex=True).str.strip()
            cleaned_series = cleaned_series.replace('', pd.NA)
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            if numeric_series.notna().any(): return numeric_series
        except Exception as e: print(f"Cleaning failed for column '{series.name}': {e}"); pass
    return series

def get_simplified_column_types(df):
    """Analyzes DataFrame columns (after potential cleaning) and returns simplified types."""
    simplified_types = {}
    if df is None or df.empty: return simplified_types
    # print(f"DEBUG: Inferring types for columns:", df.columns.tolist()) # Optional debug
    for col in df.columns:
        original_dtype_str = str(df[col].dtype)
        try:
            temp_series = df[col]
            if temp_series.dtype == 'object' and not any(substr in col.lower() for substr in ['date', 'time', 'yr', 'year', 'id', 'code', 'name']): temp_series = clean_numeric_column(temp_series)
            dtype = temp_series.dtype; unique_count = temp_series.nunique(dropna=True); non_null_count = temp_series.count()
            if non_null_count == 0: simplified_types[col] = 'empty'; continue
            if is_numeric_dtype(dtype): is_integer = pd.api.types.is_integer_dtype(dtype); simplified_types[col] = 'categorical_numeric' if unique_count < 15 and (unique_count < non_null_count * 0.1 or unique_count < 7) and is_integer else 'numerical'
            elif is_datetime64_any_dtype(dtype) or any(substr in col.lower() for substr in ['date', 'time', 'yr', 'year']):
                 if not is_datetime64_any_dtype(dtype):
                      try: pd.to_datetime(temp_series, errors='raise'); simplified_types[col] = 'datetime'
                      except: simplified_types[col] = 'categorical' if unique_count < 150 else 'id_like_text'
                 else: simplified_types[col] = 'datetime'
            elif is_string_dtype(dtype) or dtype == 'object': simplified_types[col] = 'categorical' if unique_count <= 1 or (unique_count < max(2, non_null_count * 0.7) and unique_count < 250) else 'id_like_text'
            else: simplified_types[col] = 'other'
        except Exception as e: print(f"Warning: Type check failed for '{col}' (Original dtype: {original_dtype_str}): {e}"); simplified_types[col] = 'other'
    # print(f"DEBUG Simplified Types: {simplified_types}") # Optional debug
    return simplified_types

def suggest_charts_based_on_answers(user_answers, df_sample):
    """Suggests chart types based on user preferences and data sample."""
    suggestions = [];
    if df_sample is None or df_sample.empty: return [{"name": "Cannot suggest: Data sample missing or unreadable.", "type": "Info"}]
    col_types = get_simplified_column_types(df_sample)
    numerical_cols = [c for c,t in col_types.items() if t=='numerical']; categorical_cols = [c for c,t in col_types.items() if t in ['categorical', 'categorical_numeric']]; distributable_numeric_cols = [c for c,t in col_types.items() if t in ['numerical', 'categorical_numeric']]; datetime_cols = [c for c,t in col_types.items() if t=='datetime']
    ua_count, ua_types, ua_msg = user_answers.get('variable_count','').lower(), user_answers.get('variable_types','').lower(), user_answers.get('message_insight','').lower()
    # --- Suggestion Rules ---
    if ("one" in ua_count or "1" in ua_count) and ("dist" in ua_msg or "spread" in ua_msg or "summ" in ua_msg) and ("num" in ua_types or "cat" in ua_types or "any" in ua_types or not ua_types) and distributable_numeric_cols:
        for col in distributable_numeric_cols: suggestions.extend([{"name": "Histogram", "for_col": col, "type": "Univariate Numerical", "reason": f"Distribution of '{col}'.", "required_cols_specific": [col]}, {"name": "Box Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Summary of '{col}'.", "required_cols_specific": [col]}, {"name": "Density Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Smooth distribution of '{col}'.", "required_cols_specific": [col]}])
    if ("one" in ua_count or "1" in ua_count) and ("prop" in ua_msg or "share" in ua_msg or "freq" in ua_msg or "count" in ua_msg or "val" in ua_msg) and ("cat" in ua_types or "any" in ua_types or not ua_types) and categorical_cols:
        for col in categorical_cols: # Multi-line structure
            suggestions.append({"name": "Bar Chart (Counts)", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows counts for categories in '{col}'.", "required_cols_specific": [col]})
            try:
                if col in df_sample.columns: nunique = df_sample[col].nunique(dropna=True)
                else: nunique = 0
                if 1 < nunique < 8: suggestions.append({"name": "Pie Chart", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows proportions for '{col}'.", "required_cols_specific": [col]})
            except Exception as e: print(f"Warning during pie chart suggestion for {col}: {e}")
    if ("two" in ua_count or "2" in ua_count) and ("relat" in ua_msg or "corr" in ua_msg or "scat" in ua_msg) and ("num" in ua_types or "any" in ua_types or not ua_types) and len(numerical_cols)>=2:
        for i,c1 in enumerate(numerical_cols):
            for j,c2 in enumerate(numerical_cols):
                 if j > i: suggestions.append({"name": "Scatter Plot", "for_cols": f"{c1} & {c2}", "type": "Bivariate Numerical-Numerical", "reason": f"Relationship: '{c1}' vs '{c2}'.", "required_cols_specific": [c1, c2]})
    if ("two" in ua_count or "2" in ua_count) and ("comp" in ua_msg or "across" in ua_msg or "group" in ua_msg or "dist" in ua_msg) and ("mix" in ua_types or "cat" in ua_types or "num" in ua_types or "any" in ua_types or not ua_types) and distributable_numeric_cols and categorical_cols:
        for num_col in distributable_numeric_cols:
            for cat_col in categorical_cols:
                if num_col != cat_col: suggestions.extend([{"name": "Box Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Distribution of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]}, {"name": "Violin Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Density/distribution of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]}])
        if numerical_cols and categorical_cols:
            for num_col in numerical_cols:
                for cat_col in categorical_cols:
                    if num_col != cat_col: suggestions.append({"name": "Bar Chart (Aggregated)", "for_cols": f"Avg of {num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Average '{num_col}' for each category in '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})
    if ("two" in ua_count or "2" in ua_count) and ("relat" in ua_msg or "comp" in ua_msg or "cont" in ua_msg or "joint" in ua_msg) and ("cat" in ua_types or "any" in ua_types or not ua_types) and len(categorical_cols)>=2:
         for i,c1 in enumerate(categorical_cols):
              for j,c2 in enumerate(categorical_cols):
                  if j > i: suggestions.extend([{"name": "Grouped Bar Chart", "for_cols": f"{c1} & {c2}", "type": "Bivariate Categorical-Categorical", "reason": f"Counts of '{c1}' grouped by '{c2}'.", "required_cols_specific": [c1, c2]}, {"name": "Heatmap (Counts)", "for_cols": f"{c1} & {c2}", "type": "Bivariate Categorical-Categorical", "reason": f"Co-occurrence frequency of '{c1}' & '{c2}'.", "required_cols_specific": [c1, c2]}])
    if (("two" in ua_count or "2" in ua_count) or "time" in ua_types or "trend" in ua_msg) and datetime_cols and numerical_cols:
        for dt in datetime_cols:
             for num in numerical_cols: suggestions.extend([{"name": "Line Chart", "for_cols": f"{num} over {dt}", "reason": f"Trend of '{num}' over '{dt}'.", "required_cols_specific": [dt, num]}, {"name": "Area Chart", "for_cols": f"{num} over {dt}", "reason": f"Cumulative trend of '{num}' over '{dt}'.", "required_cols_specific": [dt, num]}])
    if ("more" in ua_count or "mult" in ua_count or "pair" in ua_msg or "heat" in ua_msg or "para" in ua_msg) or ((ua_count not in ["one","1","two","2"]) and (len(numerical_cols)>2 or len(categorical_cols)>2)):
        if len(numerical_cols)>=3: suggestions.extend([{"name": "Pair Plot", "reason": "Pairwise relationships (numerical).", "required_cols_specific": numerical_cols[:min(4,len(numerical_cols))]}, {"name": "Correlation Heatmap", "reason": "Correlation matrix (numerical).", "required_cols_specific": numerical_cols}, {"name": "Parallel Coordinates Plot", "reason": "Compare multiple numerical variables.", "required_cols_specific": numerical_cols[:min(6,len(numerical_cols))]}])
    # --- Deduplication Logic ---
    final_suggestions_dict = {}; suggestions_order = []
    for s in suggestions:
        if not isinstance(s, dict): print(f"Warning: Skipping non-dict item during deduplication: {s}"); continue
        req_cols = s.get("required_cols_specific", []); s_key_cols_str = "_".join(sorted(req_cols)) if isinstance(req_cols, list) else ""
        s_key = f"{s.get('name', 'UnknownChart')}_{s_key_cols_str}"
        if s_key not in final_suggestions_dict: final_suggestions_dict[s_key] = s; suggestions_order.append(s_key)
    final_suggestions = [final_suggestions_dict[key] for key in suggestions_order]
    if not final_suggestions: final_suggestions.append({"name": "No specific chart matched well", "type": "Info", "reason": "Criteria didn't match. Pick columns manually?", "required_cols_specific": []})
    if not any(s['name']=="Pick columns manually" for s in final_suggestions): final_suggestions.append({"name": "Pick columns manually", "type": "Action", "reason": "Choose columns yourself.", "required_cols_specific": []})
    return final_suggestions


# --- Validation Function ---
def validate_columns_for_chart(chart_type: str, columns: List[str], df: pd.DataFrame) -> Optional[str]:
    """Validates columns for chart type. Returns USER-FRIENDLY error message or None."""
    if not columns: return "No columns selected."
    missing = [col for col in columns if col not in df.columns]
    if missing: return f"Column(s) not found: {', '.join(missing)}. Check spelling?"
    # Validate based on types IN THE PROVIDED DF (which should be cleaned by caller like generate_plot_and_get_uri)
    col_types = get_simplified_column_types(df)
    num_numerical = sum(1 for t in col_types.values() if t == 'numerical'); num_categorical = sum(1 for t in col_types.values() if t in ['categorical', 'categorical_numeric']); num_distributable = sum(1 for t in col_types.values() if t in ['numerical', 'categorical_numeric']); num_datetime = sum(1 for t in col_types.values() if t == 'datetime'); num_id_like = sum(1 for t in col_types.values() if t == 'id_like_text'); num_selected = len(columns)
    col_details = ", ".join([f"'{c}' (as {col_types.get(c, '?')})" for c in columns])
    requirements = {"Histogram":{'exact_cols':1,'distributable_numeric':1},"Box Plot":{'exact_cols':1,'distributable_numeric':1},"Density Plot":{'exact_cols':1,'distributable_numeric':1},"Bar Chart (Counts)":{'exact_cols':1,'categorical':1},"Pie Chart":{'exact_cols':1,'categorical':1},"Scatter Plot":{'exact_cols':2,'numerical':2},"Line Chart":{'exact_cols':2,'numerical':(1,2)},"Box Plots (by Category)":{'exact_cols':2,'categorical':1,'distributable_numeric':1},"Violin Plots (by Category)":{'exact_cols':2,'categorical':1,'distributable_numeric':1},"Bar Chart (Aggregated)":{'exact_cols':2,'categorical':1,'numerical':1},"Grouped Bar Chart":{'exact_cols':2,'categorical':2},"Heatmap (Counts)":{'exact_cols':2,'categorical':2},"Area Chart":{'exact_cols':2,'numerical':(1,2)},"Pair Plot":{'min_cols':3,'numerical':3},"Correlation Heatmap":{'min_cols':2,'numerical':2},"Parallel Coordinates Plot":{'min_cols':3,'numerical':3}}
    req = {}
    if chart_type == "Bar Chart":
         if num_selected == 1: req = requirements.get("Bar Chart (Counts)",{}); req['exact_cols']=1
         elif num_selected == 2:
             if num_categorical==2: req = requirements.get("Grouped Bar Chart",{}); req['exact_cols']=2
             elif num_categorical==1 and num_numerical==1: req = requirements.get("Bar Chart (Aggregated)",{}); req['exact_cols']=2
             else: return "needs either 2 categorical or 1 categorical & 1 numerical column."
         else: return "needs 1 or 2 columns."
    elif chart_type in requirements: req = requirements[chart_type]
    else: return None
    if 'exact_cols' in req and num_selected != req['exact_cols']: return f"needs exactly {req['exact_cols']} column(s), you chose {num_selected}"
    if 'min_cols' in req and num_selected < req['min_cols']: return f"needs at least {req['min_cols']} columns, you chose {num_selected}"
    err_msg_parts = []
    type_error = False
    target_num = req.get('numerical'); target_cat = req.get('categorical'); target_dist = req.get('distributable_numeric'); target_dt = req.get('datetime')
    if target_num is not None:
        num_needed = target_num[0] if isinstance(target_num, tuple) else target_num
        num_range_str = f"{target_num[0]}-{target_num[1]}" if isinstance(target_num, tuple) else str(target_num)
        if isinstance(target_num, int) and num_numerical < target_num: err_msg_parts.append(f"{target_num} numerical (found {num_numerical})"); type_error=True
        elif isinstance(target_num, tuple) and not (target_num[0] <= num_numerical <= target_num[1]): err_msg_parts.append(f"{num_range_str} numerical (found {num_numerical})"); type_error=True
    if target_cat is not None and num_categorical < target_cat: err_msg_parts.append(f"{target_cat} categorical (found {num_categorical})"); type_error=True
    if target_dist is not None and num_distributable < target_dist: err_msg_parts.append(f"{target_dist} numerical/rating-like (found {num_distributable})"); type_error=True
    if target_dt is not None and num_datetime < target_dt: err_msg_parts.append(f"{target_dt} datetime (found {num_datetime})"); type_error=True
    if type_error: return f"{chart_type} needs {', '.join(err_msg_parts)}. You provided columns: ({col_details})."
    if chart_type == "Pie Chart":
        if columns and columns[0] in df.columns: # Use df passed in
            try:
                nunique = df[columns[0]].nunique(dropna=True)
                if nunique > 10: return f"column '{columns[0]}' has too many categories ({nunique}) for a clear Pie Chart. Try a Bar Chart."
            except Exception as e: print(f"Warn: Could not check nunique for Pie Chart validation: {e}")
    if num_id_like > 0:
         id_cols = [c for c,t in col_types.items() if t == 'id_like_text']
         if 'categorical' in req and num_categorical < req['categorical']: return f"column '{id_cols[0]}' has too many unique text values (like names or IDs) to be used as a category here."
    return None

# --- Plotting Function (With Targeted Cleaning Before Validation) ---
def generate_plot_and_get_uri(filepath, chart_type, columns):
    """Generates plot and returns base64 URI or (None, error_msg)."""
    if not filepath: return None, "File path missing."
    try:
        df_full = pd.read_csv(filepath, low_memory=False) if filepath.endswith(".csv") else pd.read_excel(filepath)
        if not all(col in df_full.columns for col in columns): missing_cols = [c for c in columns if c not in df_full.columns]; return None, f"Column(s) not found: {', '.join(missing_cols)}."
        df_plot = df_full[columns].copy(); plot_columns = list(columns);
        print(f"DEBUG Plotting: Dtypes before cleaning for {chart_type}: {df_plot.dtypes.to_dict()}")
        cleaned_cols_info = {}
        for col_name in plot_columns:
            if col_name in df_plot.columns:
                original_dtype = df_plot[col_name].dtype
                if original_dtype == 'object' and not any(substr in col_name.lower() for substr in ['id','code','name','person','country','category','product','type','status','gender','region','city','state','date','time']):
                    cleaned_series = clean_numeric_column(df_plot[col_name])
                    if cleaned_series is not df_plot[col_name]: df_plot[col_name] = cleaned_series; cleaned_cols_info[col_name] = 'numeric'
                if any(substr in col_name.lower() for substr in ['date','time','yr','year']) and not is_datetime64_any_dtype(df_plot[col_name]):
                    try:
                        converted_date = pd.to_datetime(df_plot[col_name], errors='coerce')
                        if converted_date.notna().any(): df_plot[col_name] = converted_date; cleaned_cols_info[col_name] = 'datetime'; print(f"Cleaned '{col_name}' to datetime.")
                        else: print(f"Note: Datetime conversion for '{col_name}' resulted in all NaT.")
                    except Exception as e: print(f"Note: Failed datetime conversion for '{col_name}': {e}")
        print(f"DEBUG Plotting: Dtypes after cleaning for {chart_type}: {df_plot.dtypes.to_dict()}")
        validation_error = validate_columns_for_chart(chart_type, plot_columns, df_plot)
        if validation_error: return None, f"Invalid columns/data for {chart_type}: {validation_error}"
        print(f"DEBUG Plotting: Applying forward/backward fill for columns: {plot_columns}")
        df_plot.fillna(method='ffill', inplace=True); df_plot.fillna(method='bfill', inplace=True)
    except pd.errors.ParserError as pe: return None, f"Error reading data file. Details: {str(pe)[:100]}"
    except KeyError as ke: return None, f"Column(s) not found when reading file: {ke}. Check spelling/case."
    except Exception as e: print(f"Error reading/cleaning/validating dataframe ('{filepath}'): {e}"); return None, f"Error preparing data: {str(e)[:100]}"

    img = io.BytesIO(); plt.figure(figsize=(7.5, 5)); plt.style.use('seaborn-v0_8-whitegrid'); original_chart_type = chart_type; plot_title_detail = ""; mapped_chart_type = chart_type;
    try:
        print(f"Attempting plot generation: {original_chart_type} with {plot_columns} (using filled data)");
        col_types_final = get_simplified_column_types(df_plot) # Use types after fill for mapping Bar Chart
        if original_chart_type == "Bar Chart":
             if len(plot_columns)==1 and plot_columns[0] in col_types_final and col_types_final[plot_columns[0]] in ['categorical','categorical_numeric']: mapped_chart_type="Bar Chart (Counts)"; plot_title_detail=f" for {plot_columns[0]}"
             elif len(plot_columns)==2:
                  cat_cols=[c for c in plot_columns if col_types_final.get(c) in ['categorical','categorical_numeric']]; num_cols=[c for c in plot_columns if col_types_final.get(c)=='numerical']
                  if len(cat_cols)==1 and len(num_cols)==1: mapped_chart_type="Bar Chart (Aggregated)"; plot_columns=[cat_cols[0],num_cols[0]]; plot_title_detail=f" of {num_cols[0]} by {cat_cols[0]}"
                  elif len(cat_cols)==2: mapped_chart_type="Grouped Bar Chart"; plot_title_detail=f" for {plot_columns[0]} by {plot_columns[1]}"
                  else: raise ValueError(f"Cannot determine Bar Chart type for ({', '.join(plot_columns)}).")
             else: raise ValueError("Bar Chart needs 1 or 2 columns.")
             print(f"--> Handling '{original_chart_type}' as '{mapped_chart_type}' with columns {plot_columns}")
        plot_title=f"{mapped_chart_type}{plot_title_detail}"
        col1 = plot_columns[0]; col2 = plot_columns[1] if len(plot_columns) > 1 else None
        # --- Plotting Logic ---
        if mapped_chart_type=="Histogram": sns.histplot(data=df_plot, x=col1, kde=True); plot_title=f"Histogram of {col1}"
        elif mapped_chart_type=="Box Plot": sns.boxplot(data=df_plot, y=col1); plot_title=f"Box Plot of {col1}"
        elif mapped_chart_type=="Density Plot": sns.kdeplot(data=df_plot, x=col1, fill=True); plot_title=f"Density Plot of {col1}"
        elif mapped_chart_type=="Bar Chart (Counts)": counts=df_plot[col1].value_counts().nlargest(20); sns.barplot(x=counts.index.astype(str), y=counts.values); plot_title=f"Top Counts for {col1}"; plt.ylabel("Count"); plt.xlabel(col1); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type == "Pie Chart":
            counts = df_plot[col1].value_counts(); effective_counts = counts.nlargest(7)
            if len(counts) > 7: effective_counts.loc['Other'] = counts.iloc[7:].sum()
            plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85); plot_title = f"Pie Chart of {col1}"; plt.axis('equal')
        elif mapped_chart_type=="Scatter Plot": sns.scatterplot(data=df_plot, x=col1, y=col2); plot_title=f"Scatter: {col1} vs {col2}"; plt.xlabel(col1); plt.ylabel(col2)
        elif mapped_chart_type=="Line Chart":
            df_to_plot_lc=df_plot.copy(); sort_col_lc=col1
            try:
                 if is_datetime64_any_dtype(df_to_plot_lc[sort_col_lc]): df_to_plot_lc=df_to_plot_lc.sort_values(by=sort_col_lc)
                 elif is_numeric_dtype(df_to_plot_lc[sort_col_lc]): df_to_plot_lc=df_to_plot_lc.sort_values(by=sort_col_lc)
            except Exception as sort_e: print(f"Note: Could not sort for Line Chart: {sort_e}")
            sns.lineplot(data=df_to_plot_lc, x=col1, y=col2); plot_title=f"Line: {col2} over {col1}"; plt.xlabel(col1); plt.ylabel(col2); plt.xticks(rotation=45, ha='right', fontsize=9)
        elif mapped_chart_type=="Box Plots (by Category)": sns.boxplot(data=df_plot, x=col1, y=col2); plot_title=f"Box Plots: {col2} by {col1}"; plt.xlabel(col1); plt.ylabel(col2); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Violin Plots (by Category)": sns.violinplot(data=df_plot, x=col1, y=col2); plot_title=f"Violin Plots: {col2} by {col1}"; plt.xlabel(col1); plt.ylabel(col2); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Bar Chart (Aggregated)": cat_col,num_col=plot_columns[0],plot_columns[1]; agg_data=df_plot.groupby(cat_col)[num_col].mean().nlargest(20); sns.barplot(x=agg_data.index.astype(str), y=agg_data.values); plot_title=f"Mean of {num_col} by {cat_col}"; plt.xlabel(cat_col); plt.ylabel(f"Mean of {num_col}"); plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Grouped Bar Chart": col1_tc=df_plot[plot_columns[0]].value_counts().nlargest(10).index; col2_tc=df_plot[plot_columns[1]].value_counts().nlargest(5).index; df_f=df_plot[df_plot[plot_columns[0]].isin(col1_tc) & df_plot[plot_columns[1]].isin(col2_tc)]; sns.countplot(data=df_f, x=plot_columns[0], hue=plot_columns[1]); plot_title=f"Counts: {plot_columns[0]} by {plot_columns[1]}"; plt.xlabel(plot_columns[0]); plt.ylabel("Count"); plt.xticks(rotation=65, ha='right', fontsize=9); plt.legend(title=plot_columns[1], fontsize='x-small', title_fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')
        else: raise NotImplementedError(f"Plot type '{mapped_chart_type}' is not explicitly implemented.")
        plt.title(plot_title, fontsize=12); plt.tight_layout(pad=1.0); plt.savefig(img, format='png', bbox_inches='tight'); plt.close(); img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8'); print(f"Success: {original_chart_type} (as {mapped_chart_type})"); return f"data:image/png;base64,{plot_url}", None
    except Exception as e:
        error_info = f"{type(e).__name__}: {str(e)}"; print(f"!!! Error during plot generation execution for '{mapped_chart_type or original_chart_type}' with {plot_columns}: {error_info}"); error_message = f"Failed to generate {original_chart_type}. ({error_info[:100]}...).";
        if 'plt' in locals() and plt.get_fignums(): plt.close('all')
        return None, error_message

# --- Flask Routes ---
# (get_descriptive_stats_html should be defined here if used, or imported)
# For simplicity, I'll add it here now, ensuring numpy is available.
def get_descriptive_stats_html(df):
    """Generates HTML string for descriptive statistics."""
    if df is None or df.empty: return "<p>No data available to summarize.</p>"
    df_cleaned = df.copy()
    for col in df_cleaned.columns: # Clean before describe
        # Attempt to clean columns that look numeric but might be object
        if df_cleaned[col].dtype == 'object' and not any(substr in col.lower() for substr in ['date', 'time', 'yr', 'year', 'id', 'code', 'name', 'person', 'country', 'category', 'product', 'type', 'status', 'gender', 'region', 'city', 'state']):
            df_cleaned[col] = clean_numeric_column(df_cleaned[col])
    html_output = "<h4>Data Summary:</h4>"
    numerical_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist() # np used here
    categorical_cols = df_cleaned.select_dtypes(include='object').columns.tolist() # Original objects
    # Also consider columns we classified as categorical in get_simplified_column_types
    simplified_col_types = get_simplified_column_types(df_cleaned) # Get types of cleaned df
    other_categorical = [col for col, typ in simplified_col_types.items() if typ in ['categorical', 'categorical_numeric'] and col not in numerical_cols and col not in categorical_cols]
    all_cat_cols_for_summary = list(set(categorical_cols + other_categorical))

    if numerical_cols:
        try: desc_num = df_cleaned[numerical_cols].describe().transpose(); desc_num_html = desc_num[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_html(classes="preview-table", float_format='{:,.2f}'.format); html_output += f"<p><strong>Numerical Columns:</strong></p>{desc_num_html}"
        except Exception as e: html_output += f"<p>Could not generate numerical summary: {e}</p>"
    if all_cat_cols_for_summary:
        html_output += "<p><strong>Categorical Columns (Top 5 values):</strong></p>"
        for col in all_cat_cols_for_summary:
            try: counts = df_cleaned[col].value_counts().nlargest(5);
            if not counts.empty: counts_html = counts.to_frame().to_html(classes="preview-table", header=False); html_output += f"<p><em>{col}:</em></p>{counts_html}"
            except Exception as e: html_output += f"<p><em>{col}:</em> Error summarizing: {e}</p>"
    if not numerical_cols and not all_cat_cols_for_summary: html_output += "<p>No clear numerical or categorical columns found to summarize.</p>"
    return html_output

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
    uploaded_filepath = session.get('uploaded_filepath')

    # --- Explicit command checks BEFORE state machine ---
    if "restart questions" in user_input_lower or "restart" == user_input_lower:
        home(); bot_reply = "Okay, restarting questions.<br><br><strong>1. Variable types?</strong>"; session['visualization_questions_state'] = 'asking_variable_types'; response_data = {"suggestions": ["Categorical", "Numerical", "Time-series", "Mix", "Any"]}
    elif "suggest another chart" in user_input_lower:
        if uploaded_filepath and session.get('user_answer_variable_count') is not None: session['visualization_questions_state'] = 'visualization_info_gathered'; bot_reply = "Okay, let's look for other visualizations. How proceed?"; response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns", "Restart questions"]}
        else: bot_reply = "Let's start over. Please upload data."; home(); response_data = {"suggestions": ["Upload Data", "Help"]}
    elif "descriptive statistics" in user_input_lower or "data summary" in user_input_lower or "summarize data" in user_input_lower:
        if uploaded_filepath:
            try: df_full = pd.read_csv(uploaded_filepath) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath); stats_html = get_descriptive_stats_html(df_full); bot_reply = stats_html; response_data = {"preview": "", "suggestions": ["Suggest charts for me", "Let me choose columns", "Restart questions"]}; session['visualization_questions_state'] = 'visualization_info_gathered'
            except Exception as e: bot_reply = f"Sorry, couldn't generate summary: {str(e)[:100]}"; response_data = {"suggestions": ["Upload Data", "Help"]}
        else: bot_reply = "Please upload data first."; response_data = {"suggestions": ["Upload Data", "Help"]}
    else: # --- State Machine Logic ---
        current_viz_state = session.get('visualization_questions_state'); df_columns = session.get('df_columns', []); user_answers = {'variable_types': session.get('user_answer_variable_types', ''), 'message_insight': session.get('user_answer_visualization_message', ''), 'variable_count': session.get('user_answer_variable_count', '')}
        # ... (Insert the full state machine if/elif/else block from the previous CORRECTED version here) ...
        # For brevity, I'm only showing the start and end of where it goes.
        # Ensure the entire logic for states: asking_variable_types -> NLU else fallback is here.
        if current_viz_state == 'asking_variable_types':
            session['user_answer_variable_types'] = user_input; bot_reply = (f"Understood ({user_input}).<br><br><strong>2. Main message/insight?</strong><br>(e.g., compare, distribution, relationship, trend)"); session['visualization_questions_state'] = 'asking_visualization_message'; response_data = {"suggestions": ["Compare values/categories", "Show data distribution", "Identify relationships", "Track trends over time"]}
        elif current_viz_state == 'asking_visualization_message':
            session['user_answer_visualization_message'] = user_input; columns_reminder = ""
            if df_columns: columns_reminder = f"(Cols: {', '.join(df_columns[:3])}...)"
            bot_reply = (f"Goal: \"{user_input}\".<br><br><strong>3. How many variables per chart?</strong> {columns_reminder}<br>(e.g., one, two, more)"); session['visualization_questions_state'] = 'asking_variable_count'; response_data = {"suggestions": ["One variable", "Two variables", "More than two"]}
        elif current_viz_state == 'asking_variable_count':
            session['user_answer_variable_count'] = user_input
            bot_reply = (f"Preferences:<br>- Types: \"{user_answers['variable_types']}\"<br>- Goal: \"{user_answers['message_insight']}\"<br>- Vars: \"{user_input}\"<br><br>What next?"); session['visualization_questions_state'] = 'visualization_info_gathered'; response_data = {"suggestions": ["Show data summary","Suggest charts for me", "Let me choose columns", "Restart questions"]}
        elif current_viz_state == 'visualization_info_gathered':
            if "suggest chart" in user_input_lower:
                df_sample = None; bot_reply = ""; response_data_suggestions = []
                if uploaded_filepath:
                    try: # Multi-line try/except
                        if uploaded_filepath.endswith(".csv"): df_sample = pd.read_csv(uploaded_filepath, nrows=100)
                        else: df_sample = pd.read_excel(uploaded_filepath, nrows=100)
                    except Exception as e: print(f"Error reading df_sample: {e}"); df_sample = None
                print(f"\n--- DEBUG Suggester ---"); print(f"User Answers: {user_answers}"); col_types_sample = {};
                if df_sample is not None and not df_sample.empty: col_types_sample = get_simplified_column_types(df_sample); print(f"Sample Col Types: {col_types_sample}")
                else: print("Sample Col Types: (Could not read or empty sample)")
                chart_suggestions = suggest_charts_based_on_answers(user_answers, df_sample); session['chart_suggestions_list'] = chart_suggestions; print(f"Chart Suggestions Returned: {[s.get('name') for s in chart_suggestions]}")
                if chart_suggestions and chart_suggestions[0].get("type") not in ["Info", None]:
                     bot_reply = "Based on your info, consider:<br>"; suggestions_for_user_options = []; count = 0
                     for chart_sugg in chart_suggestions:
                         if chart_sugg.get("type") == "Action": continue
                         if count >= 2 : break; count += 1 # Limit suggestions further
                         bot_reply += f"<br><strong>{count}. {chart_sugg['name']}</strong>"; sugg_suffix = ""
                         if chart_sugg.get("for_col"): sugg_suffix = f" for '{chart_sugg['for_col']}'"
                         elif chart_sugg.get("for_cols"): sugg_suffix = f" (e.g., using '{chart_sugg['for_cols']}')"
                         bot_reply += f"{sugg_suffix}.<br><em>Use this to: {chart_sugg.get('reason', 'Understand your data better.')}</em><br>"
                         suggestions_for_user_options.append(f"Select: {chart_sugg['name']}")
                     bot_reply += "<br>Which one would you like to try? Or, you can choose columns yourself or see more suggestions."
                     manual_pick_option = "Pick columns manually"
                     if not any(s['name'] == manual_pick_option for s in chart_suggestions): chart_suggestions.append({"name": manual_pick_option, "type": "Action"})
                     response_data_suggestions = suggestions_for_user_options + ["Show more chart suggestions"] + [s['name'] for s in chart_suggestions if s.get("type") == "Action"]
                     if "Restart questions" not in response_data_suggestions: response_data_suggestions.append("Restart questions")
                     response_data = {"suggestions": response_data_suggestions[:5]}; session['visualization_questions_state'] = 'awaiting_chart_type_selection'
                else: bot_reply = "Hmm, I couldn't pinpoint specific chart suggestions with that info. You could try picking columns manually, or we can restart the questions."; response_data = {"suggestions": ["Let me pick columns", "Restart questions"]}; session['visualization_questions_state'] = 'awaiting_column_selection_general'
            elif "choose columns" in user_input_lower or "pick columns" in user_input_lower:
                 if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
                 else: bot_reply = f"Sure! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
            else: bot_reply = "What next?"; response_data = {"suggestions": session.get('last_suggestions', ["Show data summary", "Suggest charts", "Pick columns", "Restart"])}

        elif current_viz_state == 'awaiting_chart_type_selection':
            user_choice_str = user_input.replace("Select: ", "").strip(); chart_suggestions_list = session.get('chart_suggestions_list', []); selected_chart_info = next((sugg for sugg in chart_suggestions_list if sugg['name'] == user_choice_str), None)
            if user_choice_str == "Pick columns manually":
                 if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
                 else: bot_reply = f"Okay! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
            elif "show more chart suggestions" in user_input_lower:
                 # Display the next batch of suggestions
                 bot_reply = "Okay, here are some more chart types you might consider:<br>"
                 current_suggestions_shown = session.get('last_suggestions', []) # Get what was last shown
                 start_index = 0
                 for item in current_suggestions_shown: # Find how many actual chart suggestions were last shown
                     if item.startswith("Select: "): start_index +=1

                 suggestions_for_user_options = []
                 count = 0
                 for i, chart_sugg in enumerate(session.get('chart_suggestions_list', [])):
                     if chart_sugg.get("type") == "Action": continue
                     if i < start_index : continue # Skip already shown suggestions
                     if count >= 2: break # Show next 2-3
                     count += 1
                     bot_reply += f"<br><strong>{chart_sugg['name']}</strong>"; sugg_suffix = ""
                     if chart_sugg.get("for_col"): sugg_suffix = f" for '{chart_sugg['for_col']}'"
                     elif chart_sugg.get("for_cols"): sugg_suffix = f" (e.g., using '{chart_sugg['for_cols']}')"
                     bot_reply += f"{sugg_suffix}.<br><em>Use this to: {chart_sugg.get('reason', 'Understand data.')}</em><br>"
                     suggestions_for_user_options.append(f"Select: {chart_sugg['name']}")
                 if not suggestions_for_user_options: bot_reply = "No more specific chart types to suggest based on your initial answers. You can always pick columns manually."
                 else: bot_reply += "<br><br>Choose one, or pick columns yourself."
                 response_data_suggestions = suggestions_for_user_options + ["Pick columns manually", "Restart questions"]
                 response_data = {"suggestions": response_data_suggestions[:5]}; session['visualization_questions_state'] = 'awaiting_chart_type_selection' # Stay in this state

            elif selected_chart_info:
                session['selected_chart_for_plotting'] = selected_chart_info; chart_name = selected_chart_info['name']; required_cols_specific = selected_chart_info.get('required_cols_specific')
                bot_reply = f"Okay: <strong>{chart_name}</strong>. "
                if required_cols_specific:
                    cols_to_use_str = ", ".join(required_cols_specific); validation_msg = None
                    if uploaded_filepath:
                        try:
                            df_val = None;
                            if uploaded_filepath.endswith(".csv"): df_val = pd.read_csv(uploaded_filepath, usecols=required_cols_specific, nrows=5)
                            else: df_val = pd.read_excel(uploaded_filepath, usecols=required_cols_specific, nrows=5)
                            if df_val is not None and not df_val.empty:
                                for col_val in df_val.columns: df_val[col_val] = clean_numeric_column(df_val[col_val])
                                validation_msg = validate_columns_for_chart(chart_name, required_cols_specific, df_val)
                            else: validation_msg = "Could not read sample data."
                            print(f"DEBUG Pre-validation for {chart_name}, cols {required_cols_specific}: {validation_msg or 'Passed'}")
                        except Exception as e: validation_msg = f"Couldn't pre-validate ({str(e)[:50]}...)."; print(f"DEBUG Validation Read/Clean Error: {e}")
                    if validation_msg is None: bot_reply += f"Suggest using: <strong>{cols_to_use_str}</strong>. Plot?"; session['plotting_columns'] = required_cols_specific; response_data = {"suggestions": [f"Yes, plot {chart_name}", "Choose other columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
                    else: bot_reply += f"Suggested cols '{cols_to_use_str}' may not work ({validation_msg}).<br>Select columns for {chart_name}. Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
                else:
                     bot_reply += f"Which columns for {chart_name}? Available: {', '.join(df_columns)}"
                     if chart_name == "Histogram": bot_reply += "<br><i>Hint: Needs one numerical column.</i>"
                     session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
            else: bot_reply = "Didn't recognize that chart. Choose again."; response_data = {"suggestions": session.get('last_suggestions', [])}

        elif current_viz_state == 'confirm_plot_details':
            chart_to_plot_info = session.get('selected_chart_for_plotting'); cols_for_plot = session.get('plotting_columns')
            if user_input.startswith("Yes, plot"):
                if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                    chart_name = chart_to_plot_info['name']; bot_reply = f"Generating <strong>{chart_name}</strong>..."
                    plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_name, cols_for_plot)
                    if plot_image_uri: response_data["plot_image"] = plot_image_uri; bot_reply = f"Here is the <strong>{chart_name}</strong> for: {', '.join(cols_for_plot)}."
                    else: bot_reply = f"Sorry, couldn't generate the <strong>{chart_name}</strong>.<br><strong>Reason:</strong> {error_msg or 'Unknown error.'}<br>Try suggesting another chart or picking different columns."
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
                if uploaded_filepath:
                     try:
                         df_val = None
                         if uploaded_filepath.endswith(".csv"): df_val = pd.read_csv(uploaded_filepath, usecols=user_selected_cols, nrows=5)
                         else: df_val = pd.read_excel(uploaded_filepath, usecols=user_selected_cols, nrows=5)
                         if df_val is not None and not df_val.empty:
                              for col in df_val.columns: df_val[col] = clean_numeric_column(df_val[col])
                              validation_msg = validate_columns_for_chart(chart_name, user_selected_cols, df_val)
                         else: validation_msg = "Could not read sample data."
                         print(f"DEBUG Post-validation for {chart_name}, cols {user_selected_cols}: {validation_msg or 'Passed'}")
                     except Exception as e: validation_msg = f"Couldn't validate ({str(e)[:50]}...)."; print(f"DEBUG Validation Read/Clean Error: {e}")
                if validation_msg is None:
                    session['plotting_columns'] = user_selected_cols; bot_reply = f"Using: <strong>{', '.join(user_selected_cols)}</strong> for <strong>{chart_name}</strong>. Plot?"; response_data = {"suggestions": ["Yes, generate plot", "Change columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
                else: bot_reply = f"The columns <strong>{', '.join(user_selected_cols)}</strong> might not work for <strong>{chart_name}</strong>. <br><strong>Reason:</strong> {validation_msg}<br>Please select valid columns from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
            elif "list all columns" in user_input_lower: bot_reply = f"Available: {', '.join(df_columns)}.<br>Which for {chart_name}?"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
            elif "back to chart list" in user_input_lower:
                session['visualization_questions_state'] = 'awaiting_chart_type_selection'; bot_reply = "Okay, which chart type?"; chart_suggestions_list = session.get('chart_suggestions_list', []); temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]; temp_suggs.append("Pick columns manually"); response_data = {"suggestions": temp_suggs}
            else: bot_reply = f"Invalid columns for <strong>{chart_name}</strong>. Choose from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}

        elif current_viz_state == 'awaiting_column_selection_general':
            if "finished selecting" in user_input_lower:
                selected_cols = session.get('manual_columns_selected', [])
                if selected_cols: bot_reply = f"Selected: <strong>{', '.join(selected_cols)}</strong>. What chart?"; session['plotting_columns'] = selected_cols; session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'; response_data = {"suggestions": ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot"]}
                else: bot_reply = "No columns selected. List columns or cancel."; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Cancel selection"]}
            elif "cancel selection" in user_input_lower: session.pop('manual_columns_selected', None); session['visualization_questions_state'] = 'visualization_info_gathered'; bot_reply = "Selection cancelled."; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
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
                if plot_image_uri: response_data["plot_image"] = plot_image_uri; bot_reply = f"Here is the <strong>{chart_type_from_user}</strong> for: {', '.join(cols_for_plot)}."
                else: bot_reply += f"<br><strong>Plot Error:</strong> {error_msg or 'Unknown.'}" # Append error
            else: bot_reply = "Missing column/file details."
            session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])

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
            try: df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath, engine=read_engine)
            except Exception as read_err: print(f"Initial read error for {filename}: {read_err}"); df = pd.read_csv(filepath, encoding='latin1') if filename.endswith(".csv") else df # Try latin1 for CSV
            session['df_columns'] = list(df.columns); preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)
            total_rows,total_columns=len(df),len(df.columns); missing_values=df.isnull().sum().sum(); duplicate_rows=df.duplicated().sum(); total_cells=total_rows*total_columns; missing_percent=(missing_values/total_cells)*100 if total_cells else 0
            initial_bot_message = (f"✅ <strong>{filename}</strong> uploaded.<br><br>"
                                 f"🔍 Quality Check: {total_rows} R, {total_columns} C; {missing_values} missing ({missing_percent:.1f}%); {duplicate_rows} duplicates.<br><br>"
                                 f"Let's choose a visualization. I need info:<br><br>"
                                 f"<strong>1. Variable types?</strong> (e.g., text, numbers, dates)")
            session['visualization_questions_state'] = 'asking_variable_types'
            current_suggestions = ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of these types", "Not sure / Any", "Show data summary"]
            session['last_suggestions'] = current_suggestions
            return jsonify({"response": initial_bot_message, "preview": preview_html, "suggestions": current_suggestions})
        except Exception as e: home(); print(f"Error processing uploaded file {filename}: {e}"); return jsonify({"response": f"Error processing '{filename}': {str(e)[:100]}..."}), 500
    else: return jsonify({"response": "Invalid file type. Use CSV or Excel."}), 400

# --- Main Execution ---
if __name__ == "__main__":
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'figure.autolayout': True, 'figure.dpi': 90, 'font.size': 9})
    app.run(debug=True) # Set debug=False for production