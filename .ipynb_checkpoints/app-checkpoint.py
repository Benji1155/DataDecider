import base64
import io
import os
import re 
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np 
from flask import Flask, jsonify, render_template, request, session
from pandas.api.types import (is_datetime64_any_dtype, is_numeric_dtype,
                              is_string_dtype)
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from bot_logic import get_bot_response as nlu_get_bot_response

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

SUGGESTION_BATCH_SIZE = 3 
PRE_VALIDATION_SAMPLE_SIZE = 30 

# --- Helper Functions ---
def allowed_file(filename: str) -> bool:
    """Checks if the filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Attempt to clean a pandas Series to be numeric (handles $, ,, % )."""
    if series is None:
        return series
    if is_numeric_dtype(series.dtype):
        return series
    if series.dtype == 'object' or is_string_dtype(series.dtype):
        try:
            s_cleaned = series.astype(str).str.replace(r'[$,%]', '', regex=True).str.strip()
            s_cleaned = s_cleaned.replace('', pd.NA)
            s_numeric = pd.to_numeric(s_cleaned, errors='coerce')
            if s_numeric.notna().any():
                return s_numeric
        except Exception as e:
            print(f"Numeric cleaning failed for column '{series.name}': {e}")
            pass
    return series

def get_simplified_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Analyzes DataFrame columns and returns simplified types.
    Refined logic for integer columns and ID-like numeric columns.
    """
    simplified_types = {}
    if df is None or df.empty:
        return simplified_types
    
    non_numeric_object_keywords = [
        'id','code','name','person','country','category','product','type',
        'status','gender','region','city','state','text','desc','comment',
        'notes','message','address','url','path','file', 'postcode', 'zip', 'sku', 'identifier', 'key'
    ]
    date_like_keywords = ['date', 'time', 'yr', 'year', 'month', 'day', 'timestamp']
    rating_like_keywords = ['rating', 'level', 'quality', 'score', 'grade', 'tier'] 
    id_like_numeric_keywords = ['id', 'identifier', 'key', 'number', 'no', 'personid'] 

    for col_name in df.columns:
        series = df[col_name]
        original_dtype = series.dtype
        current_series_for_analysis = series.copy() 

        try:
            is_datetime_col = False
            # 1. Attempt to convert likely date columns if they are objects
            if (current_series_for_analysis.dtype == 'object' or is_string_dtype(current_series_for_analysis.dtype)) and \
               any(substr in col_name.lower() for substr in date_like_keywords) and \
               not is_datetime64_any_dtype(current_series_for_analysis.dtype):
                try:
                    sample_to_test = current_series_for_analysis.dropna().iloc[:min(10, len(current_series_for_analysis.dropna()))]
                    if not sample_to_test.empty:
                        pd.to_datetime(sample_to_test, errors='raise') 
                        converted_full = pd.to_datetime(current_series_for_analysis, errors='coerce')
                        if converted_full.notna().sum() / max(1, current_series_for_analysis.count()) > 0.7:
                             current_series_for_analysis = converted_full
                             is_datetime_col = True
                except Exception:
                    pass 
            
            # 2. Attempt numeric cleaning for 'object' types NOT in the exclusion list AND not already datetime
            if not is_datetime_col and current_series_for_analysis.dtype == 'object' and \
               not any(substr in col_name.lower() for substr in non_numeric_object_keywords + date_like_keywords):
                 cleaned_series = clean_numeric_column(current_series_for_analysis)
                 if is_numeric_dtype(cleaned_series.dtype):
                     current_series_for_analysis = cleaned_series
            
            final_dtype = current_series_for_analysis.dtype
            unique_count = current_series_for_analysis.nunique(dropna=True)
            non_null_count = current_series_for_analysis.count()

            if non_null_count == 0:
                simplified_types[col_name] = 'empty'
            elif is_numeric_dtype(final_dtype):
                is_id_name_check = any(keyword in col_name.lower() for keyword in id_like_numeric_keywords)
                if is_id_name_check and unique_count >= max(1, non_null_count * 0.90) and unique_count > 20 : 
                    simplified_types[col_name] = 'id_like_text'
                else:
                    is_effectively_integer = False
                    if current_series_for_analysis.notna().any(): 
                        try:
                            is_effectively_integer = (current_series_for_analysis.dropna().astype(float) % 1 == 0).all()
                        except ValueError: 
                            is_effectively_integer = pd.api.types.is_integer_dtype(current_series_for_analysis.dropna().dtype)
                    
                    is_rating_like_name = any(keyword in col_name.lower() for keyword in rating_like_keywords)
                    
                    if is_effectively_integer:
                        if is_rating_like_name and unique_count < 25: 
                            simplified_types[col_name] = 'categorical_numeric'
                        elif not is_rating_like_name and unique_count <= 5: 
                            simplified_types[col_name] = 'categorical_numeric'
                        else:
                            simplified_types[col_name] = 'numerical'
                    else: 
                        simplified_types[col_name] = 'numerical'
            
            elif is_datetime64_any_dtype(final_dtype):
                simplified_types[col_name] = 'datetime'
            elif is_string_dtype(final_dtype) or final_dtype == 'object':
                 is_common_cat_name = any(name_part in col_name.lower() for name_part in ['country','category','product','type','status','gender','region','city','state'])
                 if unique_count <= 1: simplified_types[col_name] = 'categorical'
                 elif is_common_cat_name and unique_count < 750 : simplified_types[col_name] = 'categorical'
                 elif unique_count < max(10, non_null_count * 0.6) and unique_count < 500: simplified_types[col_name] = 'categorical'
                 else: simplified_types[col_name] = 'id_like_text'
            else:
                simplified_types[col_name] = 'other'
        except Exception as e:
            print(f"Warning: Type check failed for column '{col_name}' (Original dtype: {original_dtype}): {e}")
            simplified_types[col_name] = 'other'
    return simplified_types

def suggest_charts_based_on_answers(user_answers: Dict[str, str], df_sample_original: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    if df_sample_original is None or df_sample_original.empty:
        return [{"name": "Cannot suggest: Data sample missing or unreadable.", "type": "Info", "score": 0}]
    
    df_sample = df_sample_original.copy()
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object' and not any(substr in col.lower() for substr in ['id','code','name','person','country','category','product','type','status','gender','region','city','state','date','time']):
            df_sample[col] = clean_numeric_column(df_sample[col])
        if any(substr in col.lower() for substr in ['date','time','yr','year']) and not is_datetime64_any_dtype(df_sample[col]):
            try:
                converted_date = pd.to_datetime(df_sample[col], errors='coerce')
                if converted_date.notna().any():
                    df_sample[col] = converted_date
            except Exception:
                pass

    col_types = get_simplified_column_types(df_sample)
    numerical_cols = [c for c,t in col_types.items() if t=='numerical']
    categorical_cols = [c for c,t in col_types.items() if t in ['categorical', 'categorical_numeric']]
    distributable_numeric_cols = [c for c,t in col_types.items() if t in ['numerical', 'categorical_numeric']]
    datetime_cols = [c for c,t in col_types.items() if t=='datetime']
    
    ua_count = user_answers.get('variable_count','').lower()
    ua_types = user_answers.get('variable_types','').lower()
    ua_msg = user_answers.get('message_insight','').lower()
    ua_emphasis = user_answers.get('emphasis', '').lower()
    ua_cols_interest_str = user_answers.get('columns_of_interest', '').lower()
    
    cols_of_interest = []
    if ua_cols_interest_str and ua_cols_interest_str not in ['any', 'none', 'skip', '']:
        cols_of_interest = [col.strip() for col in ua_cols_interest_str.split(',') if col.strip()]

    base_score_direct_match = 100; base_score_good_match = 80; base_score_general_match = 60; base_score_multivariate = 50
    emphasis_bonus_strong = 50; emphasis_bonus_moderate = 25; column_interest_bonus = 40
    penalty_id_like = -100; penalty_pie_too_many_cats = -20; penalty_bar_too_many_cats = -10 

    if ("one" in ua_count or "1" in ua_count) and ("dist" in ua_msg or "spread" in ua_msg or "summ" in ua_msg) and ("num" in ua_types or "cat" in ua_types or "any" in ua_types or not ua_types) and distributable_numeric_cols:
        for col in distributable_numeric_cols:
            if col_types[col] == 'id_like_text': continue
            score = base_score_direct_match if col_types[col] == 'numerical' else base_score_good_match
            if "distri" in ua_emphasis or "spread" in ua_emphasis: score += emphasis_bonus_strong
            if col.lower() in cols_of_interest: score += column_interest_bonus
            suggestions.append({"name": "Histogram", "for_col": col, "type": "Univariate Numerical", "reason": f"Distribution of '{col}'.", "required_cols_specific": [col], "score": score})
            suggestions.append({"name": "Box Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Summary of '{col}'.", "required_cols_specific": [col], "score": score})
            suggestions.append({"name": "Density Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Smooth distribution of '{col}'.", "required_cols_specific": [col], "score": score - 5})
    if ("one" in ua_count or "1" in ua_count) and ("prop" in ua_msg or "share" in ua_msg or "freq" in ua_msg or "count" in ua_msg or "val" in ua_msg) and ("cat" in ua_types or "any" in ua_types or not ua_types) and categorical_cols:
        for col in categorical_cols:
            if col_types[col] == 'id_like_text': continue
            score = base_score_direct_match; nunique = df_sample[col].nunique(dropna=True) if col in df_sample else 0
            current_bar_score = score
            if "exact val" in ua_emphasis or "magnitudes" in ua_emphasis: current_bar_score += emphasis_bonus_moderate
            if "comparison between groups" in ua_emphasis: current_bar_score += emphasis_bonus_moderate
            if col.lower() in cols_of_interest: current_bar_score += column_interest_bonus
            current_bar_score -= (max(0, (nunique - 10) // 5) * abs(penalty_bar_too_many_cats))
            suggestions.append({"name": "Bar Chart (Counts)", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows counts for categories in '{col}'.", "required_cols_specific": [col], "score": current_bar_score})
            if 1 < nunique < 8:
                current_pie_score = score
                if "prop" in ua_emphasis or "parts of a whole" in ua_emphasis: current_pie_score += emphasis_bonus_strong
                if col.lower() in cols_of_interest: current_pie_score += column_interest_bonus
                suggestions.append({"name": "Pie Chart", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows proportions for '{col}'.", "required_cols_specific": [col], "score": current_pie_score})
            elif nunique >= 8:
                 current_pie_score = score + (max(0, nunique - 5) * penalty_pie_too_many_cats)
                 if "prop" in ua_emphasis or "parts of a whole" in ua_emphasis: current_pie_score += emphasis_bonus_moderate
                 if col.lower() in cols_of_interest: current_pie_score += column_interest_bonus
                 suggestions.append({"name": "Pie Chart", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows proportions for '{col}'.", "required_cols_specific": [col], "score": current_pie_score})
    if ("two" in ua_count or "2" in ua_count) and ("relat" in ua_msg or "corr" in ua_msg or "scat" in ua_msg) and ("num" in ua_types or "any" in ua_types or not ua_types) and len(numerical_cols)>=2:
        for i,c1 in enumerate(numerical_cols):
            for j,c2 in enumerate(numerical_cols):
                 if j > i:
                    score = base_score_direct_match
                    if "relat" in ua_emphasis or "corr" in ua_emphasis: score += emphasis_bonus_strong
                    if col_types[c1] == 'id_like_text' or col_types[c2] == 'id_like_text': score += penalty_id_like
                    if c1.lower() in cols_of_interest or c2.lower() in cols_of_interest: score += column_interest_bonus
                    suggestions.append({"name": "Scatter Plot", "for_cols": f"{c1} & {c2}", "type": "Bivariate Numerical-Numerical", "reason": f"Relationship: '{c1}' vs '{c2}'. If points overlap, try customizing transparency.", "required_cols_specific": [c1, c2], "score": score})
                    if categorical_cols:
                        for hue_col in categorical_cols[:1]: 
                             if hue_col not in [c1, c2] and col_types[hue_col] not in ['id_like_text', 'empty']:
                                hue_scatter_score = score - 15 
                                if "comparison between groups" in ua_emphasis: hue_scatter_score += emphasis_bonus_moderate
                                if hue_col.lower() in cols_of_interest or c1.lower() in cols_of_interest or c2.lower() in cols_of_interest: hue_scatter_score += column_interest_bonus
                                suggestions.append({"name": "Scatter Plot", "for_cols": f"{c1} & {c2} by {hue_col}", "type": "Multivariate Scatter", "reason": f"Relationship: '{c1}' vs '{c2}', colored by '{hue_col}'. Adjust transparency for density.", "required_cols_specific": [c1, c2, hue_col], "score": hue_scatter_score})
    if ("two" in ua_count or "2" in ua_count) and ("comp" in ua_msg or "across" in ua_msg or "group" in ua_msg or "dist" in ua_msg) and ("mix" in ua_types or "cat" in ua_types or "num" in ua_types or "any" in ua_types or not ua_types) and distributable_numeric_cols and categorical_cols:
        for num_col in distributable_numeric_cols:
            for cat_col in categorical_cols:
                if num_col != cat_col and col_types[cat_col] not in ['id_like_text', 'empty'] and col_types[num_col] not in ['id_like_text', 'empty']:
                    score = base_score_good_match
                    if "comp" in ua_emphasis or "groups" in ua_emphasis: score += emphasis_bonus_strong
                    if "distri" in ua_emphasis or "spread" in ua_emphasis: score += emphasis_bonus_moderate
                    if num_col.lower() in cols_of_interest or cat_col.lower() in cols_of_interest: score += column_interest_bonus
                    suggestions.append({"name": "Box Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Distribution of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col], "score": score})
                    suggestions.append({"name": "Violin Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Density/distribution of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col], "score": score - 5})
        if numerical_cols and categorical_cols:
            for num_col in numerical_cols:
                for cat_col in categorical_cols:
                    if num_col != cat_col and col_types[cat_col] not in ['id_like_text', 'empty']:
                        agg_bar_score = base_score_good_match - 10
                        if "exact val" in ua_emphasis or "magnitudes" in ua_emphasis: agg_bar_score += emphasis_bonus_moderate
                        if "comp" in ua_emphasis or "groups" in ua_emphasis: agg_bar_score += emphasis_bonus_moderate
                        if num_col.lower() in cols_of_interest or cat_col.lower() in cols_of_interest: agg_bar_score += column_interest_bonus
                        suggestions.append({"name": "Bar Chart (Aggregated)", "for_cols": f"Avg of {num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Average (or sum/median) of '{num_col}' for each category in '{cat_col}'.", "required_cols_specific": [cat_col, num_col], "score": agg_bar_score})
    if ("two" in ua_count or "2" in ua_count) and ("relat" in ua_msg or "comp" in ua_msg or "cont" in ua_msg or "joint" in ua_msg) and ("cat" in ua_types or "any" in ua_types or not ua_types) and len(categorical_cols)>=2:
         for i,c1 in enumerate(categorical_cols):
              for j,c2 in enumerate(categorical_cols):
                  if j > i and col_types[c1] not in ['id_like_text', 'empty'] and col_types[c2] not in ['id_like_text', 'empty']:
                      score = base_score_good_match
                      if "comp" in ua_emphasis or "groups" in ua_emphasis: score += emphasis_bonus_moderate
                      if "relat" in ua_emphasis: score += emphasis_bonus_moderate
                      if c1.lower() in cols_of_interest or c2.lower() in cols_of_interest: score += column_interest_bonus
                      suggestions.append({"name": "Grouped Bar Chart", "for_cols": f"{c1} & {c2}", "type": "Bivariate Categorical-Categorical", "reason": f"Counts of '{c1}' grouped by '{c2}'.", "required_cols_specific": [c1, c2], "score": score})
                      suggestions.append({"name": "Heatmap (Counts)", "for_cols": f"{c1} & {c2}", "type": "Bivariate Categorical-Categorical", "reason": f"Co-occurrence frequency of '{c1}' & '{c2}'.", "required_cols_specific": [c1, c2], "score": score - 10})
    if (("two" in ua_count or "2" in ua_count) or "time" in ua_types or "trend" in ua_msg) and datetime_cols and numerical_cols:
        for dt_col in datetime_cols:
             for num_col in numerical_cols:
                  if col_types[num_col] not in ['id_like_text', 'empty']:
                    score = base_score_direct_match if "trend" in ua_msg else base_score_good_match
                    if "time" in ua_emphasis or "sequence" in ua_emphasis: score += emphasis_bonus_strong
                    if dt_col.lower() in cols_of_interest or num_col.lower() in cols_of_interest: score += column_interest_bonus
                    suggestions.append({"name": "Line Chart", "for_cols": f"{num_col} over {dt_col}", "type": "Time Series", "reason": f"Trend of '{num_col}' over '{dt_col}'.", "required_cols_specific": [dt_col, num_col], "score": score})
                    suggestions.append({"name": "Area Chart", "for_cols": f"{num_col} over {dt_col}", "type": "Time Series", "reason": f"Cumulative trend of '{num_col}' over '{dt_col}'.", "required_cols_specific": [dt_col, num_col], "score": score - 10})
    if ("more" in ua_count or "mult" in ua_count or "pair" in ua_msg or "heat" in ua_msg or "para" in ua_msg) or \
       ((ua_count not in ["one","1","two","2"]) and (len(numerical_cols)>2 or len(categorical_cols)>2)):
        if len(numerical_cols)>=3:
            score = base_score_multivariate
            if "relat" in ua_emphasis or "corr" in ua_emphasis: score += emphasis_bonus_moderate
            if any(nc.lower() in cols_of_interest for nc in numerical_cols[:min(4,len(numerical_cols))]): score += column_interest_bonus / 2 
            suggestions.append({"name": "Pair Plot", "type": "Multivariate", "reason": "Pairwise relationships (numerical).", "required_cols_specific": numerical_cols[:min(4,len(numerical_cols))], "score": score})
            suggestions.append({"name": "Correlation Heatmap", "type": "Multivariate", "reason": "Correlation matrix (numerical).", "required_cols_specific": numerical_cols, "score": score})
            suggestions.append({"name": "Parallel Coordinates Plot", "type": "Multivariate", "reason": "Compare multiple numerical variables.", "required_cols_specific": numerical_cols[:min(6,len(numerical_cols))], "score": score - 5})
    
    final_suggestions_dict = {}
    for s in suggestions:
        if not isinstance(s, dict): print(f"Warning: Skipping non-dict item during suggestion processing: {s}"); continue
        req_cols = s.get("required_cols_specific", []); s_key_cols_str = "_".join(sorted(req_cols)) if isinstance(req_cols, list) else ""
        s_key = f"{s.get('name', 'UnknownChart')}_{s_key_cols_str}"
        if s_key not in final_suggestions_dict or s.get('score', 0) > final_suggestions_dict[s_key].get('score', 0):
            final_suggestions_dict[s_key] = s
            
    final_suggestions = sorted(list(final_suggestions_dict.values()), key=lambda x: x.get('score', 0), reverse=True)
    
    if not final_suggestions and not any(s.get("type") == "Info" for s in final_suggestions):
        final_suggestions.append({"name": "No specific chart matched well", "type": "Info", "reason": "Your criteria didn't closely match common chart types. You can try picking columns manually or rephrasing your goal.", "required_cols_specific": [], "score": 0})
    if not any(s['name']=="Pick columns manually" for s in final_suggestions):
        final_suggestions.append({"name": "Pick columns manually", "type": "Action", "reason": "Choose columns yourself.", "required_cols_specific": [], "score": -100}) 
    
    return final_suggestions

# --- (Validation Function - keep as is) ---
def validate_columns_for_chart(chart_type: str, columns: List[str], df_cleaned_subset: pd.DataFrame) -> Optional[str]:
    if not columns: return "No columns selected."
    missing = [col for col in columns if col not in df_cleaned_subset.columns]
    if missing: return f"Column(s) not found: {', '.join(missing)}. Check spelling?"
    col_types = get_simplified_column_types(df_cleaned_subset)
    num_numerical = sum(1 for c in columns if col_types.get(c) == 'numerical')
    num_categorical = sum(1 for c in columns if col_types.get(c) in ['categorical', 'categorical_numeric'])
    num_distributable = sum(1 for c in columns if col_types.get(c) in ['numerical', 'categorical_numeric'])
    num_datetime = sum(1 for c in columns if col_types.get(c) == 'datetime')
    num_id_like = sum(1 for c in columns if col_types.get(c) == 'id_like_text')
    num_selected = len(columns)
    col_details_list = []
    for c in columns:
        inferred_t = col_types.get(c, 'unknown'); friendly_t = inferred_t
        if inferred_t == 'id_like_text': friendly_t = "text (many unique values)"
        elif inferred_t == 'categorical_numeric': friendly_t = "numeric (as category)"
        elif inferred_t == 'empty': friendly_t = 'empty/all missing'
        col_details_list.append(f"'{c}' (as {friendly_t})")
    col_details = "; ".join(col_details_list)
    requirements = {"Histogram":{'exact_cols':1,'distributable_numeric':1},"Box Plot":{'exact_cols':1,'distributable_numeric':1},"Density Plot":{'exact_cols':1,'distributable_numeric':1},"Bar Chart (Counts)":{'exact_cols':1,'categorical':1},"Pie Chart":{'exact_cols':1,'categorical':1},"Scatter Plot":{'min_cols':2, 'max_cols': 3, 'numerical':2, 'optional_categorical_for_hue':1},"Line Chart":{'exact_cols':2,'numerical':(1,2)},"Box Plots (by Category)":{'exact_cols':2,'categorical':1,'distributable_numeric':1},"Violin Plots (by Category)":{'exact_cols':2,'categorical':1,'distributable_numeric':1},"Bar Chart (Aggregated)":{'exact_cols':2,'categorical':1,'numerical':1},"Grouped Bar Chart":{'exact_cols':2,'categorical':2},"Heatmap (Counts)":{'exact_cols':2,'categorical':2},"Area Chart":{'exact_cols':2,'numerical':(1,2)},"Pair Plot":{'min_cols':3,'numerical':3},"Correlation Heatmap":{'min_cols':2,'numerical':2},"Parallel Coordinates Plot":{'min_cols':3,'numerical':3}}
    req = {}
    if chart_type == "Bar Chart":
         if num_selected == 1: req = requirements.get("Bar Chart (Counts)",{}); req['exact_cols']=1
         elif num_selected == 2:
             col1_actual_type = col_types.get(columns[0]); col2_actual_type = col_types.get(columns[1]) if len(columns) > 1 else None
             col1_is_cat = col1_actual_type in ['categorical', 'categorical_numeric']
             col2_is_cat = col2_actual_type in ['categorical', 'categorical_numeric']
             col1_is_num = col1_actual_type == 'numerical'
             col2_is_num = col2_actual_type == 'numerical'
             if col1_is_cat and col2_is_cat: req = requirements.get("Grouped Bar Chart",{}); req['exact_cols']=2
             elif (col1_is_cat and col2_is_num) or (col1_is_num and col2_is_cat) : req = requirements.get("Bar Chart (Aggregated)",{}); req['exact_cols']=2
             else: return f"A Bar Chart with 2 columns needs either two categorical columns (like 'Country', 'Product') or one categorical and one numerical (like 'Country', 'Amount'). You selected ({col_details})."
         else: return "A Bar Chart needs 1 or 2 columns."
    elif chart_type in requirements: req = requirements[chart_type]
    else: print(f"Info: Validation skipped for unknown chart type '{chart_type}'."); return None
    if 'exact_cols' in req and num_selected != req['exact_cols']: return f"{chart_type} needs exactly {req['exact_cols']} column(s), but you selected {num_selected} ({col_details})."
    if 'min_cols' in req and num_selected < req['min_cols']: return f"{chart_type} needs at least {req['min_cols']} columns, you selected {num_selected} ({col_details})."
    if 'max_cols' in req and num_selected > req['max_cols']: return f"{chart_type} uses at most {req['max_cols']} columns, you selected {num_selected} ({col_details})."
    err_msg_parts = []
    type_error_found = False
    target_num = req.get('numerical'); target_cat = req.get('categorical'); target_dist = req.get('distributable_numeric'); target_dt = req.get('datetime'); opt_cat_hue = req.get('optional_categorical_for_hue', 0)

    if chart_type == "Scatter Plot":
        num_numerical_selected = sum(1 for c in columns if col_types.get(c) == 'numerical')
        num_categorical_selected = sum(1 for c in columns if col_types.get(c) in ['categorical', 'categorical_numeric'])
        if not ( (num_selected == 2 and num_numerical_selected == 2) or \
                 (num_selected == 3 and num_numerical_selected == 2 and num_categorical_selected == 1 and opt_cat_hue == 1) ):
            type_error_found = True
            if num_numerical_selected < 2: err_msg_parts.append(f"2 numerical columns (found {num_numerical_selected})")
            if num_selected == 3 and num_categorical_selected < 1: err_msg_parts.append(f"1 categorical column for color (found {num_categorical_selected})")
            if not err_msg_parts: err_msg_parts.append("an incorrect number or combination of column types")
    else:
        if target_num is not None:
            num_needed = target_num[0] if isinstance(target_num, tuple) else target_num; num_range_str = f"{target_num[0]}-{target_num[1]}" if isinstance(target_num, tuple) else str(target_num)
            if isinstance(target_num, int) and num_numerical < target_num: err_msg_parts.append(f"{target_num} numerical column(s) (found {num_numerical})"); type_error_found=True
            elif isinstance(target_num, tuple) and not (target_num[0] <= num_numerical <= target_num[1]): err_msg_parts.append(f"{num_range_str} numerical column(s) (found {num_numerical})"); type_error_found=True
        if target_cat is not None and isinstance(target_cat, int) and num_categorical < target_cat: err_msg_parts.append(f"{target_cat} categorical column(s) (found {num_categorical})"); type_error_found=True
        if target_dist is not None and isinstance(target_dist, int) and num_distributable < target_dist: err_msg_parts.append(f"{target_dist} numerical or rating-like column(s) (found {num_distributable})"); type_error_found=True
        if target_dt is not None and isinstance(target_dt, int) and num_datetime < target_dt: err_msg_parts.append(f"{target_dt} datetime column(s) (found {num_datetime})"); type_error_found=True
    
    if type_error_found: return f"{chart_type} needs: {', '.join(err_msg_parts)}. You provided ({col_details}). For example, 'Amount' or 'Boxes Shipped' are numerical, while 'Country' or 'Product' are categorical."
    if chart_type == "Pie Chart":
        if columns and columns[0] in df_cleaned_subset.columns:
            try:
                nunique = df_cleaned_subset[columns[0]].nunique(dropna=True)
                if nunique > 10:
                     return f"column '{columns[0]}' (as {col_types.get(columns[0])}) has {nunique} categories, which is too many for a clear Pie Chart. Try a Bar Chart."
            except Exception as e:
                print(f"Warn: Could not check nunique for Pie Chart validation: {e}")
    if num_id_like > 0 and req.get('categorical', 0) > 0 and num_categorical < req.get('categorical', 0):
         id_cols = [c for c in columns if col_types.get(c) == 'id_like_text']
         if id_cols : return f"Column '{id_cols[0]}' (inferred as {col_types.get(id_cols[0])} with too many unique values) isn't suitable for {chart_type} which needs distinct groups. Try a column like 'Country' or 'Product'."
    return None

# --- Plotting Function ---
def generate_plot_and_get_uri(filepath, chart_type, columns,
                              custom_title=None, custom_xlabel=None, custom_ylabel=None,
                              custom_marker=None, custom_alpha=None, custom_hue_marker_map=None):
    """Generates plot and returns base64 URI or (None, error_msg)."""
    if not filepath: return None, "File path missing."
    try:
        df_full = pd.read_csv(filepath, low_memory=False) if filepath.endswith(".csv") else pd.read_excel(filepath)
        if not all(col in df_full.columns for col in columns): missing_cols = [c for c in columns if c not in df_full.columns]; return None, f"Column(s) not found: {', '.join(missing_cols)}."
        df_processed_subset = df_full[columns].copy(); plot_columns = list(columns);
        for col_name in plot_columns:
            if col_name in df_processed_subset.columns:
                if df_processed_subset[col_name].dtype == 'object' and not any(substr in col_name.lower() for substr in ['id','code','name','person','country','category','product','type','status','gender','region','city','state','date','time','desc','notes','comment']):
                    df_processed_subset[col_name] = clean_numeric_column(df_processed_subset[col_name])
                if any(substr in col_name.lower() for substr in ['date','time','yr','year']) and \
                   not is_datetime64_any_dtype(df_processed_subset[col_name]):
                      try:
                          converted_date = pd.to_datetime(df_processed_subset[col_name], errors='coerce')
                          if converted_date.notna().any(): df_processed_subset[col_name] = converted_date
                      except Exception as e: print(f"Note: Failed datetime conversion for '{col_name}' during plot prep: {e}")
        validation_error = validate_columns_for_chart(chart_type, plot_columns, df_processed_subset)
        if validation_error: return None, f"{validation_error}"
        df_plot = df_processed_subset.copy()
        df_plot.fillna(method='ffill', inplace=True); df_plot.fillna(method='bfill', inplace=True)
    except Exception as e: print(f"Error reading/cleaning/validating dataframe ('{filepath}'): {e}"); return None, f"Error preparing data: {str(e)[:100]}"

    img = io.BytesIO(); plt.figure(figsize=(7.5, 5)); plt.style.use('seaborn-v0_8-whitegrid'); original_chart_type = chart_type; plot_title_detail = ""; mapped_chart_type = chart_type;
    default_xlabel = None; default_ylabel = None; default_plot_title = ""
    try:
        col_types_final = get_simplified_column_types(df_plot)
        if original_chart_type == "Bar Chart":
             if len(plot_columns)==1 and plot_columns[0] in col_types_final and col_types_final[plot_columns[0]] in ['categorical','categorical_numeric']: mapped_chart_type="Bar Chart (Counts)"; plot_title_detail=f" for {plot_columns[0]}"
             elif len(plot_columns)==2:
                  cat_cols=[c for c in plot_columns if col_types_final.get(c) in ['categorical','categorical_numeric']]; num_cols=[c for c in plot_columns if col_types_final.get(c)=='numerical']
                  if len(cat_cols)==1 and len(num_cols)==1: mapped_chart_type="Bar Chart (Aggregated)"; plot_columns=[cat_cols[0],num_cols[0]]; plot_title_detail=f" of {num_cols[0]} by {cat_cols[0]}"
                  elif len(cat_cols)==2: mapped_chart_type="Grouped Bar Chart"; plot_title_detail=f" for {plot_columns[0]} by {plot_columns[1]}"
                  else: raise ValueError(f"Cannot determine Bar Chart type for ({', '.join(plot_columns)}).")
             else: raise ValueError("Bar Chart needs 1 or 2 columns.")
        default_plot_title = f"{mapped_chart_type}{plot_title_detail}"
        col1 = plot_columns[0] if len(plot_columns) > 0 else None
        col2 = plot_columns[1] if len(plot_columns) > 1 else None
        hue_col = plot_columns[2] if len(plot_columns) > 2 and mapped_chart_type == "Scatter Plot" else None

        default_xlabel = col1 if col1 else None
        default_ylabel = col2 if mapped_chart_type in ["Scatter Plot"] and col2 else None

        if mapped_chart_type=="Histogram": sns.histplot(data=df_plot, x=col1, kde=True); default_ylabel="Frequency"
        elif mapped_chart_type=="Box Plot": sns.boxplot(data=df_plot, y=col1); default_ylabel=col1; default_xlabel=None
        elif mapped_chart_type=="Density Plot": sns.kdeplot(data=df_plot, x=col1, fill=True); default_ylabel="Density"
        elif mapped_chart_type=="Bar Chart (Counts)": counts=df_plot[col1].value_counts().nlargest(20); sns.barplot(x=counts.index.astype(str), y=counts.values); default_ylabel="Count"; plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type == "Pie Chart":
            counts = df_plot[col1].value_counts(); effective_counts = counts.nlargest(7)
            if len(counts) > 7: effective_counts.loc['Other'] = counts.iloc[7:].sum()
            plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85); default_plot_title = f"Pie Chart of {col1}"; plt.axis('equal')
            default_xlabel=None; default_ylabel=None
        elif mapped_chart_type=="Scatter Plot":
            marker_to_use_overall = custom_marker if custom_marker else 'o'
            alpha_to_use = custom_alpha if custom_alpha is not None else 0.7
            
            plot_kwargs = {"data": df_plot, "x": col1, "y": col2, "alpha": alpha_to_use}
            
            if hue_col:
                plot_kwargs["hue"] = hue_col
                default_plot_title = f"Scatter: {col1} vs {col2} by {hue_col}"
                if custom_hue_marker_map:
                    complete_hue_marker_map = {}
                    unique_hue_values_in_data = df_plot[hue_col].dropna().unique()
                    for val in unique_hue_values_in_data:
                        complete_hue_marker_map[val] = custom_hue_marker_map.get(val, 'o') 
                    plot_kwargs["style"] = hue_col 
                    plot_kwargs["markers"] = complete_hue_marker_map
                else: 
                    plot_kwargs["marker"] = marker_to_use_overall 
            else: 
                plot_kwargs["marker"] = marker_to_use_overall
                default_plot_title = f"Scatter: {col1} vs {col2}"
            
            sns.scatterplot(**plot_kwargs)
            default_ylabel=col2
        elif mapped_chart_type=="Line Chart":
            df_to_plot_lc=df_plot.copy(); sort_col_lc=col1
            try:
                 if is_datetime64_any_dtype(df_to_plot_lc[sort_col_lc]): df_to_plot_lc=df_to_plot_lc.sort_values(by=sort_col_lc)
                 elif is_numeric_dtype(df_to_plot_lc[sort_col_lc]): df_to_plot_lc=df_to_plot_lc.sort_values(by=sort_col_lc)
            except Exception as sort_e: print(f"Note: Could not sort for Line Chart: {sort_e}")
            sns.lineplot(data=df_to_plot_lc, x=col1, y=col2); default_ylabel=col2; plt.xticks(rotation=45, ha='right', fontsize=9)
        elif mapped_chart_type=="Box Plots (by Category)": sns.boxplot(data=df_plot, x=col1, y=col2); default_ylabel=col2; plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Violin Plots (by Category)": sns.violinplot(data=df_plot, x=col1, y=col2); default_ylabel=col2; plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Bar Chart (Aggregated)": cat_col,num_col=plot_columns[0],plot_columns[1]; agg_data=df_plot.groupby(cat_col)[num_col].mean().nlargest(20); sns.barplot(x=agg_data.index.astype(str), y=agg_data.values); default_xlabel=cat_col; default_ylabel=f"Mean of {num_col}"; plt.xticks(rotation=65, ha='right', fontsize=9)
        elif mapped_chart_type=="Grouped Bar Chart": col1_tc=df_plot[plot_columns[0]].value_counts().nlargest(10).index; col2_tc=df_plot[plot_columns[1]].value_counts().nlargest(5).index; df_f=df_plot[df_plot[plot_columns[0]].isin(col1_tc) & df_plot[plot_columns[1]].isin(col2_tc)]; sns.countplot(data=df_f, x=plot_columns[0], hue=plot_columns[1]); default_ylabel="Count"; plt.xticks(rotation=65, ha='right', fontsize=9); plt.legend(title=plot_columns[1], fontsize='x-small', title_fontsize='small', bbox_to_anchor=(1.02,1), loc='upper left')
        else: raise NotImplementedError(f"Plot type '{mapped_chart_type}' is not explicitly implemented.")

        plt.title(custom_title if custom_title is not None else default_plot_title, fontsize=12)
        if custom_xlabel is not None: plt.xlabel(custom_xlabel)
        elif default_xlabel: plt.xlabel(default_xlabel)
        if custom_ylabel is not None: plt.ylabel(custom_ylabel)
        elif default_ylabel: plt.ylabel(default_ylabel)
        
        plt.tight_layout(pad=1.0); plt.savefig(img, format='png', bbox_inches='tight'); plt.close(); img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8'); print(f"Success: {original_chart_type} (as {mapped_chart_type})"); return f"data:image/png;base64,{plot_url}", None
    except Exception as e:
        error_info = f"{type(e).__name__}: {str(e)}"; print(f"!!! Error during plot generation execution for '{mapped_chart_type or original_chart_type}' with {plot_columns}: {error_info}"); error_message = f"Failed to generate {original_chart_type}. ({error_info[:100]}...).";
        if 'plt' in locals() and plt.get_fignums(): plt.close('all')
        return None, error_message

# --- (get_descriptive_stats_html - keep as is) ---
def get_descriptive_stats_html(df_original: pd.DataFrame) -> str:
    if df_original is None or df_original.empty: return "<p>No data available to summarize.</p>"
    df_for_stats = df_original.copy()
    for col in df_for_stats.columns:
        if df_for_stats[col].dtype == 'object' and not any(substr in col.lower() for substr in ['id','code','name','person','country','category','product','type','status','gender','region','city','state','date','time']):
            df_for_stats[col] = clean_numeric_column(df_for_stats[col])
        if any(substr in col.lower() for substr in ['date','time','yr','year']) and not is_datetime64_any_dtype(df_for_stats[col]):
            try: # Multi-line
                converted_date = pd.to_datetime(df_for_stats[col], errors='coerce')
                if converted_date.notna().any():
                    df_for_stats[col] = converted_date
            except:
                pass
    inferred_types = get_simplified_column_types(df_for_stats)
    html_output = "<h4>Data Column Summary:</h4><table class='preview-table'><thead><tr><th>Column Name</th><th>Inferred Type</th><th>Summary / Top Values</th></tr></thead><tbody>"
    if not df_for_stats.columns.tolist(): return "<p>No columns found to summarize.</p>"
    for col in df_for_stats.columns:
        col_type = inferred_types.get(col, 'unknown'); summary_stats_str = "Could not generate summary."
        try:
            if col_type == 'numerical': desc = df_for_stats[col].describe(); summary_stats_str = f"Count: {desc.get('count', 0):.0f}, Mean: {desc.get('mean', float('nan')):.2f}, Std: {desc.get('std', float('nan')):.2f}<br>Min: {desc.get('min', float('nan')):.2f}, Median: {desc.get('50%', float('nan')):.2f}, Max: {desc.get('max', float('nan')):.2f}"
            elif col_type == 'categorical_numeric': desc = df_for_stats[col].describe(); counts = df_for_stats[col].value_counts().nlargest(3); top_vals = ", ".join([f"{str(idx)} ({val})" for idx, val in counts.items()]); summary_stats_str = f"Count: {desc.get('count', 0):.0f}, Unique: {df_for_stats[col].nunique()}<br>Min: {desc.get('min', float('nan')):.0f}, Max: {desc.get('max', float('nan')):.0f}<br>Top: {top_vals}"
            elif col_type == 'categorical':
                unique_count = df_for_stats[col].nunique(); summary_stats_str = f"Unique Values: {unique_count}<br>"; counts = df_for_stats[col].value_counts().nlargest(3); top_vals = ", ".join([f"{str(idx)} ({val})" for idx, val in counts.items()]); summary_stats_str += f"Top: {top_vals}";
                if unique_count > 3: summary_stats_str += "..."
            elif col_type == 'datetime':
                datetime_series = pd.to_datetime(df_for_stats[col], errors='coerce'); min_date = datetime_series.min(); max_date = datetime_series.max(); summary_stats_str = f"Date Range:<br>{min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'N/A'} to <br>{max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A'}"
            elif col_type == 'id_like_text': summary_stats_str = f"Unique Values: {df_for_stats[col].nunique()} (potential ID/text)."
            elif col_type == 'empty': summary_stats_str = "All values are missing."
            else: summary_stats_str = f"Raw Data Type: {df_for_stats[col].dtype}, Unique: {df_for_stats[col].nunique()}"
        except Exception as e: summary_stats_str = f"Error summarizing: {str(e)[:50]}..."; print(f"Error summarizing column {col}: {e}")
        html_output += f"<tr><td><strong>{col}</strong></td><td><em>{col_type}</em></td><td>{summary_stats_str}</td></tr>"
    html_output += "</tbody></table>"; return html_output

# --- (Helper for Suggestion Display - keep as is) ---
def _format_suggestions_for_display(full_chart_suggestions_list: List[Dict[str, Any]], start_index: int, batch_size: int) -> tuple[str, List[str], bool]:
    bot_reply_segment = ""; suggestions_for_user_options = []
    actual_charts = [s for s in full_chart_suggestions_list if isinstance(s, dict) and s.get("type") not in ["Action", "Info", None]]
    end_index = min(start_index + batch_size, len(actual_charts)); charts_to_display_this_batch = actual_charts[start_index:end_index]
    if not charts_to_display_this_batch:
        if start_index == 0: bot_reply_segment = "Hmm, I couldn't find any specific chart suggestions based on your current criteria."
        else: bot_reply_segment = "No more specific chart suggestions found."
    else:
        if start_index == 0: bot_reply_segment = "Based on your info, here are a few chart ideas I found:<br>"
        else: bot_reply_segment = "Okay, here are some more chart types you might consider:<br>"
        for i, chart_sugg in enumerate(charts_to_display_this_batch):
            display_name = chart_sugg.get('name', 'Unknown Chart'); col_context_str = ""
            if chart_sugg.get("for_col"): col_context_str = f" for '{chart_sugg['for_col']}'"
            elif chart_sugg.get("for_cols"): col_context_str = f" for '{chart_sugg['for_cols']}'"
            # ADD SCORE TO DISPLAY TEXT
            score_text = f" (Score: {chart_sugg.get('score', 0)})"
            button_text = f"Select: {display_name}{col_context_str}{score_text}"
            bot_reply_segment += f"<br><strong>{start_index + i + 1}. {display_name}{col_context_str}{score_text}</strong>"
            bot_reply_segment += f".<br><em>Use this to: {chart_sugg.get('reason', 'Understand your data better.')}</em><br>"
            suggestions_for_user_options.append(button_text)
        if suggestions_for_user_options: bot_reply_segment += "<br>Choose one, or you can pick columns manually, or see more suggestions if available."
    more_available = end_index < len(actual_charts)
    return bot_reply_segment, suggestions_for_user_options, more_available

# --- Flask Routes ---
@app.route("/")
def home():
    keys_to_clear = ['visualization_questions_state', 'uploaded_filepath', 'uploaded_filename', 'df_columns', 
                     'user_answer_variable_types', 'user_answer_visualization_message', 'user_answer_variable_count', 'user_answer_emphasis', 'user_answer_columns_of_interest',
                     'chart_suggestions_list_actual', 'suggestion_batch_start_index', 
                     'selected_chart_for_plotting', 'plotting_columns', 
                     'last_plot_chart_type', 'last_plot_columns', 
                     'last_plot_custom_title', 'last_plot_custom_xlabel', 'last_plot_custom_ylabel', 
                     'last_plot_marker_style', 'last_plot_alpha_level', 'last_plot_hue_marker_map', 'hue_category_to_customize_marker',
                     'manual_columns_selected', 'last_suggestions']
    for key in keys_to_clear:
        session.pop(key, None)
    return render_template("index.html")

# --- UPDATED get_response function ---
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message"); response_data = {}; bot_reply = ""; user_input_lower = user_input.lower() if user_input else ""
    uploaded_filepath = session.get('uploaded_filepath'); df_columns = session.get('df_columns', [])

    if "restart questions" in user_input_lower or "restart" == user_input_lower:
        keys_to_clear_for_restart = [
            'user_answer_variable_types', 'user_answer_visualization_message', 
            'user_answer_variable_count', 'user_answer_emphasis', 'user_answer_columns_of_interest',
            'chart_suggestions_list_actual', 'suggestion_batch_start_index', 
            'selected_chart_for_plotting', 'plotting_columns', 
            'last_plot_chart_type', 'last_plot_columns', 
            'last_plot_custom_title', 'last_plot_custom_xlabel', 'last_plot_custom_ylabel', 
            'last_plot_marker_style', 'last_plot_alpha_level', 'last_plot_hue_marker_map', 
            'hue_category_to_customize_marker', 'manual_columns_selected' 
        ]
        for key in keys_to_clear_for_restart: session.pop(key, None)
        bot_reply = "Okay, I've cleared your previous answers. Let's start the questions again for the current data.<br><br><strong>1. Variable types?</strong>"
        session['visualization_questions_state'] = 'asking_variable_types'
        response_data = {"suggestions": ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of these types", "Not sure / Any", "Show data summary"]}
    elif "suggest another chart" in user_input_lower:
        if uploaded_filepath and session.get('user_answer_emphasis') is not None: # Check if all 5 questions were answered
            session['visualization_questions_state'] = 'visualization_info_gathered'
            for key in ['last_plot_chart_type', 'last_plot_columns', 'last_plot_custom_title', 'last_plot_custom_xlabel', 'last_plot_custom_ylabel', 'last_plot_marker_style', 'last_plot_alpha_level', 'last_plot_hue_marker_map', 'hue_category_to_customize_marker']:
                session.pop(key, None)
            bot_reply = "Okay, let's look for other visualizations. How proceed?"
            user_answers_for_count = {'variable_types': session.get('user_answer_variable_types', ''), 
                                      'message_insight': session.get('user_answer_visualization_message', ''), 
                                      'variable_count': session.get('user_answer_variable_count', ''),
                                      'emphasis': session.get('user_answer_emphasis', ''),
                                      'columns_of_interest': session.get('user_answer_columns_of_interest', '')
                                     }
            actual_chart_count = 0; df_sample_for_count = None
            if uploaded_filepath and all(user_answers_for_count.values()):
                try: df_sample_for_count = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e: print(f"Error reading df_sample for count: {e}")
                if df_sample_for_count is not None:
                    potential_chart_suggestions = suggest_charts_based_on_answers(user_answers_for_count, df_sample_for_count)
                    actual_chart_count = len([s for s in potential_chart_suggestions if s.get("type") not in ["Action", "Info", None]])
            suggest_button_text = "Suggest chart types for me"
            if actual_chart_count > 0: suggest_button_text += f" [{actual_chart_count} option{'s' if actual_chart_count != 1 else ''}]"
            else: suggest_button_text += " [See suggestions]"
            response_data = {"suggestions": [suggest_button_text, "Let me choose columns", "Restart questions", "Show data summary"]}
        else:
            bot_reply = "Let's start over. Please upload data and answer the initial questions first."
            home()
            response_data = {"suggestions": ["Upload Data", "Help"]}

    elif "descriptive statistics" in user_input_lower or "data summary" in user_input_lower or "summarize data" in user_input_lower:
        if uploaded_filepath:
            try:
                df_full = pd.read_csv(uploaded_filepath, low_memory=False) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath)
                stats_html = get_descriptive_stats_html(df_full)
                bot_reply = stats_html
                response_data = {"suggestions": ["Suggest charts for me", "Let me choose columns", "Restart questions"]}
                session['visualization_questions_state'] = 'visualization_info_gathered'
            except Exception as e:
                bot_reply = f"Sorry, couldn't generate summary: {str(e)[:100]}"
                response_data = {"suggestions": ["Upload Data", "Help"]}
        else:
            bot_reply = "Please upload a data file first."
            response_data = {"suggestions": ["Upload Data", "Help"]}
    else:
        current_viz_state = session.get('visualization_questions_state')
        user_answers = {
            'variable_types': session.get('user_answer_variable_types', ''),
            'message_insight': session.get('user_answer_visualization_message', ''),
            'variable_count': session.get('user_answer_variable_count', ''),
            'emphasis': session.get('user_answer_emphasis', ''),
            'columns_of_interest': session.get('user_answer_columns_of_interest', '')
        }
        
        if current_viz_state == 'asking_variable_types':
            session['user_answer_variable_types'] = user_input
            bot_reply = (f"Understood ({user_input}).<br><br><strong>2. Main message/insight?</strong>"
                         f"<br>(e.g., compare, distribution, relationship, trend)")
            session['visualization_questions_state'] = 'asking_visualization_message'
            response_data = {"suggestions": ["Compare values/categories", "Show data distribution", "Identify relationships", "Track trends over time"]}
        
        elif current_viz_state == 'asking_visualization_message':
            session['user_answer_visualization_message'] = user_input
            columns_reminder = ""
            if df_columns: columns_reminder = f"(Cols: {', '.join(df_columns[:3])}...)"
            bot_reply = (f"Goal: \"{user_input}\".<br><br><strong>3. How many variables per chart?</strong> {columns_reminder}"
                         f"<br>(e.g., one, two, more)")
            session['visualization_questions_state'] = 'asking_variable_count'
            response_data = {"suggestions": ["One variable", "Two variables", "More than two"]}

        elif current_viz_state == 'asking_variable_count':
            session['user_answer_variable_count'] = user_input
            bot_reply = (f"Got it: \"{user_input}\".<br><br>"
                         f"<strong>4. What's the primary characteristic you want to emphasize?</strong>")
            session['visualization_questions_state'] = 'asking_emphasis'
            response_data = {"suggestions": [
                "Exact values and magnitudes", "Distribution and spread of data",
                "Relationship or correlation", "Proportions or parts of a whole",
                "Change over time or sequence", "Comparison between distinct groups"
            ]}
        
        elif current_viz_state == 'asking_emphasis':
            session['user_answer_emphasis'] = user_input
            bot_reply = (f"Emphasis: \"{user_input}\".<br><br>"
                         f"<strong>5. Are there any specific columns you are most interested in exploring for this analysis?</strong><br>"
                         f"You can list one or more column names separated by commas (e.g., '{df_columns[0] if df_columns else 'Age'}, {df_columns[1] if len(df_columns)>1 else 'Sleep Duration'}'). If not, just type 'any' or 'skip'.")
            session['visualization_questions_state'] = 'asking_columns_of_interest'
            example_suggestions = [col for col in df_columns[:2]] 
            if not example_suggestions and df_columns: example_suggestions = [df_columns[0]]
            if not example_suggestions: example_suggestions = ["Age, Sleep Duration"] 
            response_data = {"suggestions": example_suggestions + ["any", "skip"]}

        elif current_viz_state == 'asking_columns_of_interest':
            session['user_answer_columns_of_interest'] = user_input
            user_answers['variable_types'] = session.get('user_answer_variable_types', '')
            user_answers['message_insight'] = session.get('user_answer_visualization_message', '')
            user_answers['variable_count'] = session.get('user_answer_variable_count', '')
            user_answers['emphasis'] = session.get('user_answer_emphasis', '')
            user_answers['columns_of_interest'] = user_input

            bot_reply = (f"Great! Here's what I have for your preferences:<br>"
                         f"- Variable Types: \"{user_answers['variable_types']}\"<br>"
                         f"- Main Goal: \"{user_answers['message_insight']}\"<br>"
                         f"- Variables per Chart: \"{user_answers['variable_count']}\"<br>"
                         f"- Emphasis: \"{user_answers['emphasis']}\"<br>"
                         f"- Columns of Interest: \"{user_answers['columns_of_interest'] if user_answers['columns_of_interest'] and user_answers['columns_of_interest'].lower() not in ['any','none','skip'] else 'Any/None Specified'}\"<br><br>"
                         f"What would you like to do next?")
            session['visualization_questions_state'] = 'visualization_info_gathered'
            
            actual_chart_count = 0; df_sample_for_count = None
            if uploaded_filepath:
                try: df_sample_for_count = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e: print(f"Error reading df_sample for count: {e}")
            if df_sample_for_count is not None and not df_sample_for_count.empty:
                potential_chart_suggestions = suggest_charts_based_on_answers(user_answers, df_sample_for_count)
                actual_chart_count = len([s for s in potential_chart_suggestions if s.get("type") not in ["Action", "Info", None]])
            
            suggest_button_text = "Suggest chart types for me"
            if actual_chart_count > 0: suggest_button_text += f" [{actual_chart_count} option{'s' if actual_chart_count != 1 else ''}]"
            else: suggest_button_text += " [No specific suggestions found]"
            response_data = {"suggestions": ["Show data summary", suggest_button_text, "Let me choose columns", "Restart questions"]}
        
        elif current_viz_state == 'visualization_info_gathered':
            if "suggest chart types for me" in user_input_lower:
                df_sample = None
                if uploaded_filepath:
                    try:
                        if uploaded_filepath.endswith(".csv"): df_sample = pd.read_csv(uploaded_filepath, nrows=100)
                        else: df_sample = pd.read_excel(uploaded_filepath, nrows=100)
                    except Exception as e: print(f"Error reading df_sample: {e}"); df_sample = None
                
                current_user_answers = {
                    'variable_types': session.get('user_answer_variable_types', ''),
                    'message_insight': session.get('user_answer_visualization_message', ''),
                    'variable_count': session.get('user_answer_variable_count', ''),
                    'emphasis': session.get('user_answer_emphasis', ''),
                    'columns_of_interest': session.get('user_answer_columns_of_interest', '')
                }
                full_suggestions_list = suggest_charts_based_on_answers(current_user_answers, df_sample)
                session['chart_suggestions_list_actual'] = [s for s in full_suggestions_list if s.get("type") not in ["Action", "Info", None]]
                session['suggestion_batch_start_index'] = 0
                
                bot_reply_segment, suggestions_for_display, more_available = _format_suggestions_for_display(session['chart_suggestions_list_actual'], 0, SUGGESTION_BATCH_SIZE)
                bot_reply = bot_reply_segment; response_data_suggestions = suggestions_for_display
                if more_available: response_data_suggestions.append("Show more chart suggestions")
                response_data_suggestions.extend(["Pick columns manually", "Restart questions"]); response_data["suggestions"] = response_data_suggestions
                session['visualization_questions_state'] = 'awaiting_chart_type_selection'
            elif "choose columns" in user_input_lower or "pick columns" in user_input_lower:
                 if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
                 else: bot_reply = f"Sure! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
            else: bot_reply = "What next?"; response_data = {"suggestions": session.get('last_suggestions', ["Show data summary", "Suggest charts for me", "Pick columns", "Restart"])}
        
        elif current_viz_state == 'awaiting_chart_type_selection':
            user_choice_full_text = user_input.strip()
            chart_suggestions_list_actual = session.get('chart_suggestions_list_actual', [])
            selected_chart_info = None; bot_reply = ""
            if "show more chart suggestions" in user_input_lower:
                start_index = session.get('suggestion_batch_start_index', 0) + SUGGESTION_BATCH_SIZE
                session['suggestion_batch_start_index'] = start_index
                bot_reply_segment, suggestions_for_display, more_available = _format_suggestions_for_display(chart_suggestions_list_actual, start_index, SUGGESTION_BATCH_SIZE)
                bot_reply = bot_reply_segment; response_data_suggestions = suggestions_for_display
                if more_available: response_data_suggestions.append("Show more chart suggestions")
                response_data_suggestions.extend(["Pick columns manually", "Restart questions"]); response_data["suggestions"] = response_data_suggestions
            elif user_choice_full_text == "Pick columns manually":
                 if not df_columns: bot_reply = "Need columns list. Upload data."; response_data = {"suggestions": ["Upload Data"]}; session['visualization_questions_state'] = None
                 else: bot_reply = f"Okay! Which columns? (Available: {', '.join(df_columns)})"; session['visualization_questions_state'] = 'awaiting_column_selection_general'; session['manual_columns_selected'] = []; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting", "Cancel selection"]}
            elif user_choice_full_text.startswith("Select: "):
                match = re.match(r"Select: ([\w\s\(\)-]+?)(?: for '([\w\s,&]+)')?(?: \(Score: -?\d+\))?$", user_choice_full_text)
                parsed_chart_name = None; parsed_col_details = None
                if match: parsed_chart_name = match.group(1).strip(); parsed_col_details = match.group(2).strip() if match.group(2) else None
                
                if parsed_chart_name:
                    for sugg in chart_suggestions_list_actual:
                        if sugg.get("name") == parsed_chart_name:
                            sugg_col_str = sugg.get("for_col") or sugg.get("for_cols")
                            if parsed_col_details:
                                if sugg_col_str == parsed_col_details: selected_chart_info = sugg; break
                            elif not sugg_col_str: selected_chart_info = sugg; break
                    if not selected_chart_info and not parsed_col_details:
                        potential_matches_by_name = [s for s in chart_suggestions_list_actual if s.get("name") == parsed_chart_name]
                        if len(potential_matches_by_name) == 1: selected_chart_info = potential_matches_by_name[0]
                        elif len(potential_matches_by_name) > 1:
                            bot_reply = f"Okay, a <strong>{parsed_chart_name}</strong>. It can be used with different columns/settings. Which specific version did you mean?<br>"
                            disambiguation_options = []
                            for s_match in potential_matches_by_name:
                                d_text = f"Select: {s_match['name']}"
                                if s_match.get("for_col"): d_text += f" for '{s_match['for_col']}'"
                                elif s_match.get("for_cols"): d_text += f" for '{s_match['for_cols']}'"
                                d_text += f" (Score: {s_match.get('score',0)})"
                                disambiguation_options.append(d_text)
                            response_data["suggestions"] = disambiguation_options + ["Pick columns manually", "Restart questions"]
                
                if selected_chart_info:
                    session['selected_chart_for_plotting'] = selected_chart_info; chart_name = selected_chart_info['name']; required_cols_specific = selected_chart_info.get('required_cols_specific')
                    bot_reply = f"Okay: <strong>{chart_name}</strong>. "
                    if required_cols_specific:
                        cols_to_use_str = ", ".join(required_cols_specific);
                        print(f"DEBUG Pre-validation SKIPPED for suggested chart: {chart_name} with cols {cols_to_use_str}")
                        bot_reply += f"I'll use columns: <strong>{cols_to_use_str}</strong>. Plot?"
                        session['plotting_columns'] = required_cols_specific
                        response_data = {"suggestions": [f"Yes, plot {chart_name}", "Choose other columns", "Back to chart list"]}
                        session['visualization_questions_state'] = 'confirm_plot_details'
                    else:
                         bot_reply += f"Which columns for the {chart_name}? Available: {', '.join(df_columns)}";
                         if chart_name == "Histogram": bot_reply += "<br><i>Hint: Needs one numerical column.</i>"
                         elif chart_name == "Scatter Plot": bot_reply += "<br><i>Hint: Needs two numerical columns (or 2 numerical & 1 categorical for color).</i>"
                         elif chart_name == "Box Plots (by Category)": bot_reply += "<br><i>Hint: Needs one categorical and one numerical column.</i>"
                         session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
                elif not bot_reply:
                    bot_reply = "I couldn't quite match that to a suggested chart. Please choose from the buttons or try rephrasing."
                    start_index = session.get('suggestion_batch_start_index', 0)
                    bot_reply_segment, suggestions_for_display, more_available = _format_suggestions_for_display(chart_suggestions_list_actual, start_index, SUGGESTION_BATCH_SIZE)
                    if bot_reply_segment and "Hmm, I couldn't find" not in bot_reply_segment and "No more specific" not in bot_reply_segment : bot_reply += "<br><br>" + bot_reply_segment
                    else: bot_reply += "<br>You can pick columns manually or restart the questions."
                    response_data_suggs = suggestions_for_display
                    if more_available: response_data_suggs.append("Show more chart suggestions")
                    response_data_suggs.extend(["Pick columns manually", "Restart questions"]); response_data = {"suggestions": response_data_suggs}
            else:
                bot_reply = "Please select a chart from the options, or choose to pick columns manually."
                response_data = {"suggestions": session.get('last_suggestions', [])} 
        
        elif current_viz_state == 'confirm_plot_details':
            chart_to_plot_info = session.get('selected_chart_for_plotting'); cols_for_plot = session.get('plotting_columns')
            custom_title = session.get('last_plot_custom_title'); custom_xlabel = session.get('last_plot_custom_xlabel'); custom_ylabel = session.get('last_plot_custom_ylabel')
            custom_marker = session.get('last_plot_marker_style'); custom_alpha = session.get('last_plot_alpha_level'); custom_hue_marker_map = session.get('last_plot_hue_marker_map')

            if user_input.startswith("Yes, plot"):
                if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                    chart_name = chart_to_plot_info['name']; bot_reply = f"Generating <strong>{chart_name}</strong>..."
                    plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, chart_name, cols_for_plot, custom_title, custom_xlabel, custom_ylabel, custom_marker, custom_alpha, custom_hue_marker_map)
                    if plot_image_uri: response_data["plot_image"] = plot_image_uri; bot_reply = f"Here is your <strong>{chart_name}</strong> (using columns: {', '.join(cols_for_plot)})."
                    else: bot_reply = f"Sorry, couldn't generate the <strong>{chart_name}</strong>.<br><strong>Reason:</strong> {error_msg or 'Unknown error.'}<br>Try suggesting another chart or picking different columns."
                    session['last_plot_chart_type'] = chart_name; session['last_plot_columns'] = cols_for_plot
                    session['visualization_questions_state'] = 'awaiting_plot_customization';
                    customization_suggestions = ["Change Title", "Change X-axis Label", "Change Y-axis Label"]
                    if chart_name == "Scatter Plot":
                        customization_suggestions.append("Change Marker Style")
                        customization_suggestions.append("Change Point Transparency")
                        if len(cols_for_plot) == 3: customization_suggestions.append("Change Group Markers")
                    customization_suggestions.extend(["Looks good, suggest another chart", "Restart questions"])
                    response_data.setdefault("suggestions", []).extend(customization_suggestions)
                else: bot_reply = "Missing details/file path."; session['visualization_questions_state'] = 'visualization_info_gathered'; response_data = {"suggestions": ["Suggest charts", "Pick columns"]}
            elif "choose other columns" in user_input_lower or "change columns" in user_input_lower:
                 chart_name = chart_to_plot_info['name'] if chart_to_plot_info else 'chart'; bot_reply = f"Okay, for <strong>{chart_name}</strong>, which columns? Available: {', '.join(df_columns)}"; session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
            else:
                bot_reply = "Okay, back to chart list."; session['visualization_questions_state'] = 'awaiting_chart_type_selection';
                full_list_actual = session.get('chart_suggestions_list_actual', []); session['suggestion_batch_start_index'] = 0
                bot_reply_segment, suggestions_for_display, more_available = _format_suggestions_for_display(full_list_actual, 0, SUGGESTION_BATCH_SIZE)
                bot_reply = bot_reply_segment; response_data_suggs = suggestions_for_display
                if more_available: response_data_suggs.append("Show more chart suggestions")
                response_data_suggs.extend(["Pick columns manually", "Restart questions"]); response_data = {"suggestions": response_data_suggs}

        elif current_viz_state == 'awaiting_plot_customization':
            base_suggestions = ["Change Title", "Change X-axis Label", "Change Y-axis Label"]
            last_chart_type = session.get('last_plot_chart_type')
            if last_chart_type == "Scatter Plot":
                base_suggestions.append("Change Marker Style")
                base_suggestions.append("Change Point Transparency")
                if len(session.get('last_plot_columns', [])) == 3: base_suggestions.append("Change Group Markers")
            base_suggestions.extend(["Looks good, suggest another chart", "Restart questions"])

            if user_input_lower == "change title": bot_reply = "Great! What would you like the new title to be?"; session['visualization_questions_state'] = 'awaiting_new_title'; response_data = {"suggestions": ["Cancel customization"]}
            elif user_input_lower == "change x-axis label": bot_reply = "Sounds good. What should the new X-axis label be?"; session['visualization_questions_state'] = 'awaiting_new_xlabel'; response_data = {"suggestions": ["Cancel customization"]}
            elif user_input_lower == "change y-axis label": bot_reply = "Okay. What should the new Y-axis label be?"; session['visualization_questions_state'] = 'awaiting_new_ylabel'; response_data = {"suggestions": ["Cancel customization"]}
            elif user_input_lower == "change marker style" and last_chart_type == "Scatter Plot":
                bot_reply = "What marker style for the scatter plot? (e.g., Circle, Square, Triangle, X, Plus, Diamond)"
                session['visualization_questions_state'] = 'awaiting_new_marker'
                response_data = {"suggestions": ["Circle (o)", "Square (s)", "Triangle (^)", "X (x)", "Plus (+)", "Diamond (D)", "Cancel customization"]}
            elif user_input_lower == "change point transparency" and last_chart_type == "Scatter Plot":
                bot_reply = "How transparent should the points be?"
                session['visualization_questions_state'] = 'awaiting_new_alpha'
                response_data = {"suggestions": ["Slightly Transparent (0.7)", "Medium Transparency (0.5)", "Very Transparent (0.3)", "Solid (No Transparency)", "Cancel customization"]}
            elif user_input_lower == "change group markers" and last_chart_type == "Scatter Plot" and len(session.get('last_plot_columns',[])) == 3:
                hue_column_name = session.get('last_plot_columns')[2]
                df_temp = None
                if uploaded_filepath:
                    try: df_temp = pd.read_csv(uploaded_filepath, usecols=[hue_column_name]) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=[hue_column_name])
                    except Exception as e: print(f"Error reading hue column for marker customization: {e}")
                if df_temp is not None and not df_temp.empty:
                    unique_hue_values = df_temp[hue_column_name].dropna().unique().tolist()
                    session['unique_hue_categories_for_marker'] = [str(val) for val in unique_hue_values[:10]]
                    bot_reply = f"Okay, for the '{hue_column_name}' groups, which group's marker do you want to change?"
                    response_data = {"suggestions": session['unique_hue_categories_for_marker'] + ["Cancel customization"]}; session['visualization_questions_state'] = 'selecting_hue_category_for_marker'
                else: bot_reply = f"Could not retrieve groups for '{hue_column_name}'. Please try again or a different customization."; response_data = {"suggestions": base_suggestions}
            elif "looks good" in user_input_lower or "suggest another" in user_input_lower:
                bot_reply = "Alright! Let's find another chart for you. How would you like to proceed?"; session['visualization_questions_state'] = 'visualization_info_gathered'
                actual_chart_count = 0; temp_answers = user_answers
                df_sample_for_count = None
                if uploaded_filepath and all(temp_answers.values()):
                    try: df_sample_for_count = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                    except Exception as e: print(f"Error reading df_sample: {e}")
                    if df_sample_for_count is not None: potential_suggestions = suggest_charts_based_on_answers(temp_answers, df_sample_for_count); actual_chart_count = len([s for s in potential_suggestions if s.get("type") not in ["Action", "Info", None]])
                suggest_button_text = "Suggest chart types for me";
                if actual_chart_count > 0: suggest_button_text += f" [{actual_chart_count} option{'s' if actual_chart_count != 1 else ''}]"
                else: suggest_button_text += " [No specific suggestions found]"
                response_data = {"suggestions": [suggest_button_text, "Let me choose columns", "Restart questions", "Show data summary"]}
                for key in ['last_plot_chart_type', 'last_plot_columns', 'last_plot_custom_title', 'last_plot_custom_xlabel', 'last_plot_custom_ylabel', 'last_plot_marker_style', 'last_plot_alpha_level', 'last_plot_hue_marker_map', 'hue_category_to_customize_marker']: session.pop(key, None)
            else:
                bot_reply = "Here's your chart again. You can customize it or move on."
                last_chart_type = session.get('last_plot_chart_type'); last_columns = session.get('last_plot_columns')
                if last_chart_type and last_columns and uploaded_filepath:
                    plot_image_uri, _ = generate_plot_and_get_uri(uploaded_filepath, last_chart_type, last_columns, session.get('last_plot_custom_title'), session.get('last_plot_custom_xlabel'), session.get('last_plot_custom_ylabel'), session.get('last_plot_marker_style'), session.get('last_plot_alpha_level'), session.get('last_plot_hue_marker_map'))
                    if plot_image_uri: response_data["plot_image"] = plot_image_uri
                response_data["suggestions"] = base_suggestions

        elif current_viz_state in ['awaiting_new_title', 'awaiting_new_xlabel', 'awaiting_new_ylabel', 'awaiting_new_marker', 'awaiting_new_alpha', 'selecting_hue_category_for_marker', 'awaiting_marker_for_specific_category']:
            last_chart_type = session.get('last_plot_chart_type'); last_columns = session.get('last_plot_columns')
            if current_viz_state == 'awaiting_new_title': session['last_plot_custom_title'] = user_input
            elif current_viz_state == 'awaiting_new_xlabel': session['last_plot_custom_xlabel'] = user_input
            elif current_viz_state == 'awaiting_new_ylabel': session['last_plot_custom_ylabel'] = user_input
            elif current_viz_state == 'awaiting_new_marker':
                marker_map = {"circle (o)": "o", "square (s)": "s", "triangle (^)": "^", "x (x)": "x", "plus (+)": "+", "diamond (d)": "D"}
                session['last_plot_marker_style'] = marker_map.get(user_input.lower(), 'o')
                session.pop('last_plot_hue_marker_map', None) 
            elif current_viz_state == 'awaiting_new_alpha':
                alpha_map = {"slightly transparent (0.7)": 0.7, "medium transparency (0.5)": 0.5, "very transparent (0.3)": 0.3, "solid (no transparency)": 1.0}
                session['last_plot_alpha_level'] = alpha_map.get(user_input.lower(), 0.7)
            elif current_viz_state == 'selecting_hue_category_for_marker':
                if user_input in session.get('unique_hue_categories_for_marker', []):
                    session['hue_category_to_customize_marker'] = user_input
                    bot_reply = f"Okay, for group '{user_input}', what marker style would you like?"
                    session['visualization_questions_state'] = 'awaiting_marker_for_specific_category'
                    response_data = {"suggestions": ["Circle (o)", "Square (s)", "Triangle (^)", "X (x)", "Plus (+)", "Diamond (D)", "Cancel customization"]}
                else: bot_reply = "Invalid group. Please choose from the list."; response_data = {"suggestions": session.get('unique_hue_categories_for_marker', []) + ["Cancel customization"]}
            elif current_viz_state == 'awaiting_marker_for_specific_category':
                category_to_change = session.get('hue_category_to_customize_marker')
                if category_to_change:
                    marker_map = {"circle (o)": "o", "square (s)": "s", "triangle (^)": "^", "x (x)": "x", "plus (+)": "+", "diamond (d)": "D"}
                    selected_marker_code = marker_map.get(user_input.lower(), 'o')
                    hue_marker_map = session.get('last_plot_hue_marker_map', {})
                    hue_marker_map[category_to_change] = selected_marker_code
                    session['last_plot_hue_marker_map'] = hue_marker_map
                    session.pop('hue_category_to_customize_marker', None)
                    session.pop('last_plot_marker_style', None) 
                else: bot_reply = "Something went wrong selecting the category. Let's try again."; session['visualization_questions_state'] = 'awaiting_plot_customization'
            
            if current_viz_state != 'selecting_hue_category_for_marker':
                custom_title = session.get('last_plot_custom_title'); custom_xlabel = session.get('last_plot_custom_xlabel'); custom_ylabel = session.get('last_plot_custom_ylabel'); custom_marker = session.get('last_plot_marker_style'); custom_alpha = session.get('last_plot_alpha_level'); custom_hue_marker_map = session.get('last_plot_hue_marker_map')
                if last_chart_type and last_columns and uploaded_filepath:
                    if not bot_reply : bot_reply = f"Okay, regenerating the <strong>{last_chart_type}</strong> with your changes..."
                    plot_image_uri, error_msg = generate_plot_and_get_uri(uploaded_filepath, last_chart_type, last_columns, custom_title, custom_xlabel, custom_ylabel, custom_marker, custom_alpha, custom_hue_marker_map)
                    if plot_image_uri: response_data["plot_image"] = plot_image_uri; bot_reply = f"Here's the updated <strong>{last_chart_type}</strong>. Anything else?"
                    else: bot_reply = f"Sorry, couldn't update the plot. Reason: {error_msg or 'Unknown error.'}"
                    session['visualization_questions_state'] = 'awaiting_plot_customization';
                    customization_suggestions = ["Change Title", "Change X-axis Label", "Change Y-axis Label"]
                    if last_chart_type == "Scatter Plot":
                        customization_suggestions.append("Change Marker Style")
                        customization_suggestions.append("Change Point Transparency")
                        if len(last_columns) == 3: customization_suggestions.append("Change Group Markers")
                    customization_suggestions.extend(["Looks good, suggest another chart", "Restart questions"])
                    response_data["suggestions"] = customization_suggestions
                elif not bot_reply: bot_reply = "Something went wrong, I've lost track of the previous plot. Let's try suggesting charts again."; session['visualization_questions_state'] = 'visualization_info_gathered'; response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns", "Restart questions"]}
        
        elif current_viz_state == 'awaiting_columns_for_selected_chart':
            potential_cols_str = user_input.replace("Use:", "").strip(); user_selected_cols = [col.strip() for col in potential_cols_str.split(',') if col.strip() and col.strip() in df_columns]
            chart_to_plot_info = session.get('selected_chart_for_plotting'); chart_name = chart_to_plot_info.get('name', 'chart') if chart_to_plot_info else 'chart'
            if user_selected_cols:
                validation_msg = None
                if uploaded_filepath:
                     try:
                         df_val = pd.read_csv(uploaded_filepath, usecols=user_selected_cols, nrows=PRE_VALIDATION_SAMPLE_SIZE) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=user_selected_cols, nrows=PRE_VALIDATION_SAMPLE_SIZE)
                         if df_val is not None and not df_val.empty:
                              for col_idx in df_val.columns:
                                  df_val[col_idx] = clean_numeric_column(df_val[col_idx])
                                  if any(substr in col_idx.lower() for substr in ['date','time','yr','year']) and not is_datetime64_any_dtype(df_val[col_idx]):
                                        try:
                                            converted_date = pd.to_datetime(df_val[col_idx], errors='coerce')
                                            if converted_date.notna().any(): df_val[col_idx] = converted_date
                                        except: pass
                              validation_msg = validate_columns_for_chart(chart_name, user_selected_cols, df_val)
                         else: validation_msg = "Could not read sample data."
                     except Exception as e: validation_msg = f"Couldn't validate ({str(e)[:50]}...)."; print(f"DEBUG Validation Read/Clean Error: {e}")
                if validation_msg is None:
                    session['plotting_columns'] = user_selected_cols; bot_reply = f"Using: <strong>{', '.join(user_selected_cols)}</strong> for <strong>{chart_name}</strong>. Plot?"; response_data = {"suggestions": ["Yes, generate plot", "Change columns", "Back to chart list"]}; session['visualization_questions_state'] = 'confirm_plot_details'
                else: bot_reply = f"The columns <strong>{', '.join(user_selected_cols)}</strong> might not work for <strong>{chart_name}</strong>. <br><strong>Reason:</strong> {validation_msg}<br>Please select valid columns from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
            elif "list all columns" in user_input_lower: bot_reply = f"Available: {', '.join(df_columns)}.<br>Which for {chart_name}?"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
            elif "back to chart list" in user_input_lower:
                session['visualization_questions_state'] = 'awaiting_chart_type_selection'; bot_reply = "Okay, which chart type?";
                full_list_actual = session.get('chart_suggestions_list_actual', []); session['suggestion_batch_start_index'] = 0
                bot_reply_segment, suggestions_for_display, more_available = _format_suggestions_for_display(full_list_actual, 0, SUGGESTION_BATCH_SIZE)
                bot_reply = bot_reply_segment; response_data_suggs = suggestions_for_display
                if more_available: response_data_suggs.append("Show more chart suggestions")
                response_data_suggs.extend(["Pick columns manually", "Restart questions"]); response_data = {"suggestions": response_data_suggs}
            else: bot_reply = f"Invalid columns for <strong>{chart_name}</strong>. Choose from: {', '.join(df_columns)}"; response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Back to chart list"]}
        
        elif current_viz_state == 'awaiting_column_selection_general':
            if "finished selecting" in user_input_lower:
                selected_cols = session.get('manual_columns_selected', [])
                if selected_cols:
                    bot_reply = f"Selected: <strong>{', '.join(selected_cols)}</strong>. What chart type would you like for these columns?"
                    session['plotting_columns'] = selected_cols
                    session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'
                    dynamic_suggestions = ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Pie Chart"] 
                    if uploaded_filepath and selected_cols:
                        try:
                            df_temp_validate = pd.read_csv(uploaded_filepath, usecols=selected_cols, nrows=PRE_VALIDATION_SAMPLE_SIZE) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, usecols=selected_cols, nrows=PRE_VALIDATION_SAMPLE_SIZE)
                            for col_val_idx in df_temp_validate.columns:
                                df_temp_validate[col_val_idx] = clean_numeric_column(df_temp_validate[col_val_idx])
                                if any(substr in col_val_idx.lower() for substr in ['date','time','yr','year']) and not is_datetime64_any_dtype(df_temp_validate[col_val_idx]):
                                    try: 
                                        converted_date = pd.to_datetime(df_temp_validate[col_val_idx], errors='coerce');
                                        if converted_date.notna().any(): df_temp_validate[col_val_idx] = converted_date
                                    except: pass
                            temp_col_types = get_simplified_column_types(df_temp_validate)
                            num_sel_numerical = sum(1 for c in selected_cols if temp_col_types.get(c) == 'numerical')
                            num_sel_categorical = sum(1 for c in selected_cols if temp_col_types.get(c) in ['categorical', 'categorical_numeric'])
                            num_sel_datetime = sum(1 for c in selected_cols if temp_col_types.get(c) == 'datetime')
                            if len(selected_cols) == 1:
                                if num_sel_numerical == 1 or (num_sel_categorical == 1 and temp_col_types.get(selected_cols[0]) == 'categorical_numeric'): dynamic_suggestions = ["Histogram", "Box Plot", "Density Plot", "Bar Chart (Counts)"]
                                elif num_sel_categorical == 1: dynamic_suggestions = ["Bar Chart (Counts)", "Pie Chart"]
                            elif len(selected_cols) == 2:
                                if num_sel_numerical == 2: dynamic_suggestions = ["Scatter Plot", "Line Chart"]
                                elif num_sel_categorical == 2: dynamic_suggestions = ["Grouped Bar Chart", "Heatmap (Counts)"]
                                elif num_sel_numerical == 1 and num_sel_categorical == 1: dynamic_suggestions = ["Box Plots (by Category)", "Violin Plots (by Category)", "Bar Chart (Aggregated)"]
                                elif num_sel_datetime == 1 and num_sel_numerical == 1: dynamic_suggestions = ["Line Chart", "Area Chart"]
                            elif len(selected_cols) > 2 and num_sel_numerical >=3 : dynamic_suggestions = ["Pair Plot", "Correlation Heatmap", "Parallel Coordinates Plot"]
                        except Exception as e: print(f"Error tailoring suggestions for manual cols: {e}")
                    response_data = {"suggestions": dynamic_suggestions}
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
                else: bot_reply = f"Sorry, couldn't generate the <strong>{chart_type_from_user}</strong>.<br><strong>Reason:</strong> {error_msg or 'Unknown error.'}"
                session['last_plot_chart_type'] = chart_type_from_user; session['last_plot_columns'] = cols_for_plot
                session['last_plot_custom_title'] = None; session['last_plot_custom_xlabel'] = None; session['last_plot_custom_ylabel'] = None; session['last_plot_marker_style'] = None; session['last_plot_alpha_level'] = None; session['last_plot_hue_marker_map'] = None
                session['visualization_questions_state'] = 'awaiting_plot_customization';
                customization_suggestions = ["Change Title", "Change X-axis Label", "Change Y-axis Label"]
                if chart_type_from_user == "Scatter Plot":
                    customization_suggestions.extend(["Change Marker Style", "Change Point Transparency"])
                    if len(cols_for_plot) == 3: customization_suggestions.append("Change Group Markers")
                customization_suggestions.extend(["Looks good, suggest another chart", "Restart questions"])
                response_data.setdefault("suggestions", []).extend(customization_suggestions)
            else: bot_reply = "Missing column/file details."; session['visualization_questions_state'] = None; response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart questions", "Upload new data"])
        
        else: 
            if not bot_reply:
                nlu_output = nlu_get_bot_response(user_input)
                if isinstance(nlu_output, dict):
                    bot_reply = nlu_output.get("response", "Sorry, I'm not sure how to respond.")
                    temp_suggestions = nlu_output.get("suggestions", [])
                else:
                    bot_reply = nlu_output
                    temp_suggestions = []
                if not temp_suggestions:
                    temp_suggestions = ["Help", "Upload Data", "What can you do?"]
                response_data["suggestions"] = temp_suggestions
            elif 'suggestions' not in response_data:
                response_data["suggestions"] = ["Help", "Upload Data"]
    
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
            df = None 
            try: 
                if filename.endswith(".csv"):
                    df = pd.read_csv(filepath, low_memory=False)
                elif filename.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(filepath, engine=read_engine)
                else: 
                    return jsonify({"response": "Unsupported file type after initial check."}), 400
            except Exception as read_err:
                print(f"Initial read error for {filename}: {read_err}")
                if filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
                        print(f"Successfully read {filename} with latin1 encoding after initial failure.")
                    except Exception as read_err_latin1:
                        print(f"Latin-1 read error for {filename}: {read_err_latin1}")
            
            if df is None or df.empty: 
                print(f"Failed to read or empty dataframe for file: {filename}")
                error_detail = f" (Details: {str(read_err)[:100]}...)" if 'read_err' in locals() and read_err else ""
                return jsonify({"response": f"Could not read or understand the file format of '{filename}'. Please ensure it's a valid CSV or Excel file and not corrupted.{error_detail}"}), 400

            session['df_columns'] = list(df.columns); preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)
            total_rows,total_columns=len(df),len(df.columns); missing_values=df.isnull().sum().sum(); duplicate_rows=df.duplicated().sum(); total_cells=total_rows*total_columns; missing_percent=(missing_values/total_cells)*100 if total_cells else 0
            initial_bot_message = (f"✅ <strong>{filename}</strong> uploaded.<br><br>"
                                 f"🔍 Quality Check: {total_rows} R, {total_columns} C; {missing_values} missing ({missing_percent:.1f}%); {duplicate_rows} duplicates.<br><br>"
                                 f"To help you visualize, I have a few questions. You can also ask for a 'data summary' or to 'pick columns manually' anytime.<br><br>"
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
    app.run(debug=True)
