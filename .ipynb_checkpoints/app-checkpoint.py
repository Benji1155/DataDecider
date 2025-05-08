from flask import Flask, render_template, request, jsonify, session
from bot_logic import get_bot_response as nlu_get_bot_response
from werkzeug.utils import secure_filename
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype
import io
import base64

# Matplotlib and Seaborn setup for plotting
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend, crucial for web servers
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

app.config['UPLOAD_FOLDER'] = 'uploaded_files'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xls', 'xlsx'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_simplified_column_types(df):
    simplified_types = {}
    if df is None or df.empty:
        return simplified_types
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            if df[col].nunique() < 10 and (df[col].nunique() < len(df) / 10 or len(df) < 50) : # Heuristic for categorical numeric
                 simplified_types[col] = 'categorical_numeric'
            else:
                simplified_types[col] = 'numerical'
        elif is_datetime64_any_dtype(df[col]):
            simplified_types[col] = 'datetime'
        elif is_string_dtype(df[col]) or df[col].dtype == 'object':
            if df[col].nunique() < len(df) * 0.8 and df[col].nunique() < 100 : # Heuristic for categorical text
                simplified_types[col] = 'categorical'
            else:
                simplified_types[col] = 'id_like_text'
        else:
            simplified_types[col] = 'other'
    return simplified_types

def suggest_charts_based_on_answers(user_answers, df_sample):
    suggestions = []
    if df_sample is None or df_sample.empty:
        return [{"name": "Cannot suggest charts without data insight.", "type": "Info", "reason": "Data sample is empty.", "required_cols_specific": []}]

    col_types = get_simplified_column_types(df_sample)
    numerical_cols = [col for col, typ in col_types.items() if typ == 'numerical']
    categorical_cols = [col for col, typ in col_types.items() if typ == 'categorical' or typ == 'categorical_numeric']
    datetime_cols = [col for col, typ in col_types.items() if typ == 'datetime']

    ua_var_count_str = user_answers.get('variable_count', '').lower()
    ua_var_types = user_answers.get('variable_types', '').lower()
    ua_message = user_answers.get('message_insight', '').lower()

    if ("one" in ua_var_count_str or "1" in ua_var_count_str) and \
       ("distribut" in ua_message or "spread" in ua_message or "summary" in ua_message) and \
       ("numer" in ua_var_types or not ua_var_types or "any" in ua_var_types) and numerical_cols:
        for col in numerical_cols:
            suggestions.append({"name": "Histogram", "for_col": col, "type": "Univariate Numerical", "reason": f"Shows distribution of '{col}'.", "required_cols_specific": [col]})
            suggestions.append({"name": "Box Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Summarizes '{col}' (median, quartiles, outliers).", "required_cols_specific": [col]})
            suggestions.append({"name": "Density Plot", "for_col": col, "type": "Univariate Numerical", "reason": f"Smoothly shows distribution of '{col}'.", "required_cols_specific": [col]})

    if ("one" in ua_var_count_str or "1" in ua_var_count_str) and \
       ("proportion" in ua_message or "share" in ua_message or "frequency" in ua_message or "count" in ua_message or "values" in ua_message) and \
       ("categor" in ua_var_types or not ua_var_types or "any" in ua_var_types) and categorical_cols:
        for col in categorical_cols:
            suggestions.append({"name": "Bar Chart (Counts)", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows counts for each category in '{col}'.", "required_cols_specific": [col]})
            if df_sample[col].nunique() < 8 and df_sample[col].nunique() > 1 : # Corrected nunique > 1
                 suggestions.append({"name": "Pie Chart", "for_col": col, "type": "Univariate Categorical", "reason": f"Shows proportions for '{col}'. Best for few categories.", "required_cols_specific": [col]})

    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
       ("relationship" in ua_message or "correlat" in ua_message or "scatter" in ua_message) and \
       ("numer" in ua_var_types or not ua_var_types or "any" in ua_var_types) and len(numerical_cols) >= 2:
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                suggestions.append({"name": "Scatter Plot", "for_cols": f"{col1} & {col2}", "type": "Bivariate Numerical-Numerical", "reason": f"Shows relationship between '{col1}' and '{col2}'.", "required_cols_specific": [col1, col2]})

    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
       ("compare" in ua_message or "across categories" in ua_message or "group by" in ua_message or "distribution" in ua_message) and \
       ("mix" in ua_var_types or "categor" in ua_var_types or "numer" in ua_var_types or not ua_var_types or "any" in ua_var_types) and numerical_cols and categorical_cols:
        for num_col in numerical_cols:
            for cat_col in categorical_cols:
                suggestions.append({"name": "Box Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares distribution of '{num_col}' across '{cat_col}' categories.", "required_cols_specific": [cat_col, num_col]})
                suggestions.append({"name": "Violin Plots (by Category)", "for_cols": f"{num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares distribution (with density) of '{num_col}' across '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})
                suggestions.append({"name": "Bar Chart (Aggregated)", "for_cols": f"Avg/Sum of {num_col} by {cat_col}", "type": "Bivariate Numerical-Categorical", "reason": f"Compares average/sum of '{num_col}' for each category in '{cat_col}'.", "required_cols_specific": [cat_col, num_col]})

    if ("two" in ua_var_count_str or "2" in ua_var_count_str) and \
        ("relationship" in ua_message or "compare" in ua_message or "contingency" in ua_message or "joint" in ua_message) and \
        ("categor" in ua_var_types or not ua_var_types or "any" in ua_var_types) and len(categorical_cols) >= 2:
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                suggestions.append({"name": "Grouped Bar Chart", "for_cols": f"{col1} & {col2}", "type": "Bivariate Categorical-Categorical", "reason": f"Shows counts of '{col1}' grouped by '{col2}'.", "required_cols_specific": [col1, col2]})
                suggestions.append({"name": "Heatmap (Counts)", "for_cols": f"{col1} & {col2}", "type": "Bivariate Categorical-Categorical", "reason": f"Shows co-occurrence frequency of '{col1}' and '{col2}'.", "required_cols_specific": [col1, col2]})

    if (("two" in ua_var_count_str or "2" in ua_var_count_str) or "time" in ua_var_types or "trend" in ua_message) and \
       datetime_cols and numerical_cols:
        for dt_col in datetime_cols:
            for num_col in numerical_cols:
                suggestions.append({"name": "Line Chart", "for_cols": f"{num_col} over {dt_col}", "type": "Time Series", "reason": f"Shows trend of '{num_col}' over '{dt_col}'.", "required_cols_specific": [dt_col, num_col]})
                suggestions.append({"name": "Area Chart", "for_cols": f"{num_col} over {dt_col}", "type": "Time Series", "reason": f"Shows cumulative trend/magnitude of '{num_col}' over '{dt_col}'.", "required_cols_specific": [dt_col, num_col]})

    if ("more" in ua_var_count_str or "multiple" in ua_var_count_str or "pair plot" in ua_message or "heatmap" in ua_message or "parallel" in ua_message) or \
       ((ua_var_count_str not in ["one", "1", "two", "2"]) and (len(numerical_cols) > 2 or len(categorical_cols) > 2) ):
        if len(numerical_cols) >= 3: 
            suggestions.append({"name": "Pair Plot", "type": "Multivariate", "reason": "Shows pairwise relationships between numerical variables.", "required_cols_specific": numerical_cols[:min(4, len(numerical_cols))] })
            suggestions.append({"name": "Correlation Heatmap", "type": "Multivariate", "reason": "Shows correlation matrix for numerical variables.", "required_cols_specific": numerical_cols })
        if len(numerical_cols) >=3:
            suggestions.append({"name": "Parallel Coordinates Plot", "type": "Multivariate", "reason": "Compares multiple numerical variables across records.", "required_cols_specific": numerical_cols[:min(6, len(numerical_cols))]})

    final_suggestions_dict = {}
    for s in suggestions:
        s_key = s["name"] + ("_" + "_".join(sorted(s["required_cols_specific"])) if "required_cols_specific" in s else "")
        if s_key not in final_suggestions_dict:
            final_suggestions_dict[s_key] = s
    final_suggestions = list(final_suggestions_dict.values())
    
    if not final_suggestions:
        final_suggestions.append({"name": "No specific chart matched well", "type": "Info", "reason": "Your criteria didn't closely match common chart types. You can try picking columns manually.", "required_cols_specific": []})
    
    manual_pick_exists = any(s['name'] == "Pick columns manually" for s in final_suggestions)
    if not manual_pick_exists:
        final_suggestions.append({"name": "Pick columns manually", "type": "Action", "reason": "If you have a specific chart in mind or want to explore freely.", "required_cols_specific": []})
    return final_suggestions

def generate_plot_and_get_uri(filepath, chart_type, columns):
    if not filepath:
        print("Error: Filepath for plotting is missing.")
        return None
    try:
        df_full = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading full dataframe ('{filepath}') for plotting: {e}")
        return None

    # Validate columns exist
    if not columns: # Check if columns list is empty
        print(f"Error: No columns provided for plotting chart type '{chart_type}'.")
        return None
    if not all(col in df_full.columns for col in columns):
        missing_cols = [col for col in columns if col not in df_full.columns]
        print(f"Error: One or more specified columns not found in dataframe for plotting: {missing_cols}. Available: {df_full.columns.tolist()}")
        return None

    img = io.BytesIO()
    plt.figure(figsize=(7, 4.5)) 
    plt.style.use('seaborn-v0_8-whitegrid')

    try:
        print(f"Attempting to generate plot: {chart_type} with columns: {columns}") # Debug print
        if chart_type == "Histogram" and len(columns) == 1:
            sns.histplot(data=df_full, x=columns[0], kde=True)
            plt.title(f"Histogram of {columns[0]}", fontsize=12)
        elif chart_type == "Box Plot" and len(columns) == 1:
            sns.boxplot(data=df_full, y=columns[0])
            plt.title(f"Box Plot of {columns[0]}", fontsize=12)
        elif chart_type == "Density Plot" and len(columns) == 1:
            sns.kdeplot(data=df_full, x=columns[0], fill=True)
            plt.title(f"Density Plot of {columns[0]}", fontsize=12)
        elif chart_type == "Bar Chart (Counts)" and len(columns) == 1:
            counts = df_full[columns[0]].value_counts().nlargest(15)
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f"Counts for {columns[0]}", fontsize=12)
            plt.ylabel("Count", fontsize=10)
            plt.xlabel(columns[0], fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Pie Chart" and len(columns) == 1:
            counts = df_full[columns[0]].value_counts()
            effective_counts = counts.nlargest(6)
            if len(counts) > 6:
                effective_counts.loc['Other'] = counts.iloc[6:].sum()
            plt.pie(effective_counts, labels=effective_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.title(f"Pie Chart of {columns[0]}", fontsize=12)
            plt.axis('equal')
        elif chart_type == "Scatter Plot" and len(columns) == 2:
            sns.scatterplot(data=df_full, x=columns[0], y=columns[1])
            plt.title(f"Scatter: {columns[0]} vs {columns[1]}", fontsize=12)
            plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10)
        elif chart_type == "Line Chart" and len(columns) == 2:
            df_to_plot = df_full.copy()
            if pd.api.types.is_datetime64_any_dtype(df_to_plot[columns[0]]):
                 df_to_plot = df_to_plot.sort_values(by=columns[0])
            elif pd.api.types.is_numeric_dtype(df_to_plot[columns[0]]):
                 df_to_plot = df_to_plot.sort_values(by=columns[0])
            sns.lineplot(data=df_to_plot, x=columns[0], y=columns[1])
            plt.title(f"Line: {columns[1]} over {columns[0]}", fontsize=12)
            plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Box Plots (by Category)" and len(columns) == 2: 
            # Expects columns[0] as categorical, columns[1] as numerical for sns.boxplot x, y
            sns.boxplot(data=df_full, x=columns[0], y=columns[1])
            plt.title(f"Box Plots: {columns[1]} by {columns[0]}", fontsize=12)
            plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Violin Plots (by Category)" and len(columns) == 2:
            sns.violinplot(data=df_full, x=columns[0], y=columns[1])
            plt.title(f"Violin Plots: {columns[1]} by {columns[0]}", fontsize=12)
            plt.xlabel(columns[0], fontsize=10); plt.ylabel(columns[1], fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
        elif chart_type == "Bar Chart (Aggregated)" and len(columns) == 2:
            agg_data = df_full.groupby(columns[0])[columns[1]].mean().nlargest(15)
            sns.barplot(x=agg_data.index, y=agg_data.values)
            plt.title(f"Mean of {columns[1]} by {columns[0]}", fontsize=12)
            plt.xlabel(columns[0], fontsize=10); plt.ylabel(f"Mean of {columns[1]}", fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=9)
        else:
            print(f"Plot type '{chart_type}' with {len(columns)} columns not implemented or cols mismatch.")
            fig, ax = plt.subplots() # Use current figure context if available
            ax.text(0.5, 0.5, f"Plot type '{chart_type}'\nnot implemented or\ncolumn mismatch for\n{', '.join(columns)}",
                    ha='center', va='center', fontsize=9, wrap=True)
            ax.axis('off')

        plt.tight_layout(pad=1.0)
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close() 
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        print(f"Successfully generated plot: {chart_type}") # Debug print
        return f"data:image/png;base64,{plot_url}"

    except Exception as e:
        print(f"!!! Error during plot generation for '{chart_type}' with {columns}: {e}")
        if 'plt' in locals() and plt.get_fignums(): 
            plt.close('all') 
        return None


@app.route("/")
def home():
    session.pop('visualization_questions_state', None)
    session.pop('uploaded_filepath', None)
    session.pop('uploaded_filename', None)
    session.pop('df_columns', None)
    session.pop('user_answer_variable_types', None)
    session.pop('user_answer_visualization_message', None)
    session.pop('user_answer_variable_count', None)
    session.pop('chart_suggestions_list', None)
    session.pop('selected_chart_for_plotting', None)
    session.pop('plotting_columns', None)
    session.pop('manual_columns_selected', None)
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
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

    if current_viz_state == 'asking_variable_types':
        session['user_answer_variable_types'] = user_input
        bot_reply = (
            f"Understood. You're working with: \"{user_input}\".<br><br>"
            f"<strong>2. What message or insight do you want your visualization to communicate?</strong>"
            f"<br>(e.g., compare values, show distribution, identify relationships, track trends)"
        )
        session['visualization_questions_state'] = 'asking_visualization_message'
        response_data = {
            "suggestions": [
                "Compare values/categories", "Show data distribution", 
                "Identify relationships", "Track trends over time"
            ]
        }
    elif current_viz_state == 'asking_visualization_message':
        session['user_answer_visualization_message'] = user_input
        columns_list_str = ""
        if df_columns:
            columns_list_str = "<ul>" + "".join([f"<li>{col}</li>" for col in df_columns]) + "</ul>"
            columns_reminder = f"For reference, columns in <strong>{session.get('uploaded_filename', 'your file')}</strong> include: {columns_list_str}"
        else:
            columns_reminder = "<p>(Could not retrieve column list from the uploaded file.)</p>"
        bot_reply = (
            f"Great! The goal is to: \"{user_input}\".<br><br>"
            f"<strong>3. How many variables would you typically like to visualize in a single chart?</strong>"
            f"<br>(e.g., one, two, three, or 'more' for multivariate)<br><br>"
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
            df_sample = None
            if uploaded_filepath:
                try: 
                    df_sample = pd.read_csv(uploaded_filepath, nrows=100) if uploaded_filepath.endswith(".csv") else pd.read_excel(uploaded_filepath, nrows=100)
                except Exception as e:
                    print(f"Error reading df_sample for suggestions: {e}")
            
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
                     if any(s['name'] == manual_pick_option for s in chart_suggestions):
                        suggestions_for_user_options.append(manual_pick_option)

                response_data_suggestions = suggestions_for_user_options
                if manual_pick_option not in response_data_suggestions: # Ensure it's an option if not already suggested
                    response_data_suggestions.append(manual_pick_option)
                response_data_suggestions.append("Restart visualization questions")

                response_data = {"suggestions": response_data_suggestions[:5]} 
                session['visualization_questions_state'] = 'awaiting_chart_type_selection'
            else:
                bot_reply = "I couldn't come up with specific chart suggestions right now. You can try picking columns manually."
                response_data = {"suggestions": ["Let me pick columns", "Restart visualization questions"]}
                session['visualization_questions_state'] = 'awaiting_column_selection_general'

        elif "choose columns" in user_input.lower() or "pick columns" in user_input.lower():
            bot_reply = "Sure! Which columns would you like to visualize? Available columns:<br><ul>" + "".join([f"<li>{col}</li>" for col in df_columns]) + "</ul>Please list them (e.g., 'Age, Salary')."
            session['visualization_questions_state'] = 'awaiting_column_selection_general'
            session['manual_columns_selected'] = [] 
            response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}
        
        elif "restart" in user_input.lower():
            session.pop('user_answer_variable_types', None); session.pop('user_answer_visualization_message', None); session.pop('user_answer_variable_count', None)
            session.pop('chart_suggestions_list', None); session.pop('selected_chart_for_plotting', None); session.pop('plotting_columns', None); session.pop('manual_columns_selected', None)
            bot_reply = "No problem, let's start over.<br><br><strong>1. What types of variables...</strong>"
            session['visualization_questions_state'] = 'asking_variable_types'
            response_data = {"suggestions": ["Categorical (text, groups)", "Numerical (numbers, counts)", "Time-series (dates/times)", "A mix of types"]}
        else: 
            bot_reply = "Sorry, I didn't catch that. What would you like to do next?"
            response_data = { "suggestions": session.get('last_suggestions', ["Suggest chart types for me", "Let me choose columns to plot", "Restart these visualization questions"]) }

    elif current_viz_state == 'awaiting_chart_type_selection':
        user_choice_str = user_input.replace("Select: ", "").strip()
        chart_suggestions_list = session.get('chart_suggestions_list', [])
        selected_chart_info = next((sugg for sugg in chart_suggestions_list if sugg['name'] == user_choice_str), None)

        if user_choice_str == "Pick columns manually":
            bot_reply = "Okay! Which columns from your file (<strong>" + ", ".join(df_columns) + "</strong>) would you like to visualize? Please list them."
            session['visualization_questions_state'] = 'awaiting_column_selection_general'
            session['manual_columns_selected'] = []
            response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}
        elif selected_chart_info:
            session['selected_chart_for_plotting'] = selected_chart_info
            chart_name = selected_chart_info['name']
            required_cols_specific = selected_chart_info.get('required_cols_specific')

            bot_reply = f"Great, let's prepare a <strong>{chart_name}</strong>. "
            if required_cols_specific and isinstance(required_cols_specific, list) and len(required_cols_specific) > 0 : 
                cols_to_use_str = ", ".join(required_cols_specific)
                bot_reply += f"I suggest using these columns based on your data: <strong>{cols_to_use_str}</strong>. Proceed with these?"
                session['plotting_columns'] = required_cols_specific 
                response_data = {"suggestions": [f"Yes, plot {chart_name} with these", "Let me choose other columns for " + chart_name, "Back to chart suggestions"]}
                session['visualization_questions_state'] = 'confirm_plot_details' 
            else: 
                bot_reply += f"Which columns would you like to use for the {chart_name}? Available: {', '.join(df_columns)}"
                session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart'
                response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns", "Back to chart suggestions"]}
        else:
            bot_reply = "I didn't recognize that selection. Please choose from the suggestions or ask to pick columns manually."
            response_data = {"suggestions": session.get('last_suggestions', [])} 

    elif current_viz_state == 'confirm_plot_details':
        chart_to_plot_info = session.get('selected_chart_for_plotting')
        cols_for_plot = session.get('plotting_columns')

        if user_input.startswith("Yes, plot"):
            if chart_to_plot_info and cols_for_plot and uploaded_filepath:
                plot_image_uri = generate_plot_and_get_uri(uploaded_filepath, chart_to_plot_info['name'], cols_for_plot)
                if plot_image_uri:
                    bot_reply = f"Here is the <strong>{chart_to_plot_info['name']}</strong> for columns: {', '.join(cols_for_plot)}."
                    response_data["plot_image"] = plot_image_uri
                else:
                    bot_reply = f"Sorry, an error occurred while generating the <strong>{chart_to_plot_info['name']}</strong>."
                session['visualization_questions_state'] = None 
                response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart visualization questions", "Upload new data"])
            else:
                bot_reply = "Something went wrong with the plot details or file path. Let's try again."
                session['visualization_questions_state'] = 'visualization_info_gathered'
                response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot"]}

        elif "choose other columns" in user_input.lower():
            bot_reply = f"Okay, for the <strong>{chart_to_plot_info['name'] if chart_to_plot_info else 'chart'}</strong>, which columns would you like to use instead? Available: {', '.join(df_columns)}"
            session['visualization_questions_state'] = 'awaiting_columns_for_selected_chart' 
            response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns"]}
        else: 
            bot_reply = "Okay, let's go back. What would you like to do?"
            session['visualization_questions_state'] = 'visualization_info_gathered'
            response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot", "Restart visualization questions"]}

    elif current_viz_state == 'awaiting_columns_for_selected_chart':
        potential_cols_str = user_input.replace("Use:", "").strip()
        user_selected_cols = [col.strip() for col in potential_cols_str.split(',') if col.strip() and col.strip() in df_columns] 
        
        chart_to_plot_info = session.get('selected_chart_for_plotting')
        chart_name = chart_to_plot_info.get('name', 'chart') if chart_to_plot_info else 'chart'

        if user_selected_cols: 
            session['plotting_columns'] = user_selected_cols
            bot_reply = f"Got it. For the <strong>{chart_name}</strong>, you've selected: <strong>{', '.join(user_selected_cols)}</strong>. Ready to plot?"
            response_data = {"suggestions": ["Yes, generate this plot", "Add/Change columns", "Back to chart suggestions"]}
            session['visualization_questions_state'] = 'confirm_plot_details'
        elif "list all columns" in user_input.lower():
            bot_reply = "Available columns are: " + ", ".join(df_columns) + f"<br>Which ones for the {chart_name}?"
            response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]]}
        elif "back to chart suggestions" in user_input.lower():
            session['visualization_questions_state'] = 'awaiting_chart_type_selection' # Corrected: go back to chart type selection
            bot_reply = "Okay, which chart type were you interested in from the suggestions?"
            # Reshow chart suggestions (this assumes chart_suggestions_list is still in session and relevant)
            # This part might need refinement to perfectly reshow previous chart suggestions.
            chart_suggestions_list = session.get('chart_suggestions_list', [])
            temp_suggs = [f"Select: {s['name']}" for s in chart_suggestions_list if s.get("type") != "Action"][:4]
            temp_suggs.append("Pick columns manually")
            response_data = {"suggestions": temp_suggs}
        else:
            bot_reply = f"I couldn't understand those column names or they aren't valid. For <strong>{chart_name}</strong>, please choose from: {', '.join(df_columns)}"
            response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["List all columns", "Back to chart suggestions"]}

    elif current_viz_state == 'awaiting_column_selection_general':
        if "finished selecting columns" in user_input.lower():
            selected_cols = session.get('manual_columns_selected', [])
            if selected_cols:
                bot_reply = f"Okay, you've selected: <strong>{', '.join(selected_cols)}</strong>. What kind of chart would you like to create with these columns?"
                session['plotting_columns'] = selected_cols
                session['visualization_questions_state'] = 'awaiting_chart_type_for_manual_cols'
                chart_type_suggestions = ["Bar Chart", "Scatter Plot", "Line Chart", "Histogram", "Box Plot", "Table View"] # Generic for now
                response_data = {"suggestions": chart_type_suggestions}
            else:
                bot_reply = "You haven't selected any columns yet. Please list some columns to use, or click 'Cancel selection'."
                response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}
        elif "cancel selection" in user_input.lower():
            session.pop('manual_columns_selected', None)
            session['visualization_questions_state'] = 'visualization_info_gathered'
            bot_reply = "Okay, column selection cancelled. What would you like to do?"
            response_data = {"suggestions": ["Suggest chart types for me", "Let me choose columns to plot"]}
        else:
            potential_col = user_input.replace("Use:","").strip()
            current_selection = session.get('manual_columns_selected', [])
            if potential_col in df_columns:
                if potential_col not in current_selection:
                    current_selection.append(potential_col)
                session['manual_columns_selected'] = current_selection
                bot_reply = f"Added '<strong>{potential_col}</strong>'. Currently selected: <strong>{', '.join(current_selection) if current_selection else 'None'}</strong>.<br>Add more, or click 'Finished selecting columns'."
                remaining_cols_suggestions = [f"Use: {col}" for col in df_columns if col not in current_selection][:2]
                response_data = {"suggestions": remaining_cols_suggestions + ["Finished selecting columns", "Cancel selection"]}
            else:
                bot_reply = f"'{potential_col}' is not a valid column. Please choose from: {', '.join(df_columns)}."
                response_data = {"suggestions": [f"Use: {col}" for col in df_columns[:2]] + ["Finished selecting columns", "Cancel selection"]}

    elif current_viz_state == 'awaiting_chart_type_for_manual_cols':
        chart_type_from_user = user_input.strip()
        cols_for_plot = session.get('plotting_columns', [])
        
        if cols_for_plot and uploaded_filepath:
            plot_image_uri = generate_plot_and_get_uri(uploaded_filepath, chart_type_from_user, cols_for_plot)
            if plot_image_uri:
                bot_reply = f"Here is the <strong>{chart_type_from_user}</strong> for columns: {', '.join(cols_for_plot)}."
                response_data["plot_image"] = plot_image_uri
            else:
                bot_reply = f"Sorry, an error occurred while generating the <strong>{chart_type_from_user}</strong> with the selected columns. They might not be suitable for this chart type."
        else:
            bot_reply = "Something went wrong with the column selection or file path. Let's try again."

        session['visualization_questions_state'] = None 
        response_data.setdefault("suggestions", []).extend(["Suggest another chart", "Restart visualization questions", "Upload new data"])

    else: 
        nlu_output = nlu_get_bot_response(user_input) 
        if isinstance(nlu_output, dict):
            bot_reply = nlu_output.get("response", "Sorry, I had trouble understanding that.")
            temp_suggestions = nlu_output.get("suggestions", [])
        else: 
            bot_reply = nlu_output
            temp_suggestions = []
        if not temp_suggestions:
            temp_suggestions = ["Help", "Upload Data", "What can you do?"]
        response_data = {"suggestions": temp_suggestions}

    response_data["response"] = bot_reply
    session['last_suggestions'] = response_data.get("suggestions", [])
    
    # print("--- DEBUG START ---")
    # print(f"Handled Viz State (this turn): {current_viz_state}")
    # print(f"Next Viz State: {session.get('visualization_questions_state')}")
    # print(f"Outgoing Bot Reply: {bot_reply[:200]}...") 
    # print(f"Outgoing Suggestions: {response_data.get('suggestions')}")
    # print(f"Plot Image in Response: {'plot_image' in response_data}")
    # print("--- DEBUG END ---")
    
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
        home() 

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            session['uploaded_filepath'] = filepath
            session['uploaded_filename'] = filename

            df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_excel(filepath)
            session['df_columns'] = list(df.columns) 
            preview_html = df.head(5).to_html(classes="preview-table", index=False, border=0)
            total_rows, total_columns = len(df), len(df.columns)
            missing_values = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            total_cells = total_rows * total_columns
            missing_percent = (missing_values / total_cells) * 100 if total_cells else 0

            initial_bot_message = (
                f"✅ <strong>{filename}</strong> uploaded and previewed.<br><br>"
                f"🔍 Data Quality Check:<br>"
                f"- Rows: <strong>{total_rows}</strong>, Columns: <strong>{total_columns}</strong><br>"
                f"- Missing values: <strong>{missing_values}</strong> ({missing_percent:.2f}%)<br>"
                f"- Duplicate rows: <strong>{duplicate_rows}</strong><br><br>"
                f"Great! To suggest visualizations, I have a few questions:<br><br>"
                f"<strong>1. What types of variables are you working with primarily?</strong>"
                f"<br>(e.g., categorical/text, numerical, dates/times)"
            )
            session['visualization_questions_state'] = 'asking_variable_types' 
            current_suggestions = [
                "Categorical (text, groups)", "Numerical (numbers, counts)",
                "Time-series (dates/times)", "A mix of these types", "Not sure / Any"
            ]
            session['last_suggestions'] = current_suggestions
            return jsonify({
                "response": initial_bot_message, "preview": preview_html,
                "suggestions": current_suggestions
            })
        except Exception as e:
            home() 
            return jsonify({"response": f"Uploaded '{filename}', but error processing: {str(e)}"}), 500
    else:
        return jsonify({"response": "File type not allowed. Please upload CSV or Excel."}), 400

if __name__ == "__main__":
    app.run(debug=True)