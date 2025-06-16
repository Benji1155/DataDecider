# This file contains the detailed explanations for common data visualizations.

VISUALIZATION_DETAILS = {
    "📊 Bar Chart": {
        "title": "📊 Bar Chart: Comparing Categories",
        "description": "A Bar Chart uses rectangular bars to show comparisons between discrete categories. The length of each bar is proportional to the value it represents, making it easy to see which category is biggest or smallest.",
        "when_to_use": [
            "Comparing numerical data across different groups (e.g., sales per country).",
            "Showing the frequency or count of items in different categories (e.g., number of students in each school house).",
            "When you have a limited number of categories to compare (usually fewer than 12)."
        ],
        "example": "Imagine you have sales data for different products (laptops, phones, tablets). A bar chart would be perfect to quickly see which product is the top seller.",
        "best_for": "Comparing values across distinct categories."
    },
    "📈 Line Chart": {
        "title": "📈 Line Chart: Tracking Trends Over Time",
        "description": "A Line Chart displays information as a series of data points connected by straight lines. It's the best way to visualize data that changes over a continuous interval, like time.",
        "when_to_use": [
            "Tracking changes and trends over a period (e.g., stock prices over a month).",
            "Comparing how multiple groups change over the same time period.",
            "When your horizontal axis (x-axis) represents time, distance, or another continuous variable."
        ],
        "example": "You could track a city's average monthly temperature over a year to easily spot the summer peak and winter low.",
        "best_for": "Showing trends and changes over a continuous period."
    },
    "🥧 Pie Chart": {
        "title": "🥧 Pie Chart: Showing Proportions",
        "description": "A Pie Chart is a circular graph divided into slices to illustrate numerical proportion. Each slice's size shows its percentage of the whole. <strong>Warning:</strong> Use with caution, as they can be misleading!",
        "when_to_use": [
            "You want to show how different parts make up a whole (100%).",
            "You have a very small number of categories (best for 2-5 categories).",
            "The values for each slice are very different from each other."
        ],
        "example": "Showing the percentage breakdown of votes in a simple two-candidate election (e.g., Candidate A got 60%, Candidate B got 40%).",
        "best_for": "Illustrating a simple part-to-whole relationship for a few categories."
    },
    "📊 Histogram": {
        "title": "📊 Histogram: Understanding Data Distribution",
        "description": "A Histogram looks like a Bar Chart, but it's used for numerical data. It groups numbers into ranges (called 'bins'), and the height of each bar shows how many data points fall into that range.",
        "when_to_use": [
            "Understanding the distribution (or 'shape') of a single numerical variable.",
            "Identifying the center, spread, and whether the data is symmetrical or skewed.",
            "Checking for gaps or outliers in your data."
        ],
        "example": "Visualizing the distribution of heights of students in a school to see if most students are clustered around an average height, forming a bell curve.",
        "best_for": "Showing the frequency and distribution of numerical data."
    },
    "📈 Scatter Plot": {
        "title": "📈 Scatter Plot: Investigating Relationships",
        "description": "A Scatter Plot uses dots to represent the values for two different numeric variables. The position of each dot on the horizontal and vertical axes indicates its values for those two variables.",
        "when_to_use": [
            "Investigating the relationship or correlation between two numerical variables.",
            "Identifying patterns like a positive relationship (as one goes up, the other goes up), a negative relationship, or no relationship.",
            "Spotting outliers that don't fit the general pattern."
        ],
        "example": "Plotting a person's hours of study against their exam score to see if there is a positive relationship between them.",
        "best_for": "Showing the relationship between two numerical variables."
    }
}
