# test_explanations.py

STATISTICAL_TEST_DETAILS = {
    "📈 Time-series decomposition": {
        "title": "📈 Time-Series Decomposition: Unpacking Your Time Data",
        "description": "This technique breaks down your time-ordered data into its core components, helping you see underlying patterns more clearly.",
        "when_to_use": [
            "You have data collected over regular time intervals (e.g., daily sales, monthly temperatures).",
            "You want to understand the main <strong>trend</strong> (long-term direction).",
            "You want to see if there are <strong>seasonal patterns</strong> (repeating cycles, e.g., higher sales in summer).",
            "You want to identify the <strong>random noise</strong> or irregularities in your data."
        ],
        "key_assumptions": [
            "Data is ordered by time.",
            "The chosen model (additive or multiplicative) fits the data reasonably well."
        ],
        "basic_idea": "Imagine your data is like a song. Decomposition tries to separate the main melody (trend), the repeating rhythm (seasonality), and any unexpected notes (residuals/noise).",
        "formula_simple": "Conceptual Models:<br>- Additive: Data = Trend + Seasonality + Residual<br>- Multiplicative: Data = Trend * Seasonality * Residual",
        "example": "Analyzing monthly ice cream sales: Decomposition might show an upward trend over years, a seasonal peak in summer, and some random fluctuations month-to-month.",
        "interpretation": "Look at the separate plots for trend, seasonality, and residuals. The trend shows the overall direction. Seasonality shows regular cycles. Residuals show what's left over after accounting for trend and seasonality."
    },
    "🔮 ARIMA / SARIMA models": {
        "title": "🔮 ARIMA/SARIMA Models: Forecasting the Future",
        "description": "These are powerful models used for analyzing and forecasting time series data. ARIMA stands for AutoRegressive Integrated Moving Average. SARIMA adds a seasonal component.",
        "when_to_use": [
            "You have time-ordered data and want to <strong>predict future values</strong>.",
            "Your data shows some form of auto-correlation (past values influence future values).",
            "The data needs to be <strong>stationary</strong> (its statistical properties like mean and variance don't change over time) or can be made stationary through differencing."
        ],
        "key_assumptions": [
            "Data is stationary (or can be made so).",
            "No missing values (or they need to be handled)."
        ],
        "basic_idea": "ARIMA models look at past values (AutoRegressive part), past forecast errors (Moving Average part), and how many times the data needs to be 'differenced' to become stationary (Integrated part). SARIMA does this for seasonal patterns too.",
        "formula_simple": "Involves complex equations based on parameters (p, d, q) for non-seasonal and (P, D, Q)m for seasonal parts. These define the lags and differencing.",
        "example": "Predicting next month's website traffic based on the traffic patterns of previous months and years.",
        "interpretation": "The model provides forecasts with confidence intervals. You also assess model fit using metrics like AIC, BIC, and by checking if residuals look like random noise."
    },
    "📊 Mann-Kendall Test": {
        "title": "📊 Mann-Kendall Test: Spotting Trends Over Time",
        "description": "This is a non-parametric test used to detect a monotonic (consistently increasing or decreasing) trend in time series data. It doesn't assume the data follows a bell curve.",
        "when_to_use": [
            "You have data collected over time.",
            "You want to know if there's a <strong>consistent upward or downward trend</strong> (not necessarily a straight line).",
            "You don't want to assume your data is normally distributed."
        ],
        "key_assumptions": [
            "Data points are independent over time (after accounting for any seasonality if present).",
            "There is only one data point per time period."
        ],
        "basic_idea": "It compares every data point to all subsequent data points. It counts how many times later values are higher than earlier values and vice-versa. A strong imbalance suggests a trend.",
        "formula_simple": "Involves calculating a test statistic 'S' based on the signs of differences between data points, and then a standardized 'Z' score.",
        "example": "Analyzing annual average temperatures for a city over 50 years to see if there's a significant warming or cooling trend.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) indicates a statistically significant monotonic trend (either increasing or decreasing, the direction is also given by the test).<br>  - A <strong>large p-value</strong> suggests no significant monotonic trend."
    },
    "🤝 Cross-correlation (CCF)": {
        "title": "🤝 Cross-Correlation (CCF): How Two Time Series Dance Together",
        "description": "Cross-correlation measures the similarity between two time series as a function of the lag (time shift) of one relative to the other. It helps see if one series leads or lags the other.",
        "when_to_use": [
            "You have <strong>two time series</strong> data sets.",
            "You want to see if they are related and, if so, if changes in one tend to precede changes in the other."
        ],
        "key_assumptions": [
            "Both series should ideally be <strong>stationary</strong> for clearer interpretation."
        ],
        "basic_idea": "It's like sliding one time series past the other and calculating the correlation at each shift. The highest correlation value indicates the lag at which the two series are most similar.",
        "formula_simple": "Calculates correlation coefficients for different lags (time shifts) between the two series.",
        "example": "Comparing monthly advertising spend with monthly sales to see if increased ad spend leads to increased sales a month or two later.",
        "interpretation": "A plot of CCF values against lags is usually examined. A significant peak at a positive lag means the first series leads the second. A peak at a negative lag means the second series leads the first. A peak at lag 0 means they change together."
    },
    "🔗 Granger Causality Test": {
        "title": "🔗 Granger Causality Test: Can One Time Series Predict Another?",
        "description": "This statistical hypothesis test is used to determine whether one time series is useful in forecasting another. It's about predictive power, not necessarily true cause-and-effect.",
        "when_to_use": [
            "You have <strong>two (or more) time series</strong>.",
            "You want to know if past values of one series help predict future values of another series, beyond what the second series' own past values can predict."
        ],
        "key_assumptions": [
            "Both time series must be <strong>stationary</strong>.",
            "There should be a linear relationship between the variables."
        ],
        "basic_idea": "It essentially checks if including past values of series X improves the prediction of series Y, compared to predicting Y using only its own past values. If it does, X is said to 'Granger-cause' Y.",
        "formula_simple": "Involves comparing the fit of regression models (autoregressive models) with and without the lagged values of the other series.",
        "example": "Testing if past changes in the unemployment rate help predict future changes in GDP growth.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests that series X Granger-causes series Y (i.e., past X helps predict Y).<br>  - A <strong>large p-value</strong> suggests no significant Granger causality."
    },
    "🧑‍🔬 One-sample t-test": {
        "title": "🧑‍🔬 One-Sample t-Test: Checking a Group's Average",
        "description": "This test helps you see if the average (mean) of a single group of numbers is significantly different from a specific, known value or a hypothesized value (like a target or a historical average).",
        "when_to_use": [
            "You have <strong>one group</strong> of numerical data (e.g., test scores for one class, weights of a sample of products).",
            "You want to compare the <strong>average</strong> of this group to a specific number (e.g., is this class's average score different from the national average of 75?).",
            "Your data for the group is approximately <strong>normally distributed</strong> (looks like a bell curve). If it's far from normal, especially with small samples, the Wilcoxon signed-rank test is a better choice."
        ],
        "key_assumptions": [
            "Your data is <strong>continuous</strong> (numerical, can take many values).",
            "Data points are <strong>independent</strong> (one data point doesn't influence another).",
            "Data is approximately <strong>normally distributed</strong>.",
            "You have a <strong>random sample</strong> from the population you're interested in."
        ],
        "basic_idea": "It calculates a 't-statistic'. Think of this as a signal-to-noise ratio: how big is the difference between your sample's average and the known value, compared to the natural variation (spread) in your sample data? A larger t-statistic (further from zero) suggests the difference is more likely to be real.",
        "formula_simple": "Conceptually: <br>t = (Sample Average - Hypothesized Average) / (Sample Standard Deviation / &radic;(Sample Size))<br><br>Where:<br>- <strong>Sample Average (Mean)</strong>: The average of your data.<br>- <strong>Hypothesized Average</strong>: The specific value you're comparing against.<br>- <strong>Sample Standard Deviation</strong>: How spread out your data points are.<br>- <strong>Sample Size</strong>: Your number of data points.",
        "example": "Imagine you're a coffee shop owner and you want your new espresso machine to dispense 30ml shots on average. You take 20 shots, measure them, and find their average is 28.5ml with some variation. A one-sample t-test could help you determine if this 28.5ml is significantly different from your 30ml target, or if it's likely just random variation.",
        "interpretation": "The test gives you a <strong>p-value</strong> (Probability value).<br>  - If the p-value is <strong>small</strong> (commonly < 0.05, meaning less than a 5% chance the difference is random), it suggests the difference you observed is 'statistically significant'. You might conclude your group's average is indeed different from the known value.<br>  - If the p-value is <strong>large</strong> (>= 0.05), it suggests the difference could just be due to random chance, and there isn't strong evidence that your group's average is truly different from the known value."
    },
    "✍️ Wilcoxon signed-rank test": { 
        "title": "✍️ Wilcoxon Signed-Rank Test: Non-Bell Curve Check",
        "description": "A 'non-parametric' test. Use it to compare a single group's median to a specific value OR to compare two related (paired) groups, especially when your data doesn't look like a bell curve (not normally distributed).",
        "when_to_use": [
            "<strong>For one group:</strong> Comparing the <strong>median</strong> of your group to a known or hypothesized value (e.g., is the median customer rating different from 3.5?).",
            "<strong>For two paired groups:</strong> Comparing measurements taken from the <strong>same subjects at two different times</strong> or under two different conditions (e.g., before vs. after treatment scores).",
            "Your data (or the differences between pairs) is <strong>not normally distributed</strong>, or you have a small sample."
        ],
        "key_assumptions": [
            "Data is <strong>continuous or ordinal</strong> (can be ranked).",
            "For one sample: Distribution of differences from the hypothesized median is symmetrical.",
            "For paired samples: Distribution of the differences between pairs is symmetrical.",
            "Observations are independent (or pairs are independent of other pairs)."
        ],
        "basic_idea": "It looks at the differences. For one sample, it's the difference from the hypothesized median. For paired samples, it's the difference within each pair. It then ranks these differences by size (ignoring the sign), and sums up the ranks for positive differences and negative differences. A big imbalance in these sums suggests a significant difference.",
        "formula_simple": "Involves ranking differences and summing them. No simple algebraic formula like a t-test.",
        "example": "One Sample: Testing if the median weight loss on a diet is 5kg, when weight loss data isn't normally distributed.<br>Paired: Testing if a training program improved employees' scores on a test, by comparing their scores before and after the program, when the score differences aren't normally distributed.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a statistically significant difference in medians.<br>  - A <strong>large p-value</strong> suggests not enough evidence to say the medians are different."
    },
     "➕ Sign Test": {
        "title": "➕ Sign Test: Simple Comparison for One Group or Pairs",
        "description": "A very simple non-parametric test. For one group, it checks if values tend to be above or below a specific median. For paired data, it checks if one member of the pair tends to be larger than the other.",
        "when_to_use": [
            "<strong>For one group:</strong> When you want to test if the median of your group is different from a hypothesized value, and you make very few assumptions about your data's shape.",
            "<strong>For two paired groups:</strong> When you want to see if there's a consistent direction of difference within pairs (e.g., is 'after' usually greater than 'before'?).",
            "Data is not normally distributed, and even the Wilcoxon test's symmetry assumption might not hold."
        ],
        "key_assumptions": [
            "Data is at least <strong>ordinal</strong> (can be ranked).",
            "Observations are independent (or pairs are independent)."
        ],
        "basic_idea": "It only looks at the signs of the differences. For one sample, it counts how many values are above (+) and below (-) the hypothesized median. For paired data, it counts how many pairs have a positive difference and how many have a negative difference. It then sees if these counts are too imbalanced to be due to chance.",
        "formula_simple": "Based on binomial distribution probabilities of observing a certain number of pluses or minuses.",
        "example": "One Sample: A restaurant claims the median wait time is 10 minutes. You record 20 wait times and count how many are above and below 10 minutes.<br>Paired: Comparing if students prefer teaching method A or B by asking each student which they prefer and counting the preferences.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> suggests a significant tendency for values to be above/below the hypothesized median (one sample) or for one paired observation to be consistently larger/smaller than the other."
    },
    "📊 One-proportion z-test": {
        "title": "📊 One-Proportion z-Test: Checking a Single Percentage",
        "description": "This test is used to see if the percentage (proportion) of a certain characteristic in a single group is significantly different from a known or hypothesized percentage.",
        "when_to_use": [
            "You have <strong>one group</strong> with categorical data where each item either has a characteristic or doesn't (e.g., success/failure, yes/no).",
            "You want to compare the <strong>observed proportion</strong> in your sample to a specific expected proportion (e.g., is the proportion of defective items in your batch different from the 2% target?).",
            "You have a reasonably <strong>large sample size</strong> (typically np > 10 and n(1-p) > 10, where n is sample size and p is hypothesized proportion)."
        ],
        "key_assumptions": [
            "Data is <strong>categorical (binary outcome)</strong>.",
            "Observations are <strong>independent</strong>.",
            "Sample size is large enough for the normal approximation to the binomial distribution."
        ],
        "basic_idea": "It calculates how many standard deviations your sample proportion is away from the hypothesized proportion, assuming the hypothesized proportion is true. If it's many standard deviations away, the difference is likely real.",
        "formula_simple": "Conceptually: <br>z = (Sample Proportion - Hypothesized Proportion) / Standard Error of the Proportion <br><br>The Standard Error depends on the hypothesized proportion and sample size.",
        "example": "A politician claims 60% of voters support them. You survey 200 voters and find 55% support. This test helps see if your 55% is significantly different from their 60% claim.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests your sample proportion is significantly different from the hypothesized proportion.<br>  - A <strong>large p-value</strong> suggests the difference could be due to random sampling variation."
    },
    "🎲 Chi-square goodness-of-fit test": {
        "title": "🎲 Chi-Square Goodness-of-Fit Test: Do Your Counts Match Expectations?",
        "description": "This test checks if the observed frequencies (counts) in different categories of a single categorical variable match the frequencies you would expect based on a specific theory or known distribution.",
        "when_to_use": [
            "You have <strong>one categorical variable</strong> with two or more categories.",
            "You have <strong>expected counts or proportions</strong> for each category.",
            "You want to see if your observed counts significantly differ from these expectations."
        ],
        "key_assumptions": [
            "Data is <strong>categorical</strong>.",
            "Observations are <strong>independent</strong>.",
            "Expected frequency in each category should generally be <strong>at least 5</strong> for the test to be reliable."
        ],
        "basic_idea": "It compares the counts you actually see in each category (observed) with the counts you'd expect to see (expected). If the differences are large across categories, the test suggests your data doesn't fit the expected distribution.",
        "formula_simple": "Calculates a Chi-square (χ²) statistic: <br>χ² = &Sigma; [ (Observed Count - Expected Count)² / Expected Count ] for all categories.<br><br>A larger χ² value indicates a bigger mismatch.",
        "example": "You roll a die 60 times. You'd expect each number (1-6) to appear 10 times if the die is fair. You observe the actual counts (e.g., '1' appeared 8 times, '2' appeared 13 times, etc.). This test sees if your observed counts are significantly different from the expected 10 for each.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests your observed counts are significantly different from the expected counts; your data doesn't fit the hypothesized distribution.<br>  - A <strong>large p-value</strong> suggests your observed counts are consistent with the expected counts."
    },
    "🧑‍🔬 Independent t-test": {
        "title": "🧑‍🔬 Independent t-Test: Comparing Averages of Two Separate Groups",
        "description": "This test is used to see if the averages (means) of a numerical variable are significantly different between two separate, unrelated (independent) groups.",
        "when_to_use": [
            "You have <strong>two independent groups</strong> (e.g., a control group and a treatment group, men vs. women).",
            "You are comparing the <strong>average</strong> of a numerical measurement between these two groups (e.g., average test scores of students using two different study methods).",
            "The data in <strong>both groups</strong> is approximately <strong>normally distributed</strong> (bell-curved).",
            "The <strong>variances (spread)</strong> of the data in both groups are roughly <strong>equal</strong> (Levene's test can check this). If not, a variation called Welch's t-test is often used."
        ],
        "key_assumptions": [
            "The dependent variable (what you're measuring) is <strong>continuous</strong>.",
            "The two groups are <strong>independent</strong> (observations in one group don't affect the other).",
            "Data in each group is approximately <strong>normally distributed</strong>.",
            "<strong>Homogeneity of variances</strong> (variances are similar between groups)."
        ],
        "basic_idea": "It calculates a t-statistic that represents the difference between the two group averages relative to the variability within the groups. A larger t-statistic suggests a more significant difference between the groups.",
        "formula_simple": "Conceptually (for equal variances): <br>t = (Average of Group 1 - Average of Group 2) / (Pooled Standard Error)<br><br>The 'Pooled Standard Error' combines the spread and sample sizes of both groups.",
        "example": "You want to see if a new drug (Group A) results in lower blood pressure compared to a placebo (Group B). You measure blood pressure for patients in each group and compare the averages.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests there's a statistically significant difference between the averages of the two groups.<br>  - A <strong>large p-value</strong> suggests there's not enough evidence to say the averages are truly different (any observed difference could be due to chance)."
    },
    "📊 Mann–Whitney U test": {
        "title": "📊 Mann–Whitney U Test (or Wilcoxon Rank-Sum): Comparing Two Unrelated Groups (Non-Bell Curve)",
        "description": "This is a non-parametric test used to compare whether there is a difference in the central tendency (often medians or distributions) between two independent groups when the data is not normally distributed (not bell-curved).",
        "when_to_use": [
            "You have <strong>two independent (unrelated) groups</strong>.",
            "You are comparing a <strong>continuous or ordinal</strong> measurement between these two groups.",
            "The data in one or both groups is <strong>not normally distributed</strong>, or you have small sample sizes."
        ],
        "key_assumptions": [
            "Observations from both groups are <strong>independent</strong>.",
            "The data can be at least <strong>ranked (ordinal)</strong>.",
            "To interpret it as a test of medians, the shapes of the distributions in both groups should be similar (though not necessarily normal)."
        ],
        "basic_idea": "It ranks all the data from both groups together, from smallest to largest. Then, it sums the ranks for each group. If one group consistently has much lower or higher ranks than the other, the test suggests the groups are different.",
        "formula_simple": "Involves calculating a 'U' statistic based on the sum of ranks for one of the groups. This is then often converted to a z-score for larger samples.",
        "example": "Comparing the customer satisfaction ratings (on a 1-10 scale, which is ordinal and might not be normal) between customers who used Product A and customers who used Product B.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a statistically significant difference between the two groups (e.g., their medians or overall distributions differ).<br>  - A <strong>large p-value</strong> suggests not enough evidence to say the groups are truly different."
    },
    "⚖️ Two-proportion z-test": {
        "title": "⚖️ Two-Proportion z-Test: Comparing Percentages of Two Groups",
        "description": "This test compares the proportions (percentages) of a certain characteristic or outcome between two independent groups.",
        "when_to_use": [
            "You have <strong>two independent groups</strong>.",
            "You have <strong>categorical data (binary outcome)</strong> for each group (e.g., success/failure, yes/no).",
            "You want to see if the <strong>proportion</strong> of the outcome is different between the two groups.",
            "Sample sizes are reasonably large in both groups."
        ],
        "key_assumptions": [
            "Data is <strong>categorical (binary)</strong>.",
            "The two samples are <strong>independent</strong>.",
            "Sample sizes are large enough (typically, at least 10 successes and 10 failures expected in each group under the null hypothesis of no difference)."
        ],
        "basic_idea": "It calculates the difference between the two sample proportions and compares it to what you'd expect if there were no real difference, considering the sample sizes.",
        "formula_simple": "Conceptually: <br>z = (Proportion Group 1 - Proportion Group 2) / Standard Error of the Difference in Proportions",
        "example": "Comparing the click-through rate (proportion of clicks) of two different website ad designs (Ad A vs. Ad B) to see if one performs significantly better.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a statistically significant difference between the proportions of the two groups.<br>  - A <strong>large p-value</strong> suggests any observed difference could be due to random chance."
    },
    "🎲 Chi-square test of independence": { 
        "title": "🎲 Chi-Square Test of Independence: Are Two Categories Related?",
        "description": "This test is used to determine if there's a significant association (relationship) between two categorical variables. It checks if the occurrence of one variable's categories depends on the other's.",
        "when_to_use": [
            "You have <strong>two categorical variables</strong> (either nominal or ordinal, though it treats them as nominal).",
            "You have data in a <strong>contingency table</strong> (cross-tabulation) showing the counts for each combination of categories.",
            "You want to see if the variables are independent or if there's a relationship between them."
        ],
        "key_assumptions": [
            "Data is <strong>categorical</strong>.",
            "Observations are <strong>independent</strong>.",
            "Expected frequency in each cell of the contingency table should generally be <strong>at least 5</strong> (for larger tables, some cells can be a bit lower, but not too many)."
        ],
        "basic_idea": "It compares the observed counts in your contingency table cells to the counts you would expect if the two variables were totally unrelated (independent). If the observed counts are far from the expected counts, it suggests a relationship.",
        "formula_simple": "Calculates a Chi-square (χ²) statistic: <br>χ² = &Sigma; [ (Observed Count - Expected Count)² / Expected Count ] for all cells in the table.",
        "example": "Investigating if there's a relationship between a person's favorite color (e.g., Red, Blue, Green) and their preferred type of movie (e.g., Action, Comedy, Drama).",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests there is a statistically significant association between the two categorical variables (they are not independent).<br>  - A <strong>large p-value</strong> suggests there's not enough evidence to say the variables are related."
    },
    "🎣 Fisher’s Exact Test": {
        "title": "🎣 Fisher’s Exact Test: Precise Test for Small 2x2 Tables",
        "description": "This test is used to determine if there are non-random associations between two categorical variables in a 2x2 contingency table, especially when sample sizes are small (making Chi-square less reliable).",
        "when_to_use": [
            "You have <strong>two categorical variables</strong>, each with <strong>two categories</strong> (forming a 2x2 table).",
            "Your <strong>sample size is small</strong>, or some cells in your 2x2 table have very low expected counts (e.g., less than 5).",
            "You want to test for independence or association between the two variables."
        ],
        "key_assumptions": [
            "Data is <strong>categorical (binary for both variables)</strong>.",
            "Observations are <strong>independent</strong>.",
            "Row and column totals are considered fixed (this is a technical point related to how the probability is calculated)."
        ],
        "basic_idea": "It calculates the exact probability of observing your specific 2x2 table (and any tables more extreme), given the row and column totals, if the two variables were truly independent. No approximations are made.",
        "formula_simple": "Involves calculating probabilities using factorials based on the cell counts and marginal totals of the 2x2 table.",
        "example": "Testing if a new, rare drug treatment (yes/no) is associated with patient recovery (yes/no) in a very small clinical trial with only 10 patients.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a statistically significant association between the two variables.<br>  - A <strong>large p-value</strong> suggests no significant association."
    },
    "🧑‍🔬 Paired t-test": {
        "title": "🧑‍🔬 Paired t-Test: Comparing Two Related Measurements",
        "description": "This test is used to compare the means of two related samples or measurements. This typically involves measuring the same subject/item at two different points in time (e.g., before and after an intervention) or under two different conditions.",
        "when_to_use": [
            "You have <strong>two related (paired) sets of numerical data</strong> (e.g., a 'before' score and an 'after' score for each participant).",
            "You want to see if there's a significant <strong>difference in the averages</strong> between these two paired measurements.",
            "The <strong>differences</strong> between the paired measurements are approximately <strong>normally distributed</strong> (bell-curved)."
        ],
        "key_assumptions": [
            "The dependent variable is <strong>continuous</strong>.",
            "The observations are <strong>paired</strong> (e.g., same subject measured twice).",
            "The differences between the paired observations are approximately <strong>normally distributed</strong>.",
            "The pairs are a <strong>random sample</strong> from the population of pairs."
        ],
        "basic_idea": "It calculates the difference for each pair, and then performs a one-sample t-test on these differences to see if their average difference is significantly different from zero.",
        "formula_simple": "Essentially a one-sample t-test on the differences: <br>t = (Average of Differences) / (Standard Deviation of Differences / &radic;(Number of Pairs))",
        "example": "Measuring students' test scores before a new teaching method and then again after the method is implemented. The paired t-test checks if there's a significant average change in scores.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a statistically significant mean difference between the paired measurements.<br>  - A <strong>large p-value</strong> suggests no strong evidence of a true mean difference."
    },
     "🔄 McNemar's Test": {
        "title": "🔄 McNemar's Test: Tracking Changes in Paired Categories",
        "description": "This test is used for paired categorical data (specifically binary, yes/no type data) to see if there's a significant change in proportions between two related time points or conditions.",
        "when_to_use": [
            "You have <strong>paired observations</strong> (e.g., same person assessed twice).",
            "The outcome variable is <strong>categorical with two levels</strong> (binary, e.g., pass/fail, agree/disagree).",
            "You want to see if there's a significant shift in responses from the first measurement to the second."
        ],
        "key_assumptions": [
            "Data is <strong>paired and binary</strong>.",
            "Sample size is reasonably large (though it's more robust than Chi-square for small paired samples)."
        ],
        "basic_idea": "It focuses only on the pairs where there was a change in response (e.g., Yes then No, or No then Yes). It tests if the number of changes in one direction is significantly different from the number of changes in the other direction.",
        "formula_simple": "Based on a Chi-square statistic calculated from the discordant pairs (those that changed).",
        "example": "Surveying voters before and after a political debate to see if their preference for Candidate A (yes/no) significantly changed.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a significant change in proportions between the two paired measurements (i.e., the intervention or time had an effect).<br>  - A <strong>large p-value</strong> suggests no significant change."
    },
    " ANOVA (One-way)": {
        "title": " ANOVA (One-Way): Comparing Averages of 3+ Groups",
        "description": "ANOVA (Analysis of Variance) is used to compare the means (averages) of a numerical variable across three or more independent groups to see if at least one group mean is significantly different from the others.",
        "when_to_use": [
            "You have <strong>three or more independent groups</strong>.",
            "You are comparing the <strong>average</strong> of a numerical measurement across these groups.",
            "The data in <strong>each group</strong> is approximately <strong>normally distributed</strong>.",
            "The <strong>variances (spread)</strong> of the data are roughly <strong>equal</strong> across all groups (homogeneity of variances)."
        ],
        "key_assumptions": [
            "The dependent variable is <strong>continuous</strong>.",
            "The groups are <strong>independent</strong>.",
            "Data within each group is approximately <strong>normally distributed</strong>.",
            "<strong>Homogeneity of variances</strong> across groups."
        ],
        "basic_idea": "It compares the variation *between* the group averages to the variation *within* each group. If the variation between groups is much larger than the variation within groups, it suggests that at least one group mean is different.",
        "formula_simple": "Calculates an F-statistic: <br>F = (Variance Between Groups) / (Variance Within Groups)",
        "example": "Comparing the average plant height resulting from three different types of fertilizer to see if any fertilizer produces significantly different heights.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests that there is a statistically significant difference in at least one of the group means. It doesn't tell you *which* groups are different, only that a difference exists somewhere.<br>  - A <strong>large p-value</strong> suggests no significant difference among the group means. <br>If significant, you'd use <strong>post-hoc tests</strong> to find out which specific groups differ."
    },
    "🔍 Post-hoc tests (e.g., Tukey's HSD)": {
        "title": "🔍 Post-Hoc Tests (e.g., Tukey's HSD, Bonferroni): Pinpointing Group Differences After ANOVA",
        "description": "If an ANOVA test tells you that there *is* a significant difference somewhere among your 3+ group means, post-hoc tests are used to find out *which specific pairs* of groups are significantly different from each other.",
        "when_to_use": [
            "You have performed an <strong>ANOVA</strong> and found a <strong>statistically significant result</strong> (small p-value).",
            "You want to make pairwise comparisons between group means while controlling the overall error rate (to avoid falsely finding differences just by doing many t-tests)."
        ],
        "key_assumptions": ["Same as for ANOVA."],
        "basic_idea": "Different post-hoc tests use different methods to adjust the p-values for multiple comparisons. Tukey's HSD (Honestly Significant Difference) is common for comparing all possible pairs. Bonferroni is more conservative.",
        "formula_simple": "Each test has its own specific calculation, often based on t-statistics or q-statistics, adjusted for the number of comparisons.",
        "example": "After finding that fertilizer type significantly affects plant height (via ANOVA), Tukey's HSD could tell you if Fertilizer A is significantly different from B, A from C, and B from C.",
        "interpretation": "Each pairwise comparison will have its own p-value or confidence interval. Look for small p-values (or confidence intervals that don't include zero) to identify which specific group means are significantly different."
    },
    "📊 Kruskal–Wallis test": {
        "title": "📊 Kruskal–Wallis Test: Comparing 3+ Groups (Non-Bell Curve)",
        "description": "This is a non-parametric alternative to one-way ANOVA. It's used to compare the medians (or general distributions) of a numerical or ordinal variable across three or more independent groups when the data is not normally distributed.",
        "when_to_use": [
            "You have <strong>three or more independent groups</strong>.",
            "You are comparing a <strong>continuous or ordinal</strong> measurement across these groups.",
            "The data in one or more groups is <strong>not normally distributed</strong>, or you have small sample sizes."
        ],
        "key_assumptions": [
            "Observations from all groups are <strong>independent</strong>.",
            "The data can be at least <strong>ranked (ordinal)</strong>.",
            "For interpreting as a test of medians, the shapes of the distributions in all groups should be similar."
        ],
        "basic_idea": "It ranks all the data from all groups together. Then, it compares the average rank for each group. If the average ranks are very different, it suggests the groups are not from the same overall distribution.",
        "formula_simple": "Calculates an H-statistic, which is approximately Chi-square distributed.",
        "example": "Comparing customer satisfaction ratings (1-5 scale) for a product across three different store locations, where ratings data might not be normal.",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests a statistically significant difference in at least one of the group's medians/distributions.<br>  - A <strong>large p-value</strong> suggests no strong evidence of a difference. <br>If significant, use <strong>post-hoc tests</strong> (like Dunn's test) to find which specific groups differ."
    },
    "🔍 Post-hoc tests (e.g., Dunn's test)": {
        "title": "🔍 Post-Hoc Tests (e.g., Dunn's Test): Pinpointing Group Differences After Kruskal-Wallis",
        "description": "If a Kruskal-Wallis test indicates a significant difference among 3+ groups (for non-normal data), Dunn's test (or similar non-parametric post-hoc tests) is used to find out which specific pairs of groups are significantly different.",
        "when_to_use": [
            "You have performed a <strong>Kruskal-Wallis test</strong> and found a <strong>statistically significant result</strong>.",
            "You want to make pairwise comparisons between group medians/distributions while controlling for multiple comparisons."
        ],
        "key_assumptions": ["Same as for Kruskal-Wallis test."],
        "basic_idea": "Dunn's test compares the average ranks between pairs of groups, adjusting p-values for the number of comparisons.",
        "formula_simple": "Involves rank sums and specific calculations for the test statistic.",
        "example": "After finding that store location significantly affects customer satisfaction ratings (via Kruskal-Wallis), Dunn's test could tell you if Location A's ratings are significantly different from Location B's, A from C, and B from C.",
        "interpretation": "Each pairwise comparison will have an adjusted p-value. Look for small p-values to identify which specific group medians/distributions are significantly different."
    },
    "📈 Pearson Correlation (r)": {
        "title": "📈 Pearson Correlation (r): Strength of Straight-Line Relationship",
        "description": "Measures the strength and direction of a linear (straight-line) relationship between two continuous numerical variables. The result 'r' ranges from -1 to +1.",
        "when_to_use": [
            "You have <strong>two continuous numerical variables</strong> (e.g., height and weight, temperature and ice cream sales).",
            "You want to see if they tend to increase or decrease together in a <strong>straight-line pattern</strong>.",
            "Both variables should ideally be <strong>normally distributed</strong> for the p-value to be most accurate, though 'r' itself can be calculated regardless."
        ],
        "key_assumptions": [
            "Both variables are <strong>continuous</strong>.",
            "The relationship is <strong>linear</strong> (a scatter plot should roughly show a line, not a curve).",
            "Data is approximately <strong>bivariate normal</strong> (for significance testing)."
        ],
        "basic_idea": "It quantifies how well the data points fit onto a straight line. <br>+1: Perfect positive linear relationship (as one goes up, the other goes up perfectly).<br>-1: Perfect negative linear relationship (as one goes up, the other goes down perfectly).<br> 0: No linear relationship.",
        "formula_simple": "Calculated based on the covariance of the two variables, divided by the product of their standard deviations.",
        "example": "Checking the relationship between hours spent studying and exam scores. A positive 'r' would suggest more study time is associated with higher scores.",
        "interpretation": "The <strong>value of r</strong> indicates strength and direction. Its <strong>p-value</strong> indicates if this correlation is statistically significant (unlikely to be due to random chance).<br>  - Common interpretations for |r|: 0.1-0.3 (weak), 0.3-0.5 (moderate), >0.5 (strong)."
    },
    "📉 Simple Linear Regression": {
        "title": "📉 Simple Linear Regression: Predicting One Variable from Another",
        "description": "This technique models the linear relationship between two continuous variables: one independent (predictor, X) variable and one dependent (outcome, Y) variable. It allows you to predict Y based on X.",
        "when_to_use": [
            "You have <strong>two continuous numerical variables</strong>.",
            "You believe one variable (X) can <strong>predict or explain</strong> the other (Y).",
            "You assume the relationship between them is approximately a <strong>straight line</strong>."
        ],
        "key_assumptions": [
            "<strong>Linear relationship</strong> between X and Y.",
            "<strong>Independence of errors</strong> (residuals).",
            "<strong>Homoscedasticity</strong> (residuals have constant variance).",
            "<strong>Normality of errors</strong> (residuals are normally distributed)."
        ],
        "basic_idea": "It finds the best-fitting straight line (Y = b0 + b1*X) through your data points. 'b0' is the intercept (where the line crosses the Y-axis) and 'b1' is the slope (how much Y changes for a one-unit change in X).",
        "formula_simple": "Line equation: Y = Intercept + (Slope * X)",
        "example": "Predicting a person's weight (Y) based on their height (X). The regression would give you an equation to make this prediction.",
        "interpretation": "Look at the <strong>slope coefficient (b1)</strong>: its sign indicates direction, and its p-value indicates if it's significantly different from zero (i.e., if X is a significant predictor). <strong>R-squared (R²)</strong> tells you the percentage of variation in Y that's explained by X."
    },
    "📊 Spearman Rank Correlation (rho)": {
        "title": "📊 Spearman Rank Correlation (rho): Relationship Strength (Non-Bell Curve / Ordered Data)",
        "description": "This non-parametric test measures the strength and direction of a monotonic relationship between two ranked variables. It's useful when your data isn't normally distributed or is ordinal (ranked).",
        "when_to_use": [
            "You have <strong>two variables that can be ranked</strong> (either continuous data that's not normal, or ordinal data like 'low, medium, high').",
            "You want to see if they tend to <strong>increase or decrease together consistently</strong>, even if not in a perfect straight line."
        ],
        "key_assumptions": [
            "Data is at least <strong>ordinal</strong> (can be ranked).",
            "Observations are paired."
        ],
        "basic_idea": "It converts the values of each variable into ranks, and then calculates Pearson correlation on these ranks. It assesses how well the relationship between two variables can be described using a monotonic function.",
        "formula_simple": "Essentially Pearson correlation applied to the ranks of the data.",
        "example": "Comparing the rank order of students in a math test with their rank order in a science test to see if performance is similarly ranked across subjects, even if the scores themselves aren't normally distributed.",
        "interpretation": "The Spearman correlation coefficient (rho, ρ) ranges from -1 to +1, similar to Pearson's r. Its <strong>p-value</strong> indicates if the correlation is statistically significant."
    },
    "🔗 Kendall's Tau": {
        "title": "🔗 Kendall's Tau: Measuring Agreement in Rankings",
        "description": "Another non-parametric measure of rank correlation. It assesses the strength of association between two ordinal variables based on the number of concordant (pairs ranked in the same order) and discordant (pairs ranked in opposite orders) pairs.",
        "when_to_use": [
            "You have <strong>two ordinal (ranked) variables</strong>.",
            "You want to measure the degree of similarity in the ordering of the data by the two variables.",
            "Often preferred over Spearman's rho for smaller sample sizes or when there are many tied ranks."
        ],
        "key_assumptions": [
            "Data is at least <strong>ordinal</strong>.",
            "Observations are paired."
        ],
        "basic_idea": "It looks at all possible pairs of observations and counts how many pairs are concordant (both variables rank one observation higher than the other) versus discordant. Tau-b and Tau-c are variations that handle ties differently.",
        "formula_simple": "Calculated based on (Number of Concordant Pairs - Number of Discordant Pairs) / (Total Number of Pairs - adjustments for ties).",
        "example": "Two judges rank 10 contestants in a competition. Kendall's Tau can measure the agreement between the two judges' rankings.",
        "interpretation": "Ranges from -1 (perfect disagreement) to +1 (perfect agreement). Its <strong>p-value</strong> indicates if the association is statistically significant."
    },
     "🔗 Cramer's V": {
        "title": "🔗 Cramer's V: Strength of Connection for Categories",
        "description": "After a Chi-square test of independence shows there *is* a relationship between two nominal categorical variables, Cramer's V measures how strong that relationship is. It ranges from 0 (no association) to 1 (perfect association).",
        "when_to_use": [
            "You have performed a <strong>Chi-square test of independence</strong> on a contingency table (e.g., 2x2, 3x2, etc.) and found a significant result.",
            "You want to quantify the <strong>strength or magnitude</strong> of the association between the two nominal categorical variables."
        ],
        "key_assumptions": ["Same as for the Chi-square test of independence."],
        "basic_idea": "It's a measure of association derived from the Chi-square statistic, adjusted for sample size and the dimensions of the contingency table.",
        "formula_simple": "V = &radic;( (Chi-square statistic / Sample Size) / (min(Number of Rows - 1, Number of Columns - 1)) )",
        "example": "If a Chi-square test shows a significant relationship between 'Region' and 'Product Preference', Cramer's V would tell you how strong that link is (e.g., a V of 0.1 is weak, 0.3 moderate, 0.5+ strong).",
        "interpretation": "Ranges from 0 to 1. Closer to 1 indicates a stronger association between the two categorical variables. There's no p-value for Cramer's V itself; the significance comes from the preceding Chi-square test."
    },
    "🔗 Point Biserial Correlation": {
        "title": "🔗 Point Biserial Correlation: Linking Continuous and Two-Category Variables",
        "description": "Measures the strength and direction of the association between a continuous variable and a dichotomous (two-category) categorical variable.",
        "when_to_use": [
            "You have <strong>one continuous numerical variable</strong> (e.g., test score, income).",
            "You have <strong>one categorical variable with only two groups</strong> (e.g., pass/fail, male/female, treatment/control).",
            "You want to see if there's a relationship between the numerical variable and group membership."
        ],
        "key_assumptions": [
            "The continuous variable is approximately <strong>normally distributed within each of the two categories</strong>.",
            "<strong>Equal variances</strong> of the continuous variable across the two categories (homoscedasticity)."
        ],
        "basic_idea": "It's mathematically equivalent to calculating a Pearson correlation if you code the two categories of the dichotomous variable as 0 and 1.",
        "formula_simple": "Can be calculated using the means and standard deviation of the continuous variable for each of the two categories, and the overall standard deviation.",
        "example": "Examining the relationship between hours spent studying (continuous) and whether a student passed or failed an exam (dichotomous).",
        "interpretation": "The coefficient ranges from -1 to +1. Its square (r²) indicates the proportion of variance in the continuous variable explained by group membership. A <strong>p-value</strong> indicates if the correlation is statistically significant."
    },
     "ANOVA (as a model)": { 
        "title": " ANOVA (as a Model): Continuous vs. Categorical Relationship",
        "description": "When looking at the relationship between one continuous variable and one categorical variable (with two or more groups/levels), ANOVA can be used. It tests if the average of the continuous variable differs significantly across the categories of the categorical variable.",
        "when_to_use": [
            "You have <strong>one continuous numerical variable</strong> (the outcome or dependent variable).",
            "You have <strong>one categorical variable</strong> (the predictor or independent variable) that defines two or more groups.",
            "You want to see if the mean of the continuous variable is different depending on the category."
        ],
        "key_assumptions": [
            "The continuous variable is approximately <strong>normally distributed within each category</strong>.",
            "<strong>Homogeneity of variances</strong> (the spread of the continuous variable is similar across categories).",
            "Observations are <strong>independent</strong>."
        ],
        "basic_idea": "It's essentially the same as a One-Way ANOVA used for comparing group means. It partitions the total variability in the continuous variable into variability between the groups (due to the categorical variable) and variability within the groups (random error).",
        "formula_simple": "Calculates an F-statistic: F = (Variance Between Groups) / (Variance Within Groups)",
        "example": "Investigating if average income (continuous) differs significantly across different education levels (categorical: High School, Bachelor's, Master's, PhD).",
        "interpretation": "Gives a <strong>p-value</strong>.<br>  - A <strong>small p-value</strong> (e.g., < 0.05) suggests that the categorical variable has a significant effect on the mean of the continuous variable (i.e., the means are different across at least some categories).<br>  - If significant, <strong>post-hoc tests</strong> can identify which specific categories differ."
    },
    "🧩 Multiple Linear Regression": {
        "title": "🧩 Multiple Linear Regression: Predicting with Several Factors",
        "description": "This technique models the linear relationship between one continuous dependent (outcome) variable and two or more independent (predictor) variables. Predictors can be continuous or categorical (if properly coded).",
        "when_to_use": [
            "You want to <strong>predict a continuous numerical outcome</strong>.",
            "You have <strong>two or more predictor variables</strong> that you believe influence the outcome.",
            "You assume the relationships are generally <strong>linear</strong>."
        ],
        "key_assumptions": [
            "<strong>Linear relationship</strong> between predictors and the outcome.",
            "<strong>Independence of errors</strong> (residuals).",
            "<strong>Homoscedasticity</strong> (residuals have constant variance).",
            "<strong>Normality of errors</strong> (residuals are normally distributed).",
            "<strong>No strong multicollinearity</strong> (predictors are not too highly correlated with each other)."
        ],
        "basic_idea": "It finds the best-fitting linear equation (Y = b0 + b1*X1 + b2*X2 + ... + bn*Xn) to describe how the predictors together influence the outcome. Each 'b' coefficient represents the change in Y for a one-unit change in its corresponding X, holding other X's constant.",
        "formula_simple": "Equation: Y = Intercept + (Coef1 * Var1) + (Coef2 * Var2) + ...",
        "example": "Predicting a house's sale price (Y) based on its size (X1), number of bedrooms (X2), and age (X3).",
        "interpretation": "Look at the <strong>coefficients</strong> for each predictor: their p-values indicate if they are significant predictors. <strong>R-squared (R²)</strong> tells you the percentage of variation in the outcome variable that's explained by all predictors in the model combined."
    },
    "🧩 Logistic Regression (Binary/Multinomial)": {
        "title": "🧩 Logistic Regression: Predicting Categories",
        "description": "Used to predict the probability of a categorical outcome (with two or more categories) based on one or more predictor variables (which can be continuous or categorical).",
        "when_to_use": [
            "Your <strong>outcome variable is categorical</strong> (e.g., yes/no, pass/fail, choice A/B/C).",
            "You have one or more predictor variables.",
            "You want to understand how predictors influence the likelihood of falling into a particular category."
        ],
        "key_assumptions": [
            "<strong>Independence of errors.</strong>",
            "For binary logistic regression, the dependent variable is dichotomous (two categories).",
            "For multinomial, the dependent variable has more than two unordered categories.",
            "Linearity between the log-odds of the outcome and continuous predictors."
        ],
        "basic_idea": "Instead of predicting the outcome directly, it predicts the probability (specifically, the log-odds) of the outcome occurring. This probability is then transformed to be between 0 and 1.",
        "formula_simple": "Involves a logistic (sigmoid) function to model probabilities. For binary: log(p/(1-p)) = b0 + b1*X1 + ...",
        "example": "Predicting whether a customer will click on an ad (yes/no) based on their age, browsing history, and time of day. Or predicting which product a customer will choose (Product A, B, or C) based on their demographics.",
        "interpretation": "Coefficients are often interpreted as <strong>odds ratios</strong>. An odds ratio > 1 means the predictor increases the odds of the outcome. An odds ratio < 1 means it decreases the odds. P-values indicate significance of predictors. Model fit can be assessed with metrics like pseudo-R-squared or Hosmer-Lemeshow test."
    },
    "🧩 Ordinal Logistic Regression": {
        "title": "🧩 Ordinal Logistic Regression: Predicting Ordered Categories",
        "description": "Used when your outcome variable is categorical AND its categories have a natural order (ordinal), and you want to predict this outcome based on one or more predictor variables.",
        "when_to_use": [
            "Your <strong>outcome variable is ordinal</strong> (e.g., satisfaction ratings like 'low, medium, high', education levels 'High School, Bachelor's, Master's').",
            "You have one or more predictor variables.",
            "You want to understand how predictors influence the likelihood of being in a higher (or lower) ordered category."
        ],
        "key_assumptions": [
            "The dependent variable is <strong>ordinal</strong>.",
            "<strong>Proportional odds assumption</strong> (or parallel lines assumption) is often made, meaning the effect of predictors is consistent across the different thresholds between categories."
        ],
        "basic_idea": "It models the cumulative probability of falling into a category or below. It's more complex than binary logistic regression because it has to account for the ordering of the categories.",
        "formula_simple": "Involves modeling cumulative log-odds. Several variations exist.",
        "example": "Predicting a student's final grade (A, B, C, D, F) based on hours studied, attendance, and prior GPA.",
        "interpretation": "Coefficients indicate how predictors shift the odds of being in a higher category. P-values assess predictor significance. Model fit is checked with various tests."
    },
    "📊 Descriptive Statistics (Mean, Median, Std Dev, etc.)": {
        "title": "📊 Descriptive Statistics: Summarizing Your Data",
        "description": "These are basic measures used to describe the main features of a dataset, such as its central point, how spread out the values are, and the shape of its distribution.",
        "when_to_use": [
            "You want to get a <strong>quick overview</strong> of a variable.",
            "To understand the <strong>typical value</strong> (e.g., mean, median, mode).",
            "To understand the <strong>variability or spread</strong> (e.g., standard deviation, range, interquartile range).",
            "To check for <strong>skewness or outliers</strong>."
        ],
        "key_assumptions": ["Generally, none for calculation, but interpretation depends on data type (e.g., mean is best for symmetric numerical data)."],
        "basic_idea": "Calculating simple summary numbers and creating basic plots (like histograms) to understand your data before doing more complex tests.",
        "formula_simple": "Examples:<br>- Mean: Sum of values / Number of values<br>- Median: The middle value when data is sorted<br>- Standard Deviation: A measure of how much values typically deviate from the mean.",
        "example": "Calculating the average age, the most common occupation, and the range of sleep durations in your dataset.",
        "interpretation": "Provides a foundational understanding of each variable's characteristics."
    },
    "📊 Frequency Table / Counts / Percentages": {
        "title": "📊 Frequency Tables: Counting Categories",
        "description": "Used for categorical data to show how many times each category appears (frequency) and what percentage of the total each category represents.",
        "when_to_use": [
            "You have <strong>categorical data</strong> (nominal or ordinal).",
            "You want to see the <strong>distribution of observations</strong> across the different categories."
        ],
        "key_assumptions": ["None."],
        "basic_idea": "Simply counting occurrences for each category.",
        "formula_simple": "Count occurrences of each category. Percentage = (Count for Category / Total Count) * 100.",
        "example": "For a 'Gender' column, counting how many 'Male' and 'Female' entries there are and their percentages. For 'Occupation', listing each job and how many people have it.",
        "interpretation": "Shows which categories are most/least common and their relative proportions."
    },
    "❓ No specific test matched perfectly": { 
        "title": "❓ Unsure Which Test Fits Best?",
        "description": "Sometimes your question or data might not perfectly align with a standard test, or it might be a more complex scenario.",
        "when_to_use": ["This message appears if the bot couldn't confidently match your answers to a common statistical test."],
        "key_assumptions": ["N/A"],
        "basic_idea": ["It's always good to double-check your understanding of your data and your research question."],
        "formula_simple": "N/A",
        "example": ["If you're trying to combine many different types of analyses or have very unusual data distributions."],
        "interpretation": ["Consider rephrasing your question to the bot, simplifying your analysis goal, or consulting a statistician or a more detailed statistical guide for advanced scenarios."]
    }
}