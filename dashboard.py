import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, silhouette_score

# Load data for all graphs
monthly_data = pd.read_csv('shopee_sales_report_2022_to_2024.csv')
real_time_data = pd.read_csv('sales_report_real_time.csv')
stock_data = pd.read_csv('22_product_quantities_random_restock_limited_no_stock.csv')

# Convert DateTime column to datetime format in monthly data
monthly_data['DateTime'] = pd.to_datetime(monthly_data['DateTime'], format='%d/%m/%Y %H:%M')

# Convert Time column in real-time data to seconds for real-time revenue updates
real_time_data['Time'] = pd.to_datetime(real_time_data['Time'], format='%H:%M:%S')
real_time_data['Seconds'] = real_time_data['Time'].dt.hour * 3600 + real_time_data['Time'].dt.minute * 60 + \
                            real_time_data['Time'].dt.second


# Display monthly revenue by product category using a selected chart type
def monthly_revenue_by_category():
    # Convert 'DateTime' to a monthly period and back to timestamp format for consistency
    monthly_data['YearMonth'] = monthly_data['DateTime'].dt.to_period('M').dt.to_timestamp()

    # Group data by product category and month, summing the revenue for each group
    grouped_data = monthly_data.groupby(['Product Category', 'YearMonth'])['Revenue (MYR)'].sum().reset_index()

    # Allow user to select the chart type (Line Chart, Bar Chart, or Area Chart)
    chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Area Chart"])

    # Generate a line chart with points for monthly revenue by product category
    if chart_type == "Line Chart":
        line = alt.Chart(grouped_data).mark_line().encode(
            x=alt.X('YearMonth:T', title='Month', axis=alt.Axis(format="%b %Y", labelAngle=-90, tickCount="month")),
            y='Revenue (MYR):Q',
            color='Product Category:N',  # Color by product category
            tooltip=['YearMonth:T', 'Revenue (MYR):Q', 'Product Category:N']  # Tooltip for interactive details
        )
        points = line.mark_point()  # Add points to highlight each month's revenue on the line chart
        chart = line + points  # Combine line and points for the final chart

    # Generate a bar chart for monthly revenue by product category
    elif chart_type == "Bar Chart":
        chart = alt.Chart(grouped_data).mark_bar().encode(
            x=alt.X('YearMonth:T', title='Month', axis=alt.Axis(format="%b %Y", labelAngle=-90, tickCount="month")),
            y='Revenue (MYR):Q',
            color='Product Category:N',
            tooltip=['YearMonth:T', 'Revenue (MYR):Q', 'Product Category:N']
        ).interactive()  # Enable interactivity for bar chart (e.g., tooltip display on hover)

    # Generate an area chart for monthly revenue by product category
    else:
        chart = alt.Chart(grouped_data).mark_area().encode(
            x=alt.X('YearMonth:T', title='Month', axis=alt.Axis(format="%b %Y", labelAngle=-90, tickCount="month")),
            y='Revenue (MYR):Q',
            color='Product Category:N',
            tooltip=['YearMonth:T', 'Revenue (MYR):Q', 'Product Category:N']
        ).interactive()  # Enable interactivity for area chart

    # Display the selected chart in Streamlit with container width
    st.altair_chart(chart, use_container_width=True)


# Calculate regression performance metrics and display as table
def regression_performance_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics_df = pd.DataFrame({
        "Performance Metric": ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "R-squared"],
        "Value": [f"{mae:.2f}", f"{mse:.2f}", f"{r2:.2f}"]
    })

    st.table(metrics_df)


# Revenue Prediction using Regression models by Month
def revenue_prediction():
    # Aggregate data by month to get total revenue per month
    monthly_revenue = monthly_data.copy()
    monthly_revenue['YearMonth'] = monthly_revenue['DateTime'].dt.to_period('M').dt.to_timestamp()
    monthly_revenue = monthly_revenue.groupby('YearMonth')['Revenue (MYR)'].sum().reset_index()

    # Calculate MonthNumber starting from January 2022
    start_date = pd.to_datetime("2022-01-01")
    monthly_revenue['MonthNumber'] = ((monthly_revenue['YearMonth'].dt.year - start_date.year) * 12 +
                                      (monthly_revenue['YearMonth'].dt.month - start_date.month) + 1)

    if monthly_revenue.empty:
        st.warning("No data available for the monthly revenue.")
        return

    # Prepare data for modeling
    X_full = monthly_revenue[['MonthNumber']]
    y_full = monthly_revenue['Revenue (MYR)']

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_full, y_full)
    monthly_revenue['Linear Regression'] = model.predict(X_full)

    # Decision Tree Regressor with different depths
    X_high_res = pd.DataFrame({
        'MonthNumber': np.arange(X_full['MonthNumber'].min(), X_full['MonthNumber'].max() + 1, 0.1)
    })
    X_high_res['YearMonth'] = start_date + pd.to_timedelta((X_high_res['MonthNumber'] - 1) * 30, unit='D')

    model_depth_2 = DecisionTreeRegressor(max_depth=2, random_state=0)
    model_depth_2.fit(X_full, y_full)
    monthly_revenue['Decision Tree Depth 2'] = model_depth_2.predict(X_full)
    X_high_res['Decision Tree Depth 2'] = model_depth_2.predict(X_high_res[['MonthNumber']])

    model_depth_5 = DecisionTreeRegressor(max_depth=5, random_state=0)
    model_depth_5.fit(X_full, y_full)
    monthly_revenue['Decision Tree Depth 5'] = model_depth_5.predict(X_full)
    X_high_res['Decision Tree Depth 5'] = model_depth_5.predict(X_high_res[['MonthNumber']])

    # Convert prediction data to long format for Altair
    melted_preds = pd.melt(
        X_high_res, id_vars='YearMonth', value_vars=['Decision Tree Depth 2', 'Decision Tree Depth 5'],
        var_name='Model', value_name='Predicted Revenue'
    )

    # Prepare Altair charts
    base = alt.Chart(monthly_revenue).encode(
        x=alt.X('YearMonth:T', title='Month', axis=alt.Axis(format="%b %Y", labelAngle=-45, tickCount="month"))
    )

    # Actual Revenue as dots and Linear Regression as a line
    actual_dots = base.mark_point(color='#90D5FF', filled=True).encode(
        y=alt.Y('Revenue (MYR):Q', title='Revenue (MYR)'),
        tooltip=['YearMonth:T', 'Revenue (MYR):Q']
    )

    linear_reg_line = base.mark_line(color='red').encode(
        y='Linear Regression:Q',
        tooltip=['YearMonth:T', 'Linear Regression:Q']
    )

    # Decision Tree Depth 2 line
    decision_tree_depth_2_line = alt.Chart(melted_preds).transform_filter(
        alt.datum.Model == 'Decision Tree Depth 2'
    ).mark_line(color='orange').encode(
        x='YearMonth:T',
        y='Predicted Revenue:Q',
        tooltip=['YearMonth:T', 'Predicted Revenue:Q']
    )

    # Decision Tree Depth 5 line
    decision_tree_depth_5_line = alt.Chart(melted_preds).transform_filter(
        alt.datum.Model == 'Decision Tree Depth 5'
    ).mark_line(color='green').encode(
        x='YearMonth:T',
        y='Predicted Revenue:Q',
        tooltip=['YearMonth:T', 'Predicted Revenue:Q']
    )

    # Combine actual dots with linear regression and decision tree lines into one chart
    chart = actual_dots + linear_reg_line + decision_tree_depth_2_line + decision_tree_depth_5_line

    # Display the chart
    st.altair_chart(chart.interactive(), use_container_width=True)

    # Display custom legend labels using HTML
    st.markdown("""
        <div style='text-align: left; margin-top: 5px; margin-bottom: 5px;'>
            <strong>Model Legend:</strong>
            <ul style='list-style-type: none; padding: 0;'>
                <li><span style='color: #90D5FF;'> ● </span> Actual Revenue</li>
                <li><span style='color: red;'> — </span> Linear Regression</li>
                <li><span style='color: orange;'> — </span> Decision Tree Depth 2</li>
                <li><span style='color: green;'> — </span> Decision Tree Depth 5</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Display metrics below the charts
    st.subheader("Performance Metrics")
    st.write("**Linear Regression**")
    regression_performance_metrics(y_full, monthly_revenue['Linear Regression'], "Linear Regression")
    st.write("**Decision Tree Depth 2**")
    regression_performance_metrics(y_full, monthly_revenue['Decision Tree Depth 2'], "Decision Tree Depth 2")
    st.write("**Decision Tree Depth 5**")
    regression_performance_metrics(y_full, monthly_revenue['Decision Tree Depth 5'], "Decision Tree Depth 5")


# Display stock levels over time for a selected product using different chart types
def stock_level_chart():
    # Reshape stock data to long format, with columns for product, date, and stock level
    stock_data_long = stock_data.melt(id_vars='Product', var_name='Date', value_name='Stock Level')

    # Convert 'Date' to datetime format, handling errors to remove invalid dates
    stock_data_long['Date'] = pd.to_datetime(stock_data_long['Date'], format='%d/%m/%Y', errors='coerce')

    # Remove rows with invalid dates (NaT) from the dataset
    stock_data_long = stock_data_long.dropna(subset=['Date'])

    # Allow user to select a specific product to view its stock level over time
    selected_product = st.selectbox("Select a product to view stock levels", stock_data['Product'].unique())

    # Allow user to select a chart type for displaying stock levels
    chart_type = st.selectbox("Select chart type", ["Line Chart", "Bar Chart", "Step Chart"], key="stock_chart_type")

    # Filter data for the selected product
    product_data = stock_data_long[stock_data_long['Product'] == selected_product]

    # Check if there is data available for the selected product; if not, show a warning
    if product_data.empty:
        st.warning("No data available for the selected product.")
        return

    # Generate a line chart with points for stock level over time
    if chart_type == "Line Chart":
        line = alt.Chart(product_data).mark_line().encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format="%d %b %Y", labelAngle=-90)),
            y='Stock Level:Q',
            tooltip=['Date:T', 'Stock Level:Q', 'Product:N']  # Tooltip for interactive details
        )
        points = line.mark_point()  # Add points to the line chart for clarity on data points
        chart = line + points  # Combine line and points for the final chart

    # Generate a bar chart for stock level over time
    elif chart_type == "Bar Chart":
        chart = alt.Chart(product_data).mark_bar().encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format="%d %b %Y", labelAngle=-90)),
            y='Stock Level:Q',
            tooltip=['Date:T', 'Stock Level:Q', 'Product:N']
        ).interactive()  # Enable interactivity for the bar chart

    # Generate a step chart to show stock level changes more clearly
    else:
        chart = alt.Chart(product_data).mark_line(interpolate='step-after').encode(
            x=alt.X('Date:T', title='Date', axis=alt.Axis(format="%d %b %Y", labelAngle=-90)),
            y='Stock Level:Q',
            tooltip=['Date:T', 'Stock Level:Q', 'Product:N']
        ).interactive()  # Enable interactivity for the step chart

    # Display the selected chart in Streamlit with container width
    st.altair_chart(chart, use_container_width=True)


# Calculate classification performance metrics
def classification_performance_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    st.write(f"**{model_name} Performance Metrics**")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write("")


# Function to calculate sales volume and restock frequency for each product
def calculate_sales_and_restock(stock_data):
    # Lists to store calculated sales volumes and restock frequencies
    sales_volumes = []
    restock_frequencies = []

    # Select only numeric columns from stock_data to focus on daily stock changes
    numeric_data = stock_data.select_dtypes(include=[np.number])

    # Iterate over each row in the numeric data to calculate daily changes
    for _, row in numeric_data.iterrows():
        # Calculate the difference between each day to determine daily stock changes
        daily_changes = row.diff().fillna(0)  # Fill NaNs with 0 to handle first-day difference

        # Calculate sales volume: Sum of all negative changes in stock levels (indicating sales)
        sales_volume = -daily_changes[daily_changes < 0].sum()
        sales_volumes.append(sales_volume)

        # Calculate restock frequency: Count of all positive changes in stock levels (indicating restocks)
        restock_frequency = (daily_changes > 0).sum()
        restock_frequencies.append(restock_frequency)

    # Add the calculated sales volumes and restock frequencies as new columns in stock_data
    stock_data['Sales_Volume'] = sales_volumes
    stock_data['Restock_Frequency'] = restock_frequencies


# Logistic Regression Classification with Graph and Table
def logistic_regression_classification():
    st.subheader("Product Demand Classification using Logistic Regression")

    # Calculate sales volume and restock frequency
    calculate_sales_and_restock(stock_data)

    # Log-transform the sales volume to handle skewness
    stock_data['Log_Sales_Volume'] = np.log1p(stock_data['Sales_Volume'])

    # Prepare feature matrix X (Log-transformed Sales Volume) and target vector y (High or Low Demand)
    X = stock_data[['Log_Sales_Volume']].values
    y = (stock_data['Sales_Volume'] > np.median(stock_data['Sales_Volume'])).astype(int)

    # Standardize the feature (Log-transformed Sales Volume) for logistic regression
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_standardized, y)

    # Generate a range of values for plotting the logistic regression curve
    X_test_standardized = np.linspace(X_standardized.min() - 1, X_standardized.max() + 1, 300).reshape(-1, 1)
    y_prob = model.predict_proba(X_test_standardized)[:, 1]  # Probability for the positive class (High Demand)

    # Prepare data for plotting
    chart_data = pd.DataFrame({
        'Sales_Volume_Standardized': X_standardized.flatten(),
        'Probability_High_Demand': y,  # True labels for the scatter plot
        'Demand': np.where(y == 1, 'High Demand', 'Low Demand'),
        'Product': stock_data['Product']
    })

    # Prepare data for plotting the logistic regression curve
    curve_data = pd.DataFrame({
        'Sales_Volume_Standardized': X_test_standardized.flatten(),
        'Probability_High_Demand': y_prob
    })

    # Scatter plot of the actual demand data with color coding for High and Low Demand
    scatter_chart = alt.Chart(chart_data).mark_point(filled=True).encode(
        x=alt.X('Sales_Volume_Standardized:Q', title='Sales Volume (Standardized)',
                scale=alt.Scale(domain=[X_standardized.min() - 1, X_standardized.max() + 1])),
        y=alt.Y('Probability_High_Demand:Q', title='Probability of High Demand', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Demand:N', scale=alt.Scale(domain=['Low Demand', 'High Demand'], range=['red', 'green'])),
        tooltip=['Product', 'Sales_Volume_Standardized']  # Tooltip showing the product and its sales volume
    ).properties(
        width=600,
        height=400,
    )

    # Logistic regression curve line showing probability of High Demand
    curve_line = alt.Chart(curve_data).mark_line(color='blue').encode(
        x='Sales_Volume_Standardized',
        y='Probability_High_Demand'
    )

    # Threshold line at 0.5 (the decision boundary) for classification
    threshold_line = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(color='yellow', strokeDash=[5, 5]).encode(
        y='y:Q'
    )

    # Arrow pointing to the threshold line
    arrow_line = alt.Chart(pd.DataFrame({'x': [1.5], 'y': [0.55], 'y2': [0.6]})).mark_line(color='yellow').encode(
        x='x:Q',
        y='y:Q',
        y2='y2:Q'
    )

    # Arrow head to visually point to the threshold line
    arrow_head = alt.Chart(pd.DataFrame({'x': [1.5], 'y': [0.55]})).mark_point(
        shape='triangle', size=100, angle=180, color='yellow'
    ).encode(
        x='x:Q',
        y='y:Q'
    )

    # Label for the threshold line
    threshold_label = alt.Chart(pd.DataFrame({'x': [1.6], 'y': [0.55], 'text': ['Threshold = 0.5']})).mark_text(
        align='left', baseline='middle', dx=5, color='yellow'
    ).encode(
        x='x:Q',
        y='y:Q',
        text='text:N'
    )

    # Combine the scatter plot, regression line, threshold line, arrow, and threshold label
    combined_chart = scatter_chart + curve_line + threshold_line + arrow_line + arrow_head + threshold_label
    st.altair_chart(combined_chart, use_container_width=True)

    # Make predictions using the trained logistic regression model
    y_pred = model.predict(X_standardized)
    stock_data['Demand_Classification_Logistic'] = y_pred  # Store the classification results in the dataframe

    # Display the products classified as high or low demand in table format
    high_demand_logistic = stock_data[stock_data['Demand_Classification_Logistic'] == 1]['Product'].tolist()
    low_demand_logistic = stock_data[stock_data['Demand_Classification_Logistic'] == 0]['Product'].tolist()

    # Adjust list lengths to ensure both lists have the same length for table display
    max_len = max(len(high_demand_logistic), len(low_demand_logistic))
    high_demand_logistic += [""] * (max_len - len(high_demand_logistic))
    low_demand_logistic += [""] * (max_len - len(low_demand_logistic))

    # Create a table displaying high demand and low demand products side by side
    logistic_table = pd.DataFrame({
        "High Demand": high_demand_logistic,
        "Low Demand": low_demand_logistic
    })
    st.table(logistic_table)

    # Display classification performance metrics
    classification_performance_metrics(y, y_pred, "Logistic Regression")


# SVM Classification with Graph and Table
def classification_svm():
    st.subheader("Product Demand Classification using SVM")

    # Log-transform the sales volume and restock frequency to reduce skewness
    stock_data['Log_Sales_Volume'] = np.log1p(stock_data['Sales_Volume'])
    stock_data['Log_Restock_Frequency'] = np.log1p(stock_data['Restock_Frequency'])

    # Prepare the feature matrix X with the log-transformed sales volume and restock frequency
    X = stock_data[['Log_Sales_Volume', 'Log_Restock_Frequency']].values
    scaler = StandardScaler()  # Standardize the features to mean=0 and variance=1
    X_standardized = scaler.fit_transform(X)

    # Define the target vector y based on the median sales volume: high demand if above median, low otherwise
    threshold = np.median(stock_data['Sales_Volume'])
    y = np.array([1 if volume > threshold else 0 for volume in stock_data['Sales_Volume']])

    # Train a Support Vector Machine (SVM) with a linear kernel
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_standardized, y)

    # Prepare data for plotting the SVM decision boundary
    chart_data = pd.DataFrame(X_standardized,
                              columns=['Standardized_Log_Sales_Volume', 'Standardized_Log_Restock_Frequency'])
    chart_data['Demand'] = np.where(y == 1, 'High Demand', 'Low Demand')
    chart_data['Product'] = stock_data['Product']

    # Get the coefficients of the SVM model to calculate the decision boundary line
    w = svm_model.coef_[0]
    b = svm_model.intercept_[0]
    a = -w[0] / w[1]  # Slope of the decision boundary

    # Define the x-axis range for plotting the decision boundary
    x_min = chart_data['Standardized_Log_Sales_Volume'].min() - 1
    x_max = chart_data['Standardized_Log_Sales_Volume'].max() + 1
    xx = np.linspace(x_min, x_max, num=100)
    yy = a * xx - (b / w[1])  # Decision boundary equation
    boundary_data = pd.DataFrame({'Standardized_Log_Sales_Volume': xx, 'Standardized_Log_Restock_Frequency': yy})

    # Scatter plot of data points (sales volume and restock frequency), color-coded by demand
    scatter_chart = alt.Chart(chart_data).mark_point(filled=True).encode(
        x=alt.X('Standardized_Log_Sales_Volume:Q', title='Log-Transformed Sales Volume (Standardized)'),
        y=alt.Y('Standardized_Log_Restock_Frequency:Q', title='Log-Transformed Restock Frequency (Standardized)'),
        color=alt.Color('Demand:N', scale=alt.Scale(domain=['Low Demand', 'High Demand'], range=['blue', 'orange'])),
        tooltip=['Product', 'Demand']
    ).properties(
        width=600,
        height=400,
    )

    # Decision boundary line (SVM separator)
    boundary_line = alt.Chart(boundary_data).mark_line(color='red', strokeDash=[5, 5]).encode(
        x='Standardized_Log_Sales_Volume',
        y='Standardized_Log_Restock_Frequency'
    )

    # Arrow pointing to the decision boundary line
    arrow_line = alt.Chart(pd.DataFrame({'x': [1.5], 'y': [-3.6], 'y2': [-3.4]})).mark_line(color='red').encode(
        x='x:Q',
        y='y:Q',
        y2='y2:Q'
    )

    # Arrow head to visually point to the decision boundary
    arrow_head = alt.Chart(pd.DataFrame({'x': [1.5], 'y': [-3.6]})).mark_point(
        shape='triangle', size=100, angle=180, color='red'
    ).encode(
        x='x:Q',
        y='y:Q'
    )

    # Label for the decision boundary line
    boundary_label = alt.Chart(pd.DataFrame({'x': [1.6], 'y': [-3.7], 'text': ['Boundary Line']})).mark_text(
        align='left', baseline='middle', dx=5, color='red'
    ).encode(
        x='x:Q',
        y='y:Q',
        text='text:N'
    )

    # Combine the scatter plot, decision boundary line, arrows, and label
    combined_chart = scatter_chart + boundary_line + arrow_line + arrow_head + boundary_label
    st.altair_chart(combined_chart, use_container_width=True)

    # Make predictions using the trained SVM model
    y_pred = svm_model.predict(X_standardized)
    stock_data['Demand_Classification_SVM'] = y_pred  # Store the predicted classifications in the dataframe

    # Display the classified products as high demand or low demand
    high_demand_svm = stock_data[stock_data['Demand_Classification_SVM'] == 1]['Product'].tolist()
    low_demand_svm = stock_data[stock_data['Demand_Classification_SVM'] == 0]['Product'].tolist()

    # Adjust the length of both lists for consistent display in the table
    max_len = max(len(high_demand_svm), len(low_demand_svm))
    high_demand_svm += [""] * (max_len - len(high_demand_svm))
    low_demand_svm += [""] * (max_len - len(low_demand_svm))

    # Create a table displaying high demand and low demand products side by side
    svm_table = pd.DataFrame({
        "High Demand": high_demand_svm,
        "Low Demand": low_demand_svm
    })
    st.table(svm_table)

    # Display classification performance metrics for the SVM model
    classification_performance_metrics(y, y_pred, "SVM")


# Calculate clustering performance metrics
def clustering_performance_metrics(X, labels, model_name):
    silhouette_avg = silhouette_score(X, labels)
    st.write(f"**{model_name} Performance Metric**")
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    st.write("")


# K-means Clustering for Product Demand Levels
def kmeans_clustering_chart():
    st.subheader("K-means Clustering for Product Demand Levels")

    # Calculate sales volume and restock frequency for the dataset
    calculate_sales_and_restock(stock_data)

    # Extract relevant features (sales volume and restock frequency) for clustering
    X = stock_data[['Sales_Volume', 'Restock_Frequency']].values

    # Initialize the K-means clustering algorithm with 2 clusters (High and Low Demand)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)  # Fit the model to the data
    y_kmeans = kmeans.predict(X)  # Predict the cluster labels

    # Identify the cluster representing High Demand by finding the cluster with the highest average sales volume
    high_demand_cluster = stock_data.groupby(y_kmeans)['Sales_Volume'].mean().idxmax()

    # Assign 'High Demand' or 'Low Demand' labels based on the cluster assignment
    stock_data['Demand'] = y_kmeans
    stock_data['Demand'] = stock_data['Demand'].apply(
        lambda x: 'High Demand' if x == high_demand_cluster else 'Low Demand')

    # Prepare the cluster centers for visualization
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Sales_Volume', 'Restock_Frequency'])
    cluster_centers['Demand'] = ['Centroid' for _ in range(len(cluster_centers))]  # Label centroid as 'Centroid'

    # Create a scatter plot showing the products classified as high or low demand
    scatter_chart = alt.Chart(stock_data).mark_circle(size=100).encode(
        x=alt.X('Sales_Volume:Q', title='Sales Volume'),
        y=alt.Y('Restock_Frequency:Q', title='Restock Frequency'),
        color=alt.Color('Demand:N',
                        scale=alt.Scale(domain=['Low Demand', 'High Demand'], range=['#FF69B4', '#00FF00'])),
        tooltip=['Product', 'Sales_Volume', 'Restock_Frequency']
    )

    # Plot the centroids of the clusters as large blue diamonds
    center_chart = alt.Chart(cluster_centers).mark_point(filled=True, shape='diamond', size=200,
                                                         color='#90D5FF').encode(
        x='Sales_Volume:Q',
        y='Restock_Frequency:Q'
    )

    # Combine the scatter chart and centroids to create the final chart
    combined_chart = scatter_chart + center_chart

    # Display the interactive chart
    st.altair_chart(combined_chart.interactive(), use_container_width=True)

    # Extract the high demand and low demand products for display in a table
    high_demand_products = stock_data[stock_data['Demand'] == 'High Demand']['Product'].tolist()
    low_demand_products = stock_data[stock_data['Demand'] == 'Low Demand']['Product'].tolist()

    # Ensure both lists have the same length for table display
    max_len = max(len(high_demand_products), len(low_demand_products))
    high_demand_products += [""] * (max_len - len(high_demand_products))
    low_demand_products += [""] * (max_len - len(low_demand_products))

    # Create and display the table showing high and low demand products
    demand_table = pd.DataFrame({
        "High Demand Products": high_demand_products,
        "Low Demand Products": low_demand_products
    })

    st.table(demand_table)

    # Evaluate clustering performance using Silhouette Score and display the result
    clustering_performance_metrics(X, y_kmeans, "K-Means")


# Hierarchical Clustering for Product Demand Levels
def hierarchical_clustering_analysis():
    st.subheader("Hierarchical Clustering for Product Demand Levels")

    # Prepare data for clustering: Remove missing values and select relevant numeric columns
    stock_data.dropna(subset=['Sales_Volume', 'Restock_Frequency'], inplace=True)
    X = stock_data[['Sales_Volume', 'Restock_Frequency']].values  # Features for clustering

    # Perform hierarchical clustering using Ward's method
    Z = linkage(X, method='ward')  # 'ward' linkage minimizes the variance within clusters
    stock_data['Hierarchical_Cluster'] = fcluster(Z, t=3, criterion='maxclust')  # Create 3 clusters

    # Define demand level based on median Sales_Volume
    median_sales_volume = stock_data['Sales_Volume'].median()  # Use the median of sales volume as the threshold
    stock_data['Demand'] = stock_data['Sales_Volume'].apply(
        lambda x: 'High Demand' if x > median_sales_volume else 'Low Demand')  # Label products as High or Low Demand

    # Create a scatter plot to visualize the clusters, color-coded by cluster label
    scatter_chart = alt.Chart(stock_data).mark_circle(size=100).encode(
        x=alt.X('Sales_Volume:Q', title='Sales Volume'),
        y=alt.Y('Restock_Frequency:Q', title='Restock Frequency'),
        color=alt.Color('Hierarchical_Cluster:N', scale=alt.Scale(scheme='category10'),
                        legend=alt.Legend(title="Cluster")),  # Use category colors for different clusters
        tooltip=['Product', 'Sales_Volume', 'Restock_Frequency', 'Hierarchical_Cluster', 'Demand']
    ).interactive()

    # Display the scatter chart
    st.altair_chart(scatter_chart, use_container_width=True)

    # Prepare data for displaying products in each cluster
    cluster_data = {}
    max_length = 0

    # Collect products belonging to each cluster
    for cluster in sorted(stock_data['Hierarchical_Cluster'].unique()):
        cluster_products = stock_data[stock_data['Hierarchical_Cluster'] == cluster]['Product'].tolist()
        cluster_data[f'Cluster {cluster} Products'] = cluster_products
        max_length = max(max_length, len(cluster_products))  # Find the maximum length of any cluster's product list

    # Ensure each list of products has the same length for display
    for key in cluster_data:
        cluster_data[key] += [""] * (max_length - len(cluster_data[key]))

    # Create and display a table showing the products in each cluster
    cluster_table = pd.DataFrame(cluster_data)
    st.table(cluster_table)

    # Evaluate clustering performance using Silhouette Score and display the result
    clustering_performance_metrics(X, stock_data['Hierarchical_Cluster'], "Hierarchical Clustering")


# Real-time Revenue visualization
def real_time_revenue():
    # Create a placeholder for the chart to be dynamically updated
    chart_placeholder = st.empty()

    # Initialize an empty DataFrame to store accumulated data (Revenue over time)
    accumulated_data = pd.DataFrame(columns=['Seconds', 'Revenue (MYR)'])

    # Iterate through each row in the real-time data to simulate real-time updates
    for index, row in real_time_data.iterrows():
        # Append the current row's data to the accumulated data DataFrame
        accumulated_data = pd.concat([accumulated_data, pd.DataFrame([row[['Seconds', 'Revenue (MYR)']]])],
                                     ignore_index=True)

        # Aggregate the data by 'Seconds' to get the sum of Revenue (MYR) at each second
        aggregated_data = accumulated_data.groupby('Seconds')['Revenue (MYR)'].sum().reset_index()

        # Create the chart to display the revenue over time
        chart = alt.Chart(aggregated_data).mark_line(point=True).encode(
            x=alt.X('Seconds:Q', title='Time (seconds)'),  # X-axis: Seconds (Time)
            y=alt.Y('Revenue (MYR):Q', title='Revenue (MYR)'),  # Y-axis: Revenue
            tooltip=['Seconds', 'Revenue (MYR)']  # Display tooltip with time and revenue values
        ).interactive()  # Make the chart interactive (zoom, pan)

        # Display the dynamically updating chart
        chart_placeholder.altair_chart(chart, use_container_width=True)

        # Calculate the time difference between the current row and the next one
        if index < len(real_time_data) - 1:  # If not the last row
            current_time = datetime.strptime(row['Time'].strftime('%H:%M:%S'), '%H:%M:%S')  # Current time
            next_time = datetime.strptime(real_time_data.iloc[index + 1]['Time'].strftime('%H:%M:%S'),
                                          '%H:%M:%S')  # Next time
            time_diff = (next_time - current_time).total_seconds()  # Calculate the difference in seconds
            time.sleep(time_diff)  # Sleep for the calculated time to simulate real-time update

        # Stop updating if it's the last row in the dataset
        else:
            break


# Run the Streamlit dashboard with tabs
st.title("Revenue and Stock Analysis Dashboard")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Monthly Revenue by Category",
    "Revenue Prediction using Regression",
    "Stock Level by Product",
    "Classification for Product Demand",
    "Clustering for Demand",
    "Real-Time Revenue"
])

with tab1:
    st.header("Monthly Revenue by Category")
    monthly_revenue_by_category()

with tab2:
    st.header("Linear and Decision Tree Regression with Actual Revenue")
    revenue_prediction()

with tab3:
    st.header("Stock Level by Product")
    stock_level_chart()

with tab4:
    logistic_regression_classification()
    classification_svm()

with tab5:
    kmeans_clustering_chart()
    hierarchical_clustering_analysis()

with tab6:
    st.header("Real-Time Revenue")
    real_time_revenue()
