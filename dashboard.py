import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import time

# Set matplotlib backend for Tkinter
import matplotlib
matplotlib.use("TkAgg")

# Function to load and prepare data for Revenue Over Time by Product Category graph
def load_and_prepare_first_graph_data():
    data = pd.read_csv('shopee_sales_report_2022_to_2024.csv').drop(columns=['Order Status'], errors='ignore')
    if 'Revenue (MYR)' in data.columns:
        data['Revenue (MYR)'] = data['Revenue (MYR)'].fillna(data['Revenue (MYR)'].mean())
        Q1, Q3 = data['Revenue (MYR)'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        data = data[(data['Revenue (MYR)'] >= Q1 - 1.5 * IQR) & (data['Revenue (MYR)'] <= Q3 + 1.5 * IQR)]
    if 'DateTime' in data.columns:
        data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
        data['MonthYear'] = data['DateTime'].dt.to_period('M')
    
    return data.groupby(['MonthYear', 'Product Category'])['Revenue (MYR)'].sum().unstack().fillna(0)

# Function to load and prepare data for Real-Time Total Revenue graph
def load_and_prepare_second_graph_data():
    real_time_data = pd.read_csv('sales_report_real_time.csv')
    real_time_data['Time'] = pd.to_datetime(real_time_data['Time'], format='%H:%M:%S', errors='coerce').dropna()
    real_time_data['TimeInSeconds'] = (real_time_data['Time'].dt.hour * 3600 +
                                       real_time_data['Time'].dt.minute * 60 +
                                       real_time_data['Time'].dt.second)
    return real_time_data[['TimeInSeconds', 'Revenue (MYR)']].groupby('TimeInSeconds').sum().reset_index()

# Function to load and prepare data for Stock Level graph
def load_and_prepare_third_graph_data():
    product_data = pd.read_csv('22_product_quantities_random_restock_limited_no_stock.csv').set_index('Product')
    data_transposed = product_data.T
    data_transposed.index = pd.to_datetime(data_transposed.index, dayfirst=True)
    return data_transposed

# Initialize data for graphs
grouped_data_first_graph = load_and_prepare_first_graph_data()
real_time_data = load_and_prepare_second_graph_data()
data_transposed = load_and_prepare_third_graph_data()

# Initialize start time for real-time revenue graph
start_time = time.time()

# Setup Tkinter main window
root = tk.Tk()
root.title("Dashboard")
root.geometry("1500x900")

# Create a Notebook widget for multiple tabs
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Define and add tabs to the notebook
home_tab = tk.Frame(notebook)
regression_tab = tk.Frame(notebook)
classification_clustering_tab = tk.Frame(notebook)

notebook.add(home_tab, text="Home")
notebook.add(regression_tab, text="Regression")
notebook.add(classification_clustering_tab, text="Classification & Clustering")

# Setup Home tab with grid layout for graphs
fig_home = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.5)

# Define subplots for each graph in the Home tab
ax1 = fig_home.add_subplot(gs[0, 0])   # Top-left for revenue by category graph
ax3 = fig_home.add_subplot(gs[0, 1])   # Top-right for stock level graph
ax2 = fig_home.add_subplot(gs[1, :])   # Bottom for real-time revenue graph

# Embed Home tab graph canvas in Tkinter
canvas_home = FigureCanvasTkAgg(fig_home, master=home_tab)
canvas_home.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Dropdown for selecting product in Home tab (for stock level graph)
dropdown_frame = tk.Frame(home_tab, bg="#2C3E50")
dropdown_frame.place(relx=0.71, rely=0.07, anchor="center")
dropdown_label = tk.Label(dropdown_frame, text="Select Product:", fg="white", bg="#2C3E50", font=("Arial", 12))
dropdown_label.pack(side=tk.LEFT, padx=5)
selected_product = tk.StringVar(value=data_transposed.columns[0])
dropdown = ttk.Combobox(dropdown_frame, textvariable=selected_product, values=list(data_transposed.columns), width=25, font=("Arial", 10))
dropdown.pack(side=tk.LEFT)

# Animation functions for each graph in Home tab
def animate_first_graph(i):
    ax1.cla()
    colormap = plt.get_cmap('tab10')
    colors = [colormap(j) for j in range(len(grouped_data_first_graph.columns))]
    for j, category in enumerate(grouped_data_first_graph.columns):
        ax1.plot(grouped_data_first_graph.index[:i].strftime('%b %Y'), grouped_data_first_graph[category][:i], label=category, color=colors[j], marker='o', linestyle='-', alpha=0.8)
    ax1.set_title('Revenue Over Time by Product Category')
    ax1.set_ylabel('Revenue (MYR)')
    ax1.set_xlabel('Year-Month')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True)
    for label in ax1.get_xticklabels():
        label.set_fontsize(6)
        label.set_rotation(55)

def animate_second_graph(i):
    ax2.cla()
    elapsed_time = int(time.time() - start_time)
    data_to_plot = real_time_data[real_time_data['TimeInSeconds'] <= elapsed_time]
    ax2.plot(data_to_plot['TimeInSeconds'], data_to_plot['Revenue (MYR)'], color='violet', marker='o', linestyle='-', alpha=0.8)
    ax2.set_title('Real-Time Total Revenue')
    ax2.set_xlabel('Time (Seconds)')
    ax2.set_ylabel('Revenue (MYR)')
    ax2.grid(True)

# Stock level animation for selected product
third_graph_anim = None

def animate_third_graph(i):
    ax3.cla()
    stock_data = data_transposed[selected_product.get()]
    ax3.plot(stock_data.index[:i], stock_data.values[:i], marker='o', linestyle='-', alpha=0.8)
    ax3.set_title(f"Stock Level for {selected_product.get()}")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Stock Amount")
    ax3.grid(True)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y'))
    for label in ax3.get_xticklabels():
        label.set_rotation(55)
        label.set_fontsize(6)
    if i == len(stock_data) - 1:
        third_graph_anim.event_source.stop()

def start_third_graph_animation():
    global third_graph_anim
    if third_graph_anim:
        third_graph_anim.event_source.stop()
    third_graph_anim = FuncAnimation(fig_home, animate_third_graph, frames=len(data_transposed.index), interval=100, repeat=False)
    canvas_home.draw()

# Callback to restart the third graph animation on dropdown change
def on_dropdown_change(*args):
    start_third_graph_animation()

# Bind the callback to the dropdown selection change
selected_product.trace("w", on_dropdown_change)

# Start animations for the graphs
anim1 = FuncAnimation(fig_home, animate_first_graph, frames=100, interval=100, repeat=False)
anim2 = FuncAnimation(fig_home, animate_second_graph, frames=1000, interval=1000, repeat=True)
start_third_graph_animation()

# Regression Tab: Set up linear and decision tree regression plots
fig_reg, (ax_reg1, ax_reg2) = plt.subplots(1, 2, figsize=(14, 6))

# Dropdown to select category for regression analysis
categories = grouped_data_first_graph.columns.tolist()
category_frame = tk.Frame(regression_tab, bg="#2C3E50")
category_frame.pack(anchor="center", pady=5)
category_label = tk.Label(category_frame, text="Select Category:", fg="white", bg="#2C3E50", font=("Arial", 12))
category_label.pack(side=tk.LEFT, padx=5)
selected_category = tk.StringVar(value=categories[0])
category_dropdown = ttk.Combobox(category_frame, textvariable=selected_category, values=categories, width=25, font=("Arial", 10))
category_dropdown.pack(side=tk.LEFT)

# Function to prepare data for regression based on selected category
def load_and_prepare_regression_data(category):
    data = pd.read_csv('shopee_sales_report_2022_to_2024.csv')
    data_filtered = data[data['Product Category'] == category].copy()
    data_filtered['DateTime'] = pd.to_datetime(data_filtered['DateTime'], dayfirst=True, errors='coerce')
    data_filtered['MonthYear'] = data_filtered['DateTime'].dt.to_period('M')
    grouped_data = data_filtered.groupby('MonthYear')['Revenue (MYR)'].sum().reset_index()
    grouped_data['MonthYear'] = grouped_data['MonthYear'].dt.to_timestamp()
    grouped_data['MonthNumber'] = np.arange(len(grouped_data))
    return grouped_data

# Function to update regression plots based on selected category
def update_regression_plots(*args):
    ax_reg1.cla()
    ax_reg2.cla()
    
    regression_data = load_and_prepare_regression_data(selected_category.get())
    if not regression_data.empty:
        X_full = regression_data[['MonthNumber']]
        y_full = regression_data['Revenue (MYR)']

        # Linear Regression Plot
        model = LinearRegression()
        model.fit(X_full, y_full)
        predictions_full = model.predict(X_full)
        ax_reg1.plot(regression_data['MonthYear'], y_full, 'o', label='Actual Revenue', color='blue')
        ax_reg1.plot(regression_data['MonthYear'], predictions_full, label='Predicted Revenue', color='red')
        ax_reg1.set_xlabel('Month')
        ax_reg1.set_ylabel('Revenue (MYR)')
        ax_reg1.set_title(f'Linear Regression of Monthly Revenue for {selected_category.get()}')
        ax_reg1.legend()
        ax_reg1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax_reg1.tick_params(axis='x', rotation=55, labelsize=6)

        # Decision Tree Regression Plot
        X_high_res = pd.DataFrame({'MonthNumber': np.arange(X_full['MonthNumber'].min(), X_full['MonthNumber'].max() + 1, 0.1)})
        ax_reg2.scatter(X_full, y_full, color='orange', label='Actual Data')
        model_min_depth = DecisionTreeRegressor(max_depth=2, random_state=0)
        model_min_depth.fit(X_full, y_full)
        pred_min_depth = model_min_depth.predict(X_high_res)
        ax_reg2.plot(X_high_res['MonthNumber'], pred_min_depth, color='blue', label='min depth (max_depth=2)')
        model_max_depth = DecisionTreeRegressor(max_depth=5, random_state=0)
        model_max_depth.fit(X_full, y_full)
        pred_max_depth = model_max_depth.predict(X_high_res)
        ax_reg2.plot(X_high_res['MonthNumber'], pred_max_depth, color='green', label='max depth (max_depth=5)')
        ax_reg2.set_xlabel('Month')
        ax_reg2.set_ylabel('Revenue (MYR)')
        ax_reg2.set_title(f'Decision Tree Regression of Monthly Revenue for {selected_category.get()}')
        ax_reg2.legend()
        month_labels = pd.date_range(start=regression_data['MonthYear'].iloc[0], periods=len(X_full), freq='MS').strftime('%b %Y')
        ax_reg2.set_xticks(X_full['MonthNumber'])
        ax_reg2.set_xticklabels(month_labels, rotation=55, ha="right", fontsize=6)

    canvas_reg.draw()

# Bind category dropdown selection to update function
selected_category.trace("w", update_regression_plots)

# Embed regression plots in Tkinter
canvas_reg = FigureCanvasTkAgg(fig_reg, master=regression_tab)
canvas_reg.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Initialize regression plots
update_regression_plots()

# Classification & Clustering Tab Content with Logistic Regression, SVM, K-Means, and Dendrogram
fig_class_cluster, axs = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)

# (Graph 1) Logistic Regression with annotations
product_data = pd.read_csv('22_product_quantities_random_restock_limited_no_stock.csv')
sales_volumes = []
restock_frequencies = []
for _, row in product_data.iterrows():
    daily_changes = pd.to_numeric(row[1:].diff(), errors='coerce').fillna(0)
    sales_volume = -daily_changes[daily_changes < 0].sum()
    restock_frequency = (daily_changes > 0).sum()
    sales_volumes.append(sales_volume)
    restock_frequencies.append(restock_frequency)
product_data['Sales_Volume'] = sales_volumes
product_data['Restock_Frequency'] = restock_frequencies
product_data['Log_Sales_Volume'] = np.log1p(product_data['Sales_Volume'])
product_data['Log_Restock_Frequency'] = np.log1p(product_data['Restock_Frequency'])
X = product_data[['Log_Sales_Volume']].values
y = (product_data['Sales_Volume'] > np.median(product_data['Sales_Volume'])).astype(int)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_standardized, y)
X_test_standardized = np.linspace(X_standardized.min() - 1, X_standardized.max() + 1, 300).reshape(-1, 1)
y_prob = model.predict_proba(X_test_standardized)[:, 1]
axs[0, 0].scatter(X_standardized[y == 0], y[y == 0], color='red', s=50, label="Low Demand", marker='o')
axs[0, 0].scatter(X_standardized[y == 1], y[y == 1], color='green', s=50, label="High Demand", marker='o')
axs[0, 0].plot(X_test_standardized, y_prob, color='blue', linewidth=2, label="Logistic Regression Curve")
axs[0, 0].axhline(y=0.5, color='gray', linestyle='--')
axs[0, 0].annotate('Threshold', xy=(0, 0.5), xytext=(-1.5, 0.4),
                   arrowprops=dict(facecolor='black', arrowstyle='->'))
axs[0, 0].text(X_standardized.min() - 0.5, 0.05, "Low Demand", color="red", verticalalignment='center')
axs[0, 0].text(X_standardized.max() + 0.5, 0.95, "High Demand", color="green", verticalalignment='center')
axs[0, 0].set_title("Logistic Regression Demand Classification")
axs[0, 0].set_xlabel("Sales Volume")
axs[0, 0].set_ylabel("Probability of High Demand")
axs[0, 0].legend()
axs[0, 0].grid(True)

# (Graph 2) SVM Classification
X = product_data[['Log_Sales_Volume', 'Log_Restock_Frequency']].values
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
threshold = np.median(sales_volumes)
y = np.array([1 if volume > threshold else 0 for volume in sales_volumes])
svm_model = SVC(kernel='linear')
svm_model.fit(X_standardized, y)
w = svm_model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X_standardized[:, 0].min() - 1, X_standardized[:, 0].max() + 1)
yy = a * xx - (svm_model.intercept_[0]) / w[1]
axs[0, 1].scatter(X_standardized[y == 0][:, 0], X_standardized[y == 0][:, 1], color='blue', s=100, label="Low Demand", marker='o')
axs[0, 1].scatter(X_standardized[y == 1][:, 0], X_standardized[y == 1][:, 1], color='orange', s=100, label="High Demand", marker='^')  # Triangle for High Demand
axs[0, 1].plot(xx, yy, 'r--', label="Hyperline")
axs[0, 1].set_xlim(-2, 2)
axs[0, 1].set_ylim(-2, 2)
axs[0, 1].set_title("SVM Demand Classification")
axs[0, 1].set_xlabel("Log-Transformed Sales Volume")
axs[0, 1].set_ylabel("Log-Transformed Restock Frequency")
axs[0, 1].legend()
axs[0, 1].grid(True)

# (Graph 3) K-Means Clustering
X = product_data[['Sales_Volume', 'Restock_Frequency']].values
kmeans = KMeans(n_clusters=2, random_state=0)
y_kmeans = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
axs[1, 0].scatter(X[y_kmeans == 0][:, 0], X[y_kmeans == 0][:, 1], color='blue', label="Low Demand", marker='^')
axs[1, 0].scatter(X[y_kmeans == 1][:, 0], X[y_kmeans == 1][:, 1], color='red', label="High Demand", marker='o')
axs[1, 0].scatter(centers[:, 0], centers[:, 1], color='black', marker='X', s=200, label="Centroids")
axs[1, 0].set_title("K-Means Clustering")
axs[1, 0].set_xlabel("Sales Volume")
axs[1, 0].set_ylabel("Restock Frequency")
axs[1, 0].legend()
axs[1, 0].grid(True)

# (Graph 4) Hierarchical Clustering Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z, labels=product_data['Product'].values, leaf_rotation=55, leaf_font_size=3, ax=axs[1, 1])
axs[1, 1].set_title("Hierarchical Clustering Dendrogram")
axs[1, 1].set_xlabel("Products")
axs[1, 1].set_ylabel("Distance")

# Embed classification & clustering plots in Tkinter
canvas_class_cluster = FigureCanvasTkAgg(fig_class_cluster, master=classification_clustering_tab)
canvas_class_cluster.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Run Tkinter main loop
root.mainloop()
