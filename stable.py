# app.py - Subsidy Tracker & Fraud Detection Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
import tempfile
import base64
from io import BytesIO
import shap

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Subsidy Tracker & Fraud Detection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.1rem 0.25rem rgba(0,0,0,0.1);
    }
    .sidebar-content {
        padding: 1rem;
    }
    .risk-high {
        color: #D32F2F;
        font-weight: bold;
    }
    .risk-medium {
        color: #FFA000;
        font-weight: bold;
    }
    .risk-low {
        color: #388E3C;
        font-weight: bold;
    }
    .explanation-box {
        background-color: #F1F8E9;
        border-left: 5px solid #7CB342;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'fraud_models' not in st.session_state:
    st.session_state.fraud_models = {}
if 'anomaly_model' not in st.session_state:
    st.session_state.anomaly_model = None
if 'propensity_model' not in st.session_state:
    st.session_state.propensity_model = None
if 'cluster_model' not in st.session_state:
    st.session_state.cluster_model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()

# Define functions for data loading and preprocessing
def load_data(file_path=None):
    """Load data from CSV file or use sample data"""
    try:
        if file_path:
            df = pd.read_csv(file_path)
        else:
            # Create sample data if no file is provided
            np.random.seed(42)
            n_records = 1000
            
            # Generate sample data
            regions = ['Lagos', 'Kano', 'Rivers', 'Oyo', 'Kaduna', 'Yobe']
            income_levels = ['Low', 'Middle', 'High']
            genders = ['Male', 'Female']
            channels = ['Bank Transfer', 'Mobile Money', 'Cash Pickup']
            subsidy_types = ['Energy', 'Food', 'Housing']
            wallet_status = ['Active', 'Inactive', 'Suspicious']
            
            df = pd.DataFrame({
                'National_ID': [f'ID{i:05d}' for i in range(1, n_records + 1)],
                'Age': np.random.randint(18, 70, n_records),
                'Gender': np.random.choice(genders, n_records),
                'Region': np.random.choice(regions, n_records),
                'Income_Level': np.random.choice(income_levels, n_records, p=[0.6, 0.3, 0.1]),
                'Household_Dependents': np.random.randint(0, 10, n_records),
                'Monthly_Energy_Consumption_kWh': np.random.normal(100, 20, n_records),
                'Amount (NGN)': np.random.normal(4000, 1000, n_records),
                'Wallet_Balance (NGN)': np.random.normal(5000, 2000, n_records),
                'Days_Since_Last_Transaction': np.random.randint(0, 90, n_records),
                'Channel': np.random.choice(channels, n_records),
                'Subsidy_Type': np.random.choice(subsidy_types, n_records),
                'Wallet_Activity_Status': np.random.choice(wallet_status, n_records, p=[0.8, 0.15, 0.05]),
                'Avg_Monthly_Wallet_Balance': np.random.normal(5000, 2000, n_records),
                'Date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)],
                'Subsidy_Eligibility': np.random.choice([0, 1], n_records, p=[0.3, 0.7]),
                'Suspected_Fraud': np.random.choice([0, 1], n_records, p=[0.95, 0.05])
            })
            
            # Ensure no negative values
            df['Amount (NGN)'] = df['Amount (NGN)'].abs()
            df['Wallet_Balance (NGN)'] = df['Wallet_Balance (NGN)'].abs()
            df['Monthly_Energy_Consumption_kWh'] = df['Monthly_Energy_Consumption_kWh'].abs()
            df['Avg_Monthly_Wallet_Balance'] = df['Avg_Monthly_Wallet_Balance'].abs()
            
        # Ensure 'Date' column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def data_cleaning_integration(df):
    """Complete data cleaning and integration process"""
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        if col != 'Date' and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Standardize ID format
    if 'National_ID' in df.columns:
        df['National_ID'] = df['National_ID'].str.upper().str.strip()
    
    # Remove duplicates
    initial_count = len(df)
    if 'National_ID' in df.columns and 'Date' in df.columns:
        df.drop_duplicates(subset=['National_ID', 'Date'], inplace=True)
    
    # Feature Engineering
    if 'Amount (NGN)' in df.columns and 'Household_Dependents' in df.columns:
        # Avoid division by zero for Household_Dependents
        df['Amount_per_Dependent'] = df.apply(
            lambda row: row['Amount (NGN)'] / row['Household_Dependents'] if row['Household_Dependents'] > 0 else row['Amount (NGN)'],
            axis=1
        )
    
    if 'Monthly_Energy_Consumption_kWh' in df.columns and 'Household_Dependents' in df.columns:
        # Avoid division by zero for Household_Dependents
        df['Energy_per_Dependent'] = df.apply(
            lambda row: row['Monthly_Energy_Consumption_kWh'] / row['Household_Dependents'] if row['Household_Dependents'] > 0 else row['Monthly_Energy_Consumption_kWh'],
            axis=1
        )
    
    if 'Date' in df.columns and 'Days_Since_Distribution' not in df.columns:
        try:
            # Ensure 'Date' is datetime type before calculating difference
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Days_Since_Distribution'] = (datetime.now() - df['Date']).dt.days
        except Exception as e:
            pass
    
    return df

def calculate_subsidy_metrics(df):
    """Calculate key subsidy performance metrics"""
    metrics = {}
    
    # Ensure required columns exist
    if all(col in df.columns for col in ['Subsidy_Eligibility', 'Amount (NGN)', 'Income_Level', 'Suspected_Fraud']):
        # 1. Subsidy Coverage Rate
        eligible_population = df[df['Subsidy_Eligibility'] == 1]
        received_subsidy = eligible_population[eligible_population['Amount (NGN)'] > 0]
        coverage_rate = len(received_subsidy) / len(eligible_population) if len(eligible_population) > 0 else 0
        
        # 2. Targeting Accuracy (% reaching low income)
        low_income_recipients = df[
            (df['Income_Level'] == 'Low') & (df['Amount (NGN)'] > 0)
        ]
        all_recipients = df[df['Amount (NGN)'] > 0]
        targeting_accuracy = len(low_income_recipients) / len(all_recipients) if len(all_recipients) > 0 else 0
        
        # 3. Leakage Rate
        ineligible_recipients = df[
            (df['Subsidy_Eligibility'] == 0) & (df['Amount (NGN)'] > 0)
        ]
        leakage_rate = len(ineligible_recipients) / len(all_recipients) if len(all_recipients) > 0 else 0
        
        # 5. Fraud Rate
        fraud_rate = df['Suspected_Fraud'].mean()
        
        # Store metrics
        metrics = {
            'subsidy_coverage_rate': coverage_rate,
            'targeting_accuracy': targeting_accuracy,
            'leakage_rate': leakage_rate,
            'fraud_rate': fraud_rate
        }
    
    return metrics

# Define functions for exploratory data analysis
def correlation_analysis(df):
    """Perform correlation analysis on key variables"""
    # List of desired numeric columns focusing on key subsidy features
    desired_numeric_cols = [
        'Amount (NGN)',
        'Household_Dependents',
        'Monthly_Energy_Consumption_kWh',
        'Wallet_Balance (NGN)',
        'Days_Since_Last_Transaction',
        'Amount_per_Dependent',
        'Energy_per_Dependent',
        'Days_Since_Distribution',
        'Suspected_Fraud'  # numeric (0 or 1)
    ]
    
    # Filter for columns that exist in the current DataFrame and are numeric
    existing_numeric_cols = [col for col in desired_numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    # Check if there are enough numeric columns to calculate correlation
    if len(existing_numeric_cols) < 2:
        return None
    
    correlation_matrix = df[existing_numeric_cols].corr()
    
    return correlation_matrix

# Define functions for anomaly detection
def anomaly_detection_unsupervised(df):
    """Implement unsupervised anomaly detection using Isolation Forest"""
    # Prepare features for anomaly detection
    feature_cols = [
        'Age', 'Household_Dependents', 'Monthly_Energy_Consumption_kWh',
        'Amount (NGN)', 'Wallet_Balance (NGN)',
        'Days_Since_Last_Transaction'
    ]
    
    # Handle categorical variables
    df_encoded = df.copy()
    le_dict = {}
    for col in ['Gender', 'Region', 'Income_Level', 'Channel', 'Subsidy_Type']:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df[col])
            le_dict[col] = le
            feature_cols.append(f'{col}_encoded')
    
    # Prepare data
    X_anomaly = df_encoded[feature_cols].fillna(0)
    X_anomaly_scaled = st.session_state.scaler.fit_transform(X_anomaly)
    
    # Train Isolation Forest
    anomaly_model = IsolationForest(
        contamination=0.15,  # Expected fraud rate
        random_state=42,
        n_estimators=100
    )
    
    # Fit and predict
    anomaly_predictions = anomaly_model.fit_predict(X_anomaly_scaled)
    anomaly_scores = anomaly_model.decision_function(X_anomaly_scaled)
    
    # Convert predictions (-1 for anomaly, 1 for normal) to (1 for anomaly, 0 for normal)
    df_encoded['Anomaly_Predicted'] = (anomaly_predictions == -1).astype(int)
    df_encoded['Anomaly_Score'] = anomaly_scores
    
    return df_encoded, anomaly_model

# Define functions for supervised fraud detection
def supervised_fraud_detection(df):
    """Implement supervised machine learning for fraud detection"""
    # Prepare features
    feature_cols = [
        'Age', 'Household_Dependents', 'Monthly_Energy_Consumption_kWh',
        'Amount (NGN)', 'Wallet_Balance (NGN)',
        'Days_Since_Last_Transaction', 'Amount_per_Dependent', 'Energy_per_Dependent'
    ]
    
    # Encode categorical variables
    df_ml = df.copy()
    for col in ['Gender', 'Region', 'Income_Level', 'Channel', 'Subsidy_Type']:
        if col in df.columns:
            dummies = pd.get_dummies(df_ml[col], prefix=col)
            df_ml = pd.concat([df_ml, dummies], axis=1)
            # Only add dummy columns that don't already exist in feature_cols
            new_dummy_cols = [col for col in dummies.columns.tolist() if col not in feature_cols]
            feature_cols.extend(new_dummy_cols)
    
    # Prepare data
    # Ensure all selected feature columns exist in the dataframe
    existing_feature_cols = [col for col in feature_cols if col in df_ml.columns]
    X = df_ml[existing_feature_cols].fillna(0)
    y = df_ml['Suspected_Fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler_ml = StandardScaler()
    # Only scale numeric columns
    numeric_cols_train = X_train.select_dtypes(include=np.number).columns
    numeric_cols_test = X_test.select_dtypes(include=np.number).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if len(numeric_cols_train) > 0:
        X_train_scaled[numeric_cols_train] = scaler_ml.fit_transform(X_train[numeric_cols_train])
    
    if len(numeric_cols_test) > 0:
        X_test_scaled[numeric_cols_test] = scaler_ml.transform(X_test[numeric_cols_test])
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'X_train': X_train,
            'X_test': X_test
        }
    
    # Store best model and all results
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    
    # Store the exact feature names used during training
    for model_name in results:
        results[model_name]['feature_names'] = existing_feature_cols
    
    return results, best_model_name

# Define function for clustering analysis
def behavioral_clustering(df):
    """Implement KMeans clustering for behavioral analysis"""
    # Prepare features for clustering
    feature_cols = [
        'Amount (NGN)',
        'Wallet_Balance (NGN)',
        'Days_Since_Last_Transaction',
        'Monthly_Energy_Consumption_kWh',
        'Household_Dependents'
    ]
    
    # Filter for columns that exist in the current DataFrame
    existing_feature_cols = [col for col in feature_cols if col in df.columns]
    
    if len(existing_feature_cols) < 3:
        st.warning("Not enough features available for clustering analysis")
        return df, None
    
    # Prepare data
    X_cluster = df[existing_feature_cols].fillna(0)
    X_cluster_scaled = st.session_state.scaler.fit_transform(X_cluster)
    
    # Determine optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X_cluster_scaled)
        wcss.append(kmeans.inertia_)
    
    # Fit KMeans with 4 clusters (Dormant, Frequent Low, Hoarders, Active)
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    
    # Add cluster labels to the dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Define cluster names based on characteristics
    cluster_names = {
        0: "Dormant Users",
        1: "Frequent Low Users",
        2: "Hoarders",
        3: "Active Users"
    }
    
    df_clustered['Cluster_Name'] = df_clustered['Cluster'].map(cluster_names)
    
    return df_clustered, kmeans

# Define function for risk explanation
def explain_risk(instance, model, X_train, feature_names):
    """Explain why a prediction is high or low risk"""
    try:
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Get SHAP values for the instance
        shap_values = explainer.shap_values(instance)
        
        # If it's a multi-class output, get the values for the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create a dataframe of feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_values,
            'Absolute_Value': np.abs(shap_values)
        }).sort_values('Absolute_Value', ascending=False)
        
        # Get the top 5 contributing factors
        top_factors = feature_importance.head(5)
        
        # Create explanation text
        explanation = []
        for _, row in top_factors.iterrows():
            feature = row['Feature']
            value = instance[feature].values[0] if feature in instance.columns else 0
            shap_val = row['SHAP_Value']
            
            if shap_val > 0:
                explanation.append(f"**{feature}**: {value:.2f} increases fraud risk")
            else:
                explanation.append(f"**{feature}**: {value:.2f} decreases fraud risk")
        
        return explanation, feature_importance
    except Exception as e:
        # Fallback to feature importance if SHAP fails
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_factors = feature_importance.head(5)
        
        explanation = []
        for _, row in top_factors.iterrows():
            feature = row['Feature']
            importance = row['Importance']
            value = instance[feature].values[0] if feature in instance.columns else 0
            
            if importance > 0.05:  # Arbitrary threshold for "high" importance
                explanation.append(f"**{feature}**: {value:.2f} is a significant factor in this prediction")
        
        return explanation, feature_importance

# Define function to create download link
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Define function to create interactive plots
def create_plotly_correlation_matrix(correlation_matrix):
    """Create an interactive correlation matrix using Plotly"""
    if correlation_matrix is None:
        return None
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix of Key Variables'
    )
    
    fig.update_layout(
        width=800,
        height=600
    )
    
    return fig

# Main app
def main():
    # Sidebar
    st.sidebar.title("Subsidy Tracker Dashboard")
    st.sidebar.markdown("### Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select a page",
        ["📊 Overview", "🔍 Exploratory Analysis", "🤖 Fraud Detection", "🧩 Clustering", "🔮 Single Prediction", "📊 Batch Prediction"]
    )
    
    # Data upload section
    st.sidebar.markdown("### Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload a CSV file with subsidy data"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load data from uploaded file
        df = load_data(tmp_file_path)
        
        if df is not None:
            # Clean and preprocess data
            df = data_cleaning_integration(df)
            
            # Calculate metrics
            metrics = calculate_subsidy_metrics(df)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.metrics = metrics
            st.session_state.data_loaded = True
            
            st.sidebar.success("Data loaded successfully!")
    else:
        # Use sample data if no file is uploaded
        if not st.session_state.data_loaded:
            if st.sidebar.button("Use Sample Data"):
                df = load_data()
                
                if df is not None:
                    # Clean and preprocess data
                    df = data_cleaning_integration(df)
                    
                    # Calculate metrics
                    metrics = calculate_subsidy_metrics(df)
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.metrics = metrics
                    st.session_state.data_loaded = True
                    
                    st.sidebar.success("Sample data loaded successfully!")
    
    # Display data info in sidebar
    if st.session_state.data_loaded:
        st.sidebar.markdown("### Data Information")
        st.sidebar.write(f"Records: {len(st.session_state.df):,}")
        st.sidebar.write(f"Columns: {len(st.session_state.df.columns)}")
        
        # Display metrics in sidebar
        if st.session_state.metrics:
            st.sidebar.markdown("### Key Metrics")
            st.sidebar.write(f"Coverage Rate: {st.session_state.metrics.get('subsidy_coverage_rate', 0):.1%}")
            st.sidebar.write(f"Targeting Accuracy: {st.session_state.metrics.get('targeting_accuracy', 0):.1%}")
            st.sidebar.write(f"Leakage Rate: {st.session_state.metrics.get('leakage_rate', 0):.1%}")
            st.sidebar.write(f"Fraud Rate: {st.session_state.metrics.get('fraud_rate', 0):.1%}")
    
    # Main content based on selected page
    if page == "📊 Overview":
        display_overview()
    elif page == "🔍 Exploratory Analysis":
        display_exploratory_analysis()
    elif page == "🤖 Fraud Detection":
        display_fraud_detection()
    elif page == "🧩 Clustering":
        display_clustering()
    elif page == "🔮 Single Prediction":
        display_single_prediction()
    elif page == "📊 Batch Prediction":
        display_batch_prediction()

def display_overview():
    """Display the overview page with key metrics"""
    st.markdown("<h1 class='main-header'>Subsidy Tracker Overview</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or use sample data to view the dashboard.")
        return
    
    df = st.session_state.df
    metrics = st.session_state.metrics
    
    # Key metrics cards
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_beneficiaries = len(df)
        st.metric(
            label="Total Beneficiaries",
            value=f"{total_beneficiaries:,}",
            delta=None
        )
    
    with col2:
        total_amount = df['Amount (NGN)'].sum()
        st.metric(
            label="Total Subsidy (NGN)",
            value=f"₦{total_amount:,.0f}",
            delta=None
        )
    
    with col3:
        avg_subsidy = df['Amount (NGN)'].mean()
        st.metric(
            label="Avg. Subsidy (NGN)",
            value=f"₦{avg_subsidy:,.0f}",
            delta=None
        )
    
    with col4:
        fraud_rate = metrics.get('fraud_rate', 0) * 100
        st.metric(
            label="Fraud Rate",
            value=f"{fraud_rate:.2f}%",
            delta=None
        )
    
    # Additional metrics
    st.markdown("### Program Effectiveness")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        coverage_rate = metrics.get('subsidy_coverage_rate', 0) * 100
        st.metric(
            label="Coverage Rate",
            value=f"{coverage_rate:.1f}%",
            delta=None
        )
    
    with col2:
        targeting_accuracy = metrics.get('targeting_accuracy', 0) * 100
        st.metric(
            label="Targeting Accuracy",
            value=f"{targeting_accuracy:.1f}%",
            delta=None
        )
    
    with col3:
        leakage_rate = metrics.get('leakage_rate', 0) * 100
        st.metric(
            label="Leakage Rate",
            value=f"{leakage_rate:.1f}%",
            delta=None
        )
    
    # Regional distribution
    st.markdown("### Regional Distribution")
    
    if 'Region' in df.columns and 'Amount (NGN)' in df.columns:
        regional_data = df.groupby('Region')['Amount (NGN)'].sum().reset_index()
        
        fig = px.bar(
            regional_data,
            x='Region',
            y='Amount (NGN)',
            title='Total Subsidy Amount by Region',
            color='Region',
            labels={'Amount (NGN)': 'Total Amount (NGN)'}
        )
        
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Total Amount (NGN)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Income level distribution
    st.markdown("### Income Level Distribution")
    
    if 'Income_Level' in df.columns and 'Amount (NGN)' in df.columns:
        income_data = df.groupby('Income_Level')['Amount (NGN)'].sum().reset_index()
        
        fig = px.pie(
            income_data,
            values='Amount (NGN)',
            names='Income_Level',
            title='Subsidy Distribution by Income Level'
        )
        
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud by region
    if 'Region' in df.columns and 'Suspected_Fraud' in df.columns:
        st.markdown("### Fraud Rate by Region")
        
        fraud_by_region = df.groupby('Region')['Suspected_Fraud'].mean().reset_index()
        fraud_by_region['Fraud Rate'] = fraud_by_region['Suspected_Fraud'] * 100
        
        fig = px.bar(
            fraud_by_region,
            x='Region',
            y='Fraud Rate',
            title='Fraud Rate by Region',
            color='Region',
            labels={'Fraud Rate': 'Fraud Rate (%)'}
        )
        
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title="Fraud Rate (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_exploratory_analysis():
    """Display the exploratory analysis page"""
    st.markdown("<h1 class='main-header'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or use sample data to view the dashboard.")
        return
    
    df = st.session_state.df
    
    # Correlation analysis
    st.markdown("### Correlation Analysis")
    
    correlation_matrix = correlation_analysis(df)
    
    if correlation_matrix is not None:
        fig = create_plotly_correlation_matrix(correlation_matrix)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns available for correlation analysis.")
    
    # Distribution analysis
    st.markdown("### Distribution Analysis")
    
    # Create tabs for different distribution analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Age Distribution", "Subsidy Amount", "Energy Consumption", "Household Dependents"])
    
    with tab1:
        if 'Age' in df.columns:
            fig = px.histogram(
                df,
                x='Age',
                nbins=20,
                title='Age Distribution of Beneficiaries',
                labels={'Age': 'Age', 'count': 'Count'}
            )
            
            fig.update_layout(
                xaxis_title="Age",
                yaxis_title="Count",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Age data not available.")
    
    with tab2:
        if 'Amount (NGN)' in df.columns:
            fig = px.histogram(
                df,
                x='Amount (NGN)',
                nbins=30,
                title='Distribution of Subsidy Amounts',
                labels={'Amount (NGN)': 'Amount (NGN)', 'count': 'Count'}
            )
            
            fig.update_layout(
                xaxis_title="Amount (NGN)",
                yaxis_title="Count",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Subsidy amount data not available.")
    
    with tab3:
        if 'Monthly_Energy_Consumption_kWh' in df.columns:
            fig = px.histogram(
                df,
                x='Monthly_Energy_Consumption_kWh',
                nbins=30,
                title='Distribution of Monthly Energy Consumption',
                labels={'Monthly_Energy_Consumption_kWh': 'Energy Consumption (kWh)', 'count': 'Count'}
            )
            
            fig.update_layout(
                xaxis_title="Monthly Energy Consumption (kWh)",
                yaxis_title="Count",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Energy consumption data not available.")
    
    with tab4:
        if 'Household_Dependents' in df.columns:
            fig = px.histogram(
                df,
                x='Household_Dependents',
                nbins=10,
                title='Distribution of Household Dependents',
                labels={'Household_Dependents': 'Number of Dependents', 'count': 'Count'}
            )
            
            fig.update_layout(
                xaxis_title="Number of Dependents",
                yaxis_title="Count",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Household dependents data not available.")
    
    # Relationship analysis
    st.markdown("### Relationship Analysis")
    
    # Create tabs for different relationship analyses
    tab1, tab2, tab3 = st.tabs(["Energy vs. Subsidy", "Dependents vs. Subsidy", "Wallet Balance vs. Days Since Transaction"])
    
    with tab1:
        if 'Monthly_Energy_Consumption_kWh' in df.columns and 'Amount (NGN)' in df.columns:
            fig = px.scatter(
                df,
                x='Monthly_Energy_Consumption_kWh',
                y='Amount (NGN)',
                title='Energy Consumption vs. Subsidy Amount',
                labels={
                    'Monthly_Energy_Consumption_kWh': 'Monthly Energy Consumption (kWh)',
                    'Amount (NGN)': 'Subsidy Amount (NGN)'
                },
                opacity=0.7
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Energy consumption or subsidy amount data not available.")
    
    with tab2:
        if 'Household_Dependents' in df.columns and 'Amount (NGN)' in df.columns:
            # Calculate average subsidy by number of dependents
            dep_subsidy = df.groupby('Household_Dependents')['Amount (NGN)'].mean().reset_index()
            
            fig = px.line(
                dep_subsidy,
                x='Household_Dependents',
                y='Amount (NGN)',
                title='Average Subsidy by Number of Dependents',
                markers=True,
                labels={
                    'Household_Dependents': 'Number of Dependents',
                    'Amount (NGN)': 'Average Subsidy (NGN)'
                }
            )
            
            fig.update_layout(
                xaxis_title="Number of Dependents",
                yaxis_title="Average Subsidy (NGN)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Household dependents or subsidy amount data not available.")
    
    with tab3:
        if 'Wallet_Balance (NGN)' in df.columns and 'Days_Since_Last_Transaction' in df.columns:
            fig = px.scatter(
                df,
                x='Days_Since_Last_Transaction',
                y='Wallet_Balance (NGN)',
                title='Wallet Balance vs. Days Since Last Transaction',
                labels={
                    'Days_Since_Last_Transaction': 'Days Since Last Transaction',
                    'Wallet_Balance (NGN)': 'Wallet Balance (NGN)'
                },
                opacity=0.7
            )
            
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Wallet balance or days since last transaction data not available.")
    
    # Time series analysis
    if 'Date' in df.columns:
        st.markdown("### Time Series Analysis")
        
        # Group by month
        monthly_data = df.groupby(df['Date'].dt.to_period('M')).agg({
            'Amount (NGN)': 'sum',
            'National_ID': 'count',
            'Suspected_Fraud': 'mean'
        }).reset_index()
        
        monthly_data['Date'] = monthly_data['Date'].astype(str)
        monthly_data['Fraud Rate'] = monthly_data['Suspected_Fraud'] * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Subsidy Distribution', 'Monthly Fraud Rate'),
            vertical_spacing=0.1
        )
        
        # Add subsidy distribution trace
        fig.add_trace(
            go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['Amount (NGN)'],
                mode='lines+markers',
                name='Subsidy Amount',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Add fraud rate trace
        fig.add_trace(
            go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['Fraud Rate'],
                mode='lines+markers',
                name='Fraud Rate',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Amount (NGN)", row=1, col=1)
        fig.update_yaxes(title_text="Fraud Rate (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def display_fraud_detection():
    """Display the fraud detection page"""
    st.markdown("<h1 class='main-header'>Fraud Detection</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or use sample data to view the dashboard.")
        return
    
    df = st.session_state.df
    
    # Run fraud detection models if not already run
    if not st.session_state.fraud_models:
        with st.spinner("Running fraud detection models..."):
            # Run supervised fraud detection
            fraud_results, best_model_name = supervised_fraud_detection(df)
            
            # Store in session state
            st.session_state.fraud_models = fraud_results
            st.session_state.best_fraud_model = best_model_name
    
    # Run anomaly detection if not already run
    if st.session_state.anomaly_model is None:
        with st.spinner("Running anomaly detection..."):
            # Run unsupervised anomaly detection
            df_encoded, anomaly_model = anomaly_detection_unsupervised(df)
            
            # Store in session state
            st.session_state.df_encoded = df_encoded
            st.session_state.anomaly_model = anomaly_model
    
    fraud_results = st.session_state.fraud_models
    best_model_name = st.session_state.best_fraud_model
    df_encoded = st.session_state.df_encoded
    
    # Model performance comparison
    st.markdown("### Model Performance Comparison")
    
    # Create a dataframe for model performance
    model_performance = []
    for name, result in fraud_results.items():
        model_performance.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score']
        })
    
    model_performance_df = pd.DataFrame(model_performance)
    
    # Display model performance table
    st.dataframe(model_performance_df.style.highlight_max(subset=['F1-Score'], color='lightgreen'))
    
    # Feature importance for the best model
    st.markdown(f"### Feature Importance - {best_model_name}")
    
    if best_model_name == 'Random Forest':
        best_model = fraud_results[best_model_name]['model']
        X_train = fraud_results[best_model_name]['X_train']
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importances',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix for the best model
    st.markdown(f"### Confusion Matrix - {best_model_name}")
    
    best_result = fraud_results[best_model_name]
    y_test = best_result['y_test']
    y_pred = best_result['y_pred']
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a heatmap for the confusion matrix
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        title='Confusion Matrix',
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    
    fig.update_xaxes(ticktext=['Normal', 'Fraud'], tickvals=[0, 1])
    fig.update_yaxes(ticktext=['Normal', 'Fraud'], tickvals=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Fraud predictions
    st.markdown("### Fraud Predictions")
    
    # Add fraud predictions to the original dataframe
    df_with_predictions = df.copy()
    
    # Initialize prediction columns with default values
    df_with_predictions['Predicted_Fraud'] = 0
    df_with_predictions['Fraud_Probability'] = 0.0
    
    # Get the test indices and assign predictions only to test data
    X_test = best_result['X_test']
    test_indices = X_test.index
    
    df_with_predictions.loc[test_indices, 'Predicted_Fraud'] = best_result['y_pred']
    df_with_predictions.loc[test_indices, 'Fraud_Probability'] = best_result['y_pred_proba']
    
    # Add anomaly predictions
    df_with_predictions['Anomaly_Predicted'] = df_encoded['Anomaly_Predicted']
    df_with_predictions['Anomaly_Score'] = df_encoded['Anomaly_Score']
    
    # Create a combined risk score
    df_with_predictions['Combined_Risk_Score'] = (
        0.7 * df_with_predictions['Fraud_Probability'] + 
        0.3 * (df_with_predictions['Anomaly_Score'] + 0.5) / 1.0  # Normalize anomaly score to 0-1
    )
    
    # Categorize risk
    df_with_predictions['Risk_Category'] = pd.cut(
        df_with_predictions['Combined_Risk_Score'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Display high-risk cases
    st.markdown("#### High-Risk Cases")
    
    high_risk = df_with_predictions[df_with_predictions['Risk_Category'] == 'High Risk'].sort_values('Combined_Risk_Score', ascending=False)
    
    if len(high_risk) > 0:
        # Show top 10 high-risk cases
        st.dataframe(high_risk.head(10)[[
            'National_ID', 'Age', 'Gender', 'Region', 'Income_Level', 
            'Amount (NGN)', 'Combined_Risk_Score', 'Risk_Category'
        ]])
        
        # Allow user to select a case for detailed analysis
        st.markdown("#### Risk Analysis Explanation")
        
        selected_id = st.selectbox(
            "Select a beneficiary ID for detailed risk analysis:",
            high_risk['National_ID'].unique()
        )
        
        if selected_id:
            # Get the selected case
            selected_case = high_risk[high_risk['National_ID'] == selected_id].iloc[0]
            
            # Get the model and training data
            best_model = fraud_results[best_model_name]['model']
            X_train = fraud_results[best_model_name]['X_train']
            
            # Get the instance for explanation
            instance = X_train[X_train.index.isin([selected_case.name])]
            
            if len(instance) > 0:
                # Explain the prediction
                explanation, feature_importance = explain_risk(
                    instance, best_model, X_train, X_train.columns.tolist()
                )
                
                # Display the risk score
                risk_score = selected_case['Combined_Risk_Score']
                
                if risk_score >= 0.7:
                    risk_class = "risk-high"
                    risk_text = "High Risk"
                elif risk_score >= 0.3:
                    risk_class = "risk-medium"
                    risk_text = "Medium Risk"
                else:
                    risk_class = "risk-low"
                    risk_text = "Low Risk"
                
                st.markdown(f"### Risk Analysis for {selected_id}")
                st.markdown(f"<p class='{risk_class}'>Risk Category: {risk_text} (Score: {risk_score:.3f})</p>", unsafe_allow_html=True)
                
                # Display explanation
                st.markdown("#### Top Contributing Factors:")
                
                for i, factor in enumerate(explanation[:5]):
                    st.markdown(f"{i+1}. {factor}")
                
                # Display feature importance plot
                st.markdown("#### Feature Importance:")
                
                fig = px.bar(
                    feature_importance.head(10),
                    x='Importance' if 'Importance' in feature_importance.columns else 'Absolute_Value',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Feature Importances for this Prediction',
                    color='Importance' if 'Importance' in feature_importance.columns else 'Absolute_Value',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not find the selected case in the training data.")
    else:
        st.info("No high-risk cases detected.")
    
    # Download fraud report
    st.markdown("### Export Fraud Report")
    
    if st.button("Generate Fraud Report"):
        # Create a report dataframe
        fraud_report = df_with_predictions[[
            'National_ID', 'Age', 'Gender', 'Region', 'Income_Level', 
            'Amount (NGN)', 'Predicted_Fraud', 'Fraud_Probability', 
            'Anomaly_Predicted', 'Anomaly_Score', 'Combined_Risk_Score', 'Risk_Category'
        ]].sort_values('Combined_Risk_Score', ascending=False)
        
        # Create a CSV file
        csv = fraud_report.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="fraud_report.csv">Download Fraud Report</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_clustering():
    """Display the clustering analysis page"""
    st.markdown("<h1 class='main-header'>Behavioral Clustering Analysis</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or use sample data to view the dashboard.")
        return
    
    df = st.session_state.df
    
    # Run clustering if not already run
    if st.session_state.cluster_model is None:
        with st.spinner("Running clustering analysis..."):
            # Run KMeans clustering
            df_clustered, cluster_model = behavioral_clustering(df)
            
            # Store in session state
            st.session_state.df_clustered = df_clustered
            st.session_state.cluster_model = cluster_model
    
    df_clustered = st.session_state.df_clustered
    cluster_model = st.session_state.cluster_model
    
    if cluster_model is None:
        st.warning("Clustering analysis could not be performed due to insufficient features.")
        return
    
    # Cluster distribution
    st.markdown("### Cluster Distribution")
    
    cluster_counts = df_clustered['Cluster_Name'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    fig = px.bar(
        cluster_counts,
        x='Cluster',
        y='Count',
        title='Distribution of Behavioral Clusters',
        color='Cluster',
        labels={'Count': 'Number of Beneficiaries'}
    )
    
    fig.update_layout(
        xaxis_title="Behavioral Cluster",
        yaxis_title="Number of Beneficiaries",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster characteristics
    st.markdown("### Cluster Characteristics")
    
    # Calculate cluster characteristics
    cluster_characteristics = df_clustered.groupby('Cluster_Name').agg({
        'Amount (NGN)': ['mean', 'std'],
        'Wallet_Balance (NGN)': ['mean', 'std'],
        'Days_Since_Last_Transaction': ['mean', 'std'],
        'Monthly_Energy_Consumption_kWh': ['mean', 'std'],
        'Household_Dependents': ['mean', 'std'],
        'Suspected_Fraud': 'mean'
    }).round(2)
    
    # Flatten multi-level columns
    cluster_characteristics.columns = ['_'.join(col).strip() for col in cluster_characteristics.columns.values]
    
    # Display cluster characteristics
    st.dataframe(cluster_characteristics)
    
    # Cluster visualization
    st.markdown("### Cluster Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Amount vs. Wallet Balance", "Energy vs. Dependents"])
    
    with tab1:
        fig = px.scatter(
            df_clustered,
            x='Amount (NGN)',
            y='Wallet_Balance (NGN)',
            color='Cluster_Name',
            title='Amount vs. Wallet Balance by Cluster',
            labels={
                'Amount (NGN)': 'Subsidy Amount (NGN)',
                'Wallet_Balance (NGN)': 'Wallet Balance (NGN)'
            },
            opacity=0.7
        )
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.scatter(
            df_clustered,
            x='Monthly_Energy_Consumption_kWh',
            y='Household_Dependents',
            color='Cluster_Name',
            title='Energy Consumption vs. Household Dependents by Cluster',
            labels={
                'Monthly_Energy_Consumption_kWh': 'Energy Consumption (kWh)',
                'Household_Dependents': 'Number of Dependents'
            },
            opacity=0.7
        )
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    st.markdown("### Cluster Analysis")
    
    # Allow user to select a cluster for detailed analysis
    selected_cluster = st.selectbox(
        "Select a cluster for detailed analysis:",
        df_clustered['Cluster_Name'].unique()
    )
    
    if selected_cluster:
        # Filter data for the selected cluster
        cluster_data = df_clustered[df_clustered['Cluster_Name'] == selected_cluster]
        
        # Display cluster statistics
        st.markdown(f"#### {selected_cluster} Cluster")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Beneficiaries",
                value=f"{len(cluster_data):,}"
            )
        
        with col2:
            avg_amount = cluster_data['Amount (NGN)'].mean()
            st.metric(
                label="Avg. Subsidy (NGN)",
                value=f"₦{avg_amount:,.0f}"
            )
        
        with col3:
            avg_wallet = cluster_data['Wallet_Balance (NGN)'].mean()
            st.metric(
                label="Avg. Wallet (NGN)",
                value=f"₦{avg_wallet:,.0f}"
            )
        
        with col4:
            fraud_rate = cluster_data['Suspected_Fraud'].mean() * 100
            st.metric(
                label="Fraud Rate",
                value=f"{fraud_rate:.2f}%"
            )
        
        # Display cluster description based on characteristics
        st.markdown("#### Cluster Description")
        
        if selected_cluster == "Dormant Users":
            st.markdown("""
            **Dormant Users** are beneficiaries who:
            - Have low wallet activity
            - Haven't made recent transactions
            - May have low energy consumption
            - Could be at risk of financial exclusion
            """)
        elif selected_cluster == "Frequent Low Users":
            st.markdown("""
            **Frequent Low Users** are beneficiaries who:
            - Make regular transactions but with low amounts
            - Have moderate wallet balances
            - May be using subsidies for essential needs only
            - Typically have lower fraud risk
            """)
        elif selected_cluster == "Hoarders":
            st.markdown("""
            **Hoarders** are beneficiaries who:
            - Accumulate high wallet balances
            - Make infrequent transactions
            - May not be using subsidies as intended
            - Could indicate potential fraud or misuse
            """)
        elif selected_cluster == "Active Users":
            st.markdown("""
            **Active Users** are beneficiaries who:
            - Make regular transactions with moderate to high amounts
            - Maintain healthy wallet balances
            - Use subsidies as intended
            - Typically have lower fraud risk
            """)
        
        # Display sample beneficiaries from the cluster
        st.markdown("#### Sample Beneficiaries")
        
        st.dataframe(cluster_data.head(10)[[
            'National_ID', 'Age', 'Gender', 'Region', 'Income_Level', 
            'Amount (NGN)', 'Wallet_Balance (NGN)', 'Days_Since_Last_Transaction'
        ]])

def display_single_prediction():
    """Display single transaction fraud prediction page"""
    st.markdown("<h1 class='main-header'>🔮 Single Transaction Fraud Prediction</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or use sample data to train the fraud detection models.")
        return
    
    st.markdown("### Enter Transaction Details")
    st.markdown("Fill in the transaction information below to get a fraud risk assessment.")
    
    # Create input form
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Personal Information**")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["Lagos", "Kano", "Rivers", "Oyo", "Kaduna", "Yobe"])
            income_level = st.selectbox("Income Level", ["Low", "Middle", "High"])
            household_dependents = st.number_input("Household Dependents", min_value=0, max_value=20, value=3)
        
        with col2:
            st.markdown("**Transaction Information**")
            amount = st.number_input("Transaction Amount (NGN)", min_value=0.0, value=4000.0, step=100.0)
            channel = st.selectbox("Transaction Channel", ["Bank Transfer", "Mobile Money", "Cash Pickup"])
            subsidy_type = st.selectbox("Subsidy Type", ["Energy", "Food", "Housing"])
            wallet_balance = st.number_input("Current Wallet Balance (NGN)", min_value=0.0, value=5000.0, step=100.0)
            avg_monthly_balance = st.number_input("Average Monthly Wallet Balance (NGN)", min_value=0.0, value=5000.0, step=100.0)
        
        with col3:
            st.markdown("**Behavioral Information**")
            days_since_last = st.number_input("Days Since Last Transaction", min_value=0, max_value=365, value=7)
            energy_consumption = st.number_input("Monthly Energy Consumption (kWh)", min_value=0.0, value=100.0, step=10.0)
            wallet_status = st.selectbox("Wallet Activity Status", ["Active", "Inactive", "Suspicious"])
            subsidy_eligibility = st.selectbox("Subsidy Eligibility", ["Eligible", "Not Eligible"])
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Fraud Risk", type="primary")
    
    if submitted:
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'Region': region,
            'Income_Level': income_level,
            'Household_Dependents': household_dependents,
            'Monthly_Energy_Consumption_kWh': energy_consumption,
            'Amount (NGN)': amount,
            'Wallet_Balance (NGN)': wallet_balance,
            'Days_Since_Last_Transaction': days_since_last,
            'Channel': channel,
            'Subsidy_Type': subsidy_type,
            'Wallet_Activity_Status': wallet_status,
            'Avg_Monthly_Wallet_Balance': avg_monthly_balance,
            'Subsidy_Eligibility': 1 if subsidy_eligibility == "Eligible" else 0
        }
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Add engineered features
        input_df['Amount_per_Dependent'] = input_df['Amount (NGN)'] / max(input_df['Household_Dependents'].iloc[0], 1)
        input_df['Energy_per_Dependent'] = input_df['Monthly_Energy_Consumption_kWh'] / max(input_df['Household_Dependents'].iloc[0], 1)
        input_df['Days_Since_Distribution'] = input_df['Days_Since_Last_Transaction']  # Simplified
        
        # Make prediction using existing fraud detection logic
        try:
            # Train models if not already trained
            if not st.session_state.fraud_models:
                with st.spinner("Training fraud detection models..."):
                    fraud_results, best_model_name = supervised_fraud_detection(st.session_state.df)
                    st.session_state.fraud_models = fraud_results
                    st.session_state.best_fraud_model = best_model_name
            
            # Get prediction and probability
            fraud_prob, explanation, recommendations = predict_single_fraud(input_df)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Fraud Risk Assessment")
            
            # Risk level and probability
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if fraud_prob >= 0.7:
                    risk_level = "HIGH"
                    risk_class = "risk-high"
                    risk_color = "#D32F2F"
                elif fraud_prob >= 0.4:
                    risk_level = "MEDIUM"
                    risk_class = "risk-medium"
                    risk_color = "#FFA000"
                else:
                    risk_level = "LOW"
                    risk_class = "risk-low"
                    risk_color = "#388E3C"
                
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; border-left: 5px solid {risk_color};">
                    <h2 class="{risk_class}">FRAUD RISK: {risk_level}</h2>
                    <h1 style="color: {risk_color}; margin: 0;">{fraud_prob:.1%}</h1>
                    <p style="margin: 0; color: #666;">Probability of Fraud</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Risk explanation
                st.markdown("### 🔍 Risk Analysis")
                if explanation:
                    for i, factor in enumerate(explanation[:5], 1):
                        st.markdown(f"**{i}.** {factor}")
                else:
                    st.markdown("No specific risk factors identified.")
            
            # Recommendations section
            st.markdown("---")
            st.markdown("## 💡 Recommendations")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            else:
                if fraud_prob >= 0.7:
                    st.markdown("• **Immediate Action Required**: Flag this transaction for manual review")
                    st.markdown("• **Enhanced Verification**: Require additional identity verification")
                    st.markdown("• **Transaction Monitoring**: Monitor future transactions from this beneficiary")
                elif fraud_prob >= 0.4:
                    st.markdown("• **Moderate Monitoring**: Include in routine fraud monitoring")
                    st.markdown("• **Pattern Analysis**: Check for similar transaction patterns")
                else:
                    st.markdown("• **Low Risk**: Process transaction normally")
                    st.markdown("• **Routine Monitoring**: Continue standard monitoring procedures")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure the fraud detection models are properly trained.")

def predict_single_fraud(input_df):
    """Predict fraud for a single transaction"""
    try:
        # Get the trained model (use Random Forest if available)
        if 'Random Forest' in st.session_state.fraud_models:
            model_info = st.session_state.fraud_models['Random Forest']
            model = model_info['model']
            training_features = model_info['feature_names']
        elif st.session_state.fraud_models:
            model_info = list(st.session_state.fraud_models.values())[0]
            model = model_info['model']
            training_features = model_info['feature_names']
        else:
            raise ValueError("No trained fraud detection models available")
        
        # Get the original training data to match feature encoding
        df = st.session_state.df
        
        # Encode categorical variables using the same approach as training
        input_encoded = input_df.copy()
        for col in ['Gender', 'Region', 'Income_Level', 'Channel', 'Subsidy_Type']:
            if col in input_encoded.columns:
                dummies = pd.get_dummies(input_encoded[col], prefix=col)
                input_encoded = pd.concat([input_encoded, dummies], axis=1)
        
        # Align input features with the exact training features
        for feature in training_features:
            if feature not in input_encoded.columns:
                input_encoded[feature] = 0
        
        # Select only the exact training features in the same order
        input_final = input_encoded[training_features].fillna(0)
        
        # Make prediction
        fraud_prob = model.predict_proba(input_final)[0][1]  # Probability of fraud (class 1)
        
        # Generate explanation based on feature values
        explanation = generate_fraud_explanation(input_df.iloc[0], fraud_prob)
        
        # Generate recommendations
        recommendations = generate_fraud_recommendations(input_df.iloc[0], fraud_prob)
        
        return fraud_prob, explanation, recommendations
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.0, [], []

def generate_fraud_explanation(transaction, fraud_prob):
    """Generate explanation for fraud prediction"""
    explanation = []
    
    # Amount-based factors
    if transaction['Amount (NGN)'] > 8000:
        explanation.append(f"**High Transaction Amount**: ₦{transaction['Amount (NGN)']:,.0f} is above average, increasing fraud risk")
    elif transaction['Amount (NGN)'] < 1000:
        explanation.append(f"**Low Transaction Amount**: ₦{transaction['Amount (NGN)']:,.0f} is unusually low, which may indicate testing behavior")
    
    # Wallet balance factors
    if transaction['Wallet_Balance (NGN)'] > transaction['Avg_Monthly_Wallet_Balance'] * 2:
        explanation.append(f"**Unusual Wallet Balance**: Current balance is significantly higher than average monthly balance")
    
    # Behavioral factors
    if transaction['Days_Since_Last_Transaction'] > 60:
        explanation.append(f"**Long Inactivity Period**: {transaction['Days_Since_Last_Transaction']} days since last transaction suggests irregular usage")
    elif transaction['Days_Since_Last_Transaction'] == 0:
        explanation.append(f"**Immediate Repeat Transaction**: Same-day transactions may indicate suspicious activity")
    
    # Demographic factors
    if transaction['Age'] < 25 or transaction['Age'] > 65:
        explanation.append(f"**Age Factor**: Age {transaction['Age']} is in a demographic with different risk patterns")
    
    # Household factors
    if transaction['Household_Dependents'] > 8:
        explanation.append(f"**Large Household**: {transaction['Household_Dependents']} dependents is unusually high")
    elif transaction['Household_Dependents'] == 0:
        explanation.append(f"**No Dependents**: Zero dependents may indicate eligibility issues")
    
    # Channel factors
    if transaction['Channel'] == 'Cash Pickup':
        explanation.append(f"**Cash Pickup Channel**: This channel has higher fraud risk compared to digital channels")
    
    # Wallet status factors
    if transaction['Wallet_Activity_Status'] == 'Suspicious':
        explanation.append(f"**Suspicious Wallet Status**: Account has been flagged for suspicious activity")
    elif transaction['Wallet_Activity_Status'] == 'Inactive':
        explanation.append(f"**Inactive Wallet**: Low activity level may indicate account misuse")
    
    return explanation

def generate_fraud_recommendations(transaction, fraud_prob):
    """Generate recommendations based on fraud risk"""
    recommendations = []
    
    if fraud_prob >= 0.7:
        recommendations.extend([
            "**Immediate Manual Review**: Flag transaction for urgent investigation",
            "**Enhanced KYC**: Require additional identity verification documents",
            "**Transaction Hold**: Temporarily hold transaction pending verification",
            "**Account Monitoring**: Place account under enhanced monitoring for 30 days"
        ])
    elif fraud_prob >= 0.4:
        recommendations.extend([
            "**Routine Review**: Include in next batch review cycle",
            "**Pattern Analysis**: Check for similar transactions from this beneficiary",
            "**Automated Monitoring**: Enable automated alerts for future transactions"
        ])
    else:
        recommendations.extend([
            "**Standard Processing**: Process transaction through normal channels",
            "**Routine Monitoring**: Continue standard monitoring procedures"
        ])
    
    # Specific recommendations based on risk factors
    if transaction['Amount (NGN)'] > 8000:
        recommendations.append("**Amount Verification**: Verify legitimacy of high-value transaction")
    
    if transaction['Days_Since_Last_Transaction'] > 60:
        recommendations.append("**Reactivation Check**: Verify account holder identity after long inactivity")
    
    if transaction['Wallet_Activity_Status'] == 'Suspicious':
        recommendations.append("**Account Investigation**: Conduct thorough investigation of account history")
    
    return recommendations

def display_batch_prediction():
    """Display batch transaction fraud prediction page"""
    st.markdown("<h1 class='main-header'>📊 Batch Transaction Fraud Prediction</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data or use sample data to train the fraud detection models.")
        return
    
    st.markdown("### Upload Transaction Batch")
    st.markdown("Upload a CSV file containing multiple transactions to get fraud risk assessments for all transactions.")
    
    # File upload
    uploaded_batch = st.file_uploader(
        "Choose CSV file for batch prediction",
        type=["csv"],
        help="Upload a CSV file with transaction data. Required columns: Age, Gender, Region, Income_Level, Household_Dependents, Amount (NGN), etc."
    )
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        
        **Required Columns:**
        - `Age`: Beneficiary age (18-100)
        - `Gender`: Male or Female
        - `Region`: Lagos, Kano, Rivers, Oyo, Kaduna, or Yobe
        - `Income_Level`: Low, Middle, or High
        - `Household_Dependents`: Number of dependents (0-20)
        - `Amount (NGN)`: Transaction amount
        - `Wallet_Balance (NGN)`: Current wallet balance
        - `Days_Since_Last_Transaction`: Days since last transaction
        - `Channel`: Bank Transfer, Mobile Money, or Cash Pickup
        - `Subsidy_Type`: Energy, Food, or Housing
        - `Wallet_Activity_Status`: Active, Inactive, or Suspicious
        - `Monthly_Energy_Consumption_kWh`: Energy consumption
        - `Avg_Monthly_Wallet_Balance`: Average monthly balance
        - `Subsidy_Eligibility`: 1 (Eligible) or 0 (Not Eligible)
        
        **Optional Columns:**
        - `National_ID`: Beneficiary identifier
        - `Transaction_ID`: Transaction identifier
        """)
        
        # Show sample data
        sample_data = pd.DataFrame({
            'National_ID': ['ID00001', 'ID00002', 'ID00003'],
            'Age': [35, 42, 28],
            'Gender': ['Male', 'Female', 'Male'],
            'Region': ['Lagos', 'Kano', 'Rivers'],
            'Income_Level': ['Low', 'Middle', 'Low'],
            'Household_Dependents': [3, 5, 2],
            'Amount (NGN)': [4000, 6500, 3200],
            'Wallet_Balance (NGN)': [5000, 7200, 4100],
            'Days_Since_Last_Transaction': [7, 15, 3],
            'Channel': ['Bank Transfer', 'Mobile Money', 'Cash Pickup'],
            'Subsidy_Type': ['Energy', 'Food', 'Housing'],
            'Wallet_Activity_Status': ['Active', 'Active', 'Inactive'],
            'Monthly_Energy_Consumption_kWh': [100, 150, 80],
            'Avg_Monthly_Wallet_Balance': [5000, 7000, 4000],
            'Subsidy_Eligibility': [1, 1, 1]
        })
        
        st.markdown("**Sample CSV Format:**")
        st.dataframe(sample_data)
        
        # Download sample template
        csv_template = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample Template",
            data=csv_template,
            file_name="batch_prediction_template.csv",
            mime="text/csv"
        )
    
    if uploaded_batch is not None:
        try:
            # Load the batch data
            batch_df = pd.read_csv(uploaded_batch)
            
            st.markdown(f"### 📊 Batch Data Preview ({len(batch_df)} transactions)")
            st.dataframe(batch_df.head(10))
            
            # Validate required columns
            required_cols = [
                'Age', 'Gender', 'Region', 'Income_Level', 'Household_Dependents',
                'Amount (NGN)', 'Wallet_Balance (NGN)', 'Days_Since_Last_Transaction',
                'Channel', 'Subsidy_Type', 'Wallet_Activity_Status',
                'Monthly_Energy_Consumption_kWh', 'Avg_Monthly_Wallet_Balance'
            ]
            
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV file contains all required columns as shown in the expected format above.")
                return
            
            # Process batch prediction
            if st.button("🔍 Predict Fraud Risk for Batch", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    # Train models if not already trained
                    if not st.session_state.fraud_models:
                        fraud_results, best_model_name = supervised_fraud_detection(st.session_state.df)
                        st.session_state.fraud_models = fraud_results
                        st.session_state.best_fraud_model = best_model_name
                    
                    # Process each transaction
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in batch_df.iterrows():
                        # Prepare transaction data
                        transaction_data = {
                            'Age': row['Age'],
                            'Gender': row['Gender'],
                            'Region': row['Region'],
                            'Income_Level': row['Income_Level'],
                            'Household_Dependents': row['Household_Dependents'],
                            'Monthly_Energy_Consumption_kWh': row['Monthly_Energy_Consumption_kWh'],
                            'Amount (NGN)': row['Amount (NGN)'],
                            'Wallet_Balance (NGN)': row['Wallet_Balance (NGN)'],
                            'Days_Since_Last_Transaction': row['Days_Since_Last_Transaction'],
                            'Channel': row['Channel'],
                            'Subsidy_Type': row['Subsidy_Type'],
                            'Wallet_Activity_Status': row['Wallet_Activity_Status'],
                            'Avg_Monthly_Wallet_Balance': row['Avg_Monthly_Wallet_Balance'],
                            'Subsidy_Eligibility': row.get('Subsidy_Eligibility', 1)
                        }
                        
                        # Create DataFrame for prediction
                        input_df = pd.DataFrame([transaction_data])
                        
                        # Add engineered features
                        input_df['Amount_per_Dependent'] = input_df['Amount (NGN)'] / max(input_df['Household_Dependents'].iloc[0], 1)
                        input_df['Energy_per_Dependent'] = input_df['Monthly_Energy_Consumption_kWh'] / max(input_df['Household_Dependents'].iloc[0], 1)
                        input_df['Days_Since_Distribution'] = input_df['Days_Since_Last_Transaction']
                        
                        # Make prediction
                        fraud_prob, explanation, recommendations = predict_single_fraud(input_df)
                        
                        # Determine risk level
                        if fraud_prob >= 0.7:
                            risk_level = "HIGH"
                        elif fraud_prob >= 0.4:
                            risk_level = "MEDIUM"
                        else:
                            risk_level = "LOW"
                        
                        # Store results
                        result = {
                            'Transaction_Index': idx + 1,
                            'National_ID': row.get('National_ID', f'TX_{idx+1:05d}'),
                            'Fraud_Probability': fraud_prob,
                            'Risk_Level': risk_level,
                            'Top_Risk_Factors': '; '.join(explanation[:3]) if explanation else 'No specific factors',
                            'Recommendations': '; '.join(recommendations[:2]) if recommendations else 'Standard processing'
                        }
                        
                        # Add original transaction data
                        for col in ['Age', 'Gender', 'Region', 'Amount (NGN)', 'Channel']:
                            if col in row:
                                result[col] = row[col]
                        
                        results.append(result)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(batch_df))
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Display batch results
                    st.markdown("---")
                    st.markdown("## 📊 Batch Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_transactions = len(results_df)
                        st.metric("Total Transactions", total_transactions)
                    
                    with col2:
                        high_risk = len(results_df[results_df['Risk_Level'] == 'HIGH'])
                        st.metric("High Risk", high_risk, delta=f"{high_risk/total_transactions:.1%}")
                    
                    with col3:
                        medium_risk = len(results_df[results_df['Risk_Level'] == 'MEDIUM'])
                        st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/total_transactions:.1%}")
                    
                    with col4:
                        low_risk = len(results_df[results_df['Risk_Level'] == 'LOW'])
                        st.metric("Low Risk", low_risk, delta=f"{low_risk/total_transactions:.1%}")
                    
                    # Risk distribution chart
                    st.markdown("### Risk Distribution")
                    
                    risk_counts = results_df['Risk_Level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Fraud Risk Distribution",
                        color_discrete_map={
                            'HIGH': '#D32F2F',
                            'MEDIUM': '#FFA000',
                            'LOW': '#388E3C'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results table
                    st.markdown("### Detailed Results")
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_filter = st.multiselect(
                            "Filter by Risk Level:",
                            options=['HIGH', 'MEDIUM', 'LOW'],
                            default=['HIGH', 'MEDIUM', 'LOW']
                        )
                    
                    with col2:
                        prob_threshold = st.slider(
                            "Minimum Fraud Probability:",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1
                        )
                    
                    # Apply filters
                    filtered_results = results_df[
                        (results_df['Risk_Level'].isin(risk_filter)) &
                        (results_df['Fraud_Probability'] >= prob_threshold)
                    ]
                    
                    # Display filtered results
                    st.dataframe(
                        filtered_results.style.format({
                            'Fraud_Probability': '{:.1%}'
                        }).background_gradient(
                            subset=['Fraud_Probability'],
                            cmap='Reds'
                        ),
                        use_container_width=True
                    )
                    
                    # Aggregate insights and recommendations
                    st.markdown("---")
                    st.markdown("## 💡 Aggregate Insights & Recommendations")
                    
                    # Overall fraud rate
                    avg_fraud_prob = results_df['Fraud_Probability'].mean()
                    high_risk_pct = (results_df['Risk_Level'] == 'HIGH').mean()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Key Insights")
                        st.markdown(f"• **Average Fraud Probability**: {avg_fraud_prob:.1%}")
                        st.markdown(f"• **High-Risk Transactions**: {high_risk_pct:.1%} of total")
                        
                        # Most common risk factors
                        all_factors = []
                        for factors in results_df['Top_Risk_Factors']:
                            if factors != 'No specific factors':
                                all_factors.extend([f.split(':')[0].strip('*') for f in factors.split(';')])
                        
                        if all_factors:
                            factor_counts = pd.Series(all_factors).value_counts().head(3)
                            st.markdown("• **Most Common Risk Factors**:")
                            for factor, count in factor_counts.items():
                                st.markdown(f"  - {factor}: {count} transactions")
                    
                    with col2:
                        st.markdown("### 🎯 Business Recommendations")
                        
                        if high_risk_pct > 0.1:  # More than 10% high risk
                            st.markdown("• **Enhanced Monitoring**: Implement stricter fraud controls")
                            st.markdown("• **Process Review**: Review eligibility criteria and verification processes")
                        
                        if avg_fraud_prob > 0.3:  # Average fraud probability > 30%
                            st.markdown("• **System Alert**: Consider temporary suspension of high-risk channels")
                            st.markdown("• **Investigation**: Conduct detailed investigation of flagged transactions")
                        
                        st.markdown("• **Regular Monitoring**: Implement automated monitoring for similar patterns")
                        st.markdown("• **Staff Training**: Train staff on identified fraud patterns")
                    
                    # Export results
                    st.markdown("### 📥 Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export full results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Full Results",
                            data=csv_results,
                            file_name=f"batch_fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Export high-risk only
                        high_risk_df = results_df[results_df['Risk_Level'] == 'HIGH']
                        if not high_risk_df.empty:
                            csv_high_risk = high_risk_df.to_csv(index=False)
                            st.download_button(
                                label="🚨 Download High-Risk Only",
                                data=csv_high_risk,
                                file_name=f"high_risk_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No high-risk transactions to export")
        
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")
            st.info("Please check that your CSV file format matches the expected format.")


# Run the app
if __name__ == "__main__":
    main()