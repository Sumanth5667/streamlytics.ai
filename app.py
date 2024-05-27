import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Step 1: Upload Dataset
st.title("ML Model Trainer")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Step 2: Ask for the type of problem
    problem_type = st.selectbox("What type of problem are you solving?",
                                ["", "Multi-class Classification", "Binary Classification", "Regression", "Multi-output Regression", "Clustering"],
                                index=0)

    # Step 3: Handle missing values
    missing_columns = df.columns[df.isnull().any()]
    if len(missing_columns) > 0:
        st.write("Columns with missing values:")
        st.write(missing_columns)
        impute_methods = ["Mean", "Median", "Most Frequent", "Constant"]
        impute_method_dict = {}
        for column in missing_columns:
            st.write(f"Imputation method for {column}:")
            impute_method = st.selectbox("", impute_methods, index=0, key=f"impute_method_{column}")
            if impute_method == "Constant":
                fill_value = st.text_input("Enter constant value to fill missing values:", key=f"fill_value_{column}")
                impute_method_dict[column] = (impute_method, fill_value)
            else:
                impute_method_dict[column] = (impute_method, None)

        for column, (impute_method, fill_value) in impute_method_dict.items():
            if impute_method == "Mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif impute_method == "Median":
                df[column].fillna(df[column].median(), inplace=True)
            elif impute_method == "Most Frequent":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif impute_method == "Constant":
                df[column].fillna(fill_value, inplace=True)

    # Step 4: Handle categorical columns
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) > 0:
        st.write("Categorical columns:")
        st.write(categorical_columns)
        encoding_methods = ["OneHot", "Ordinal"]
        encoding_method_dict = {}
        for i, column in enumerate(categorical_columns):
            st.write(f"Encoding method for {column}:")
            encoding_method = st.selectbox("", encoding_methods, index=0, key=f"encoding_method_{column}")
            if encoding_method == "Ordinal":
                unique_categories = df[column].unique()
                category_order = st.multiselect("Category order:", unique_categories, unique_categories,
                                                key=f"category_order_{column}")
                encoding_method_dict[column] = (encoding_method, category_order)
            else:
                encoding_method_dict[column] = (encoding_method, None)

        for column, (encoding_method, category_order) in encoding_method_dict.items():
            if encoding_method == "OneHot":
                df = pd.get_dummies(df, columns=[column])
            elif encoding_method == "Ordinal":
                ordinal_encoder = OrdinalEncoder(categories=[category_order])
                df[column] = ordinal_encoder.fit_transform(df[[column]])
                df[column] = df[column].astype(int)

    st.write("Data after preprocessing:")
    st.write(df.head())

    # Step 5: Choose visualization
    visualization = st.selectbox("Choose a visualization type:",
                                 ["", "Correlation Heatmap", "Pairplot", "Distribution Plot", "Scatter Plot",
                                  "Box Plot", "Histogram", "Bar Plot", "Line Plot"],
                                 index=0)

    if visualization:
        if visualization == "Correlation Heatmap":
            st.write("Correlation Heatmap:")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), ax=ax, annot=True, cmap="coolwarm")
            st.pyplot(fig)
        elif visualization == "Pairplot":
            st.write("Pairplot:")
            st.pyplot(sns.pairplot(df))
        elif visualization == "Distribution Plot":
            selected_column = st.selectbox("Select a column to plot distribution:", df.columns)
            st.write(f"Distribution plot for {selected_column}:")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], kde=True, ax=ax)
            st.pyplot(fig)
        elif visualization == "Scatter Plot":
            x_column = st.selectbox("Select X column:", df.columns)
            y_column = st.selectbox("Select Y column:", df.columns)
            fig = px.scatter(df, x=x_column, y=y_column)
            st.plotly_chart(fig)
        elif visualization == "Box Plot":
            box_column = st.selectbox("Select column for box plot:", df.columns)
            fig = px.box(df, y=box_column)
            st.plotly_chart(fig)
        elif visualization == "Histogram":
            selected_column = st.selectbox("Select a column to plot histogram:", df.columns)
            st.write(f"Histogram for {selected_column}:")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], ax=ax)
            st.pyplot(fig)
        elif visualization == "Bar Plot":
            selected_column = st.selectbox("Select a column to plot bar plot:", df.columns)
            st.write(f"Bar plot for {selected_column}:")
            fig, ax = plt.subplots()
            sns.barplot(x=df[selected_column].value_counts().index, y=df[selected_column].value_counts().values, ax=ax)
            st.pyplot(fig)
        elif visualization == "Line Plot":
            selected_column = st.selectbox("Select a column to plot line plot:", df.columns)
            st.write(f"Line plot for {selected_column}:")
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=df.index, y=selected_column, ax=ax)
            st.pyplot(fig)

    # Step 6: Select the target column(s)
    if problem_type in ["Multi-class Classification", "Binary Classification", "Regression", "Multi-output Regression"]:
        target_columns = st.multiselect("Select the target column(s):", df.columns)

    # Step 7: Select the ML model
    model_type = ""
    if problem_type == "Multi-class Classification":
        model_type = st.selectbox("Select a model", ["", "Logistic Regression", "Random Forest Classifier"], index=0)
    elif problem_type == "Binary Classification":
        model_type = st.selectbox("Select a model", ["", "Logistic Regression", "Random Forest Classifier"], index=0)
    elif problem_type == "Regression":
        model_type = st.selectbox("Select a model", ["", "Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Random Forest Regressor"], index=0)
    elif problem_type == "Multi-output Regression":
        model_type = st.selectbox("Select a model", ["", "Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "Random Forest Regressor"], index=0)
    elif problem_type == "Clustering":
        model_type = st.selectbox("Select a model", ["", "KMeans Clustering"], index=0)

    # Step 8: Set hyperparameters
    hyperparams = {}
    if model_type == "Logistic Regression":
        hyperparams['C'] = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
        hyperparams['max_iter'] = st.number_input("Maximum iterations", 100, 1000, 100)
        hyperparams['solver'] = st.selectbox("Solver", ["lbfgs", "liblinear", "sag", "saga"], index=0)
        hyperparams['penalty'] = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], index=0)
        hyperparams['class_weight'] = st.selectbox("Class weight", ["balanced", None], index=0)
    elif model_type in ["Random Forest Classifier", "Random Forest Regressor"]:
        hyperparams['n_estimators'] = st.slider("Number of trees", 10, 200, 100)
        hyperparams['max_depth'] = st.slider("Maximum depth of tree", 1, 20, 10)
        hyperparams['min_samples_split'] = st.slider("Minimum samples split", 2, 10, 2)
        hyperparams['min_samples_leaf'] = st.slider("Minimum samples leaf", 1, 10, 1)
        hyperparams['bootstrap'] = st.selectbox("Bootstrap", [True, False], index=0)
    elif model_type == "Linear Regression":
        hyperparams = {}  # No hyperparameters for linear regression
    elif model_type == "Ridge Regression":
        hyperparams['alpha'] = st.slider("Regularization parameter (alpha)", 0.01, 10.0, 1.0)
        hyperparams['fit_intercept'] = st.selectbox("Fit intercept", [True, False], index=0)
    elif model_type == "Lasso Regression":
        hyperparams['alpha'] = st.slider("Regularization parameter (alpha)", 0.01, 10.0, 1.0)
        hyperparams['fit_intercept'] = st.selectbox("Fit intercept", [True, False], index=0)
    elif model_type == "ElasticNet Regression":
        hyperparams['alpha'] = st.slider("Regularization parameter (alpha)", 0.01, 10.0, 1.0)
        hyperparams['l1_ratio'] = st.slider("L1 ratio", 0.0, 1.0, 0.5)
        hyperparams['fit_intercept'] = st.selectbox("Fit intercept", [True, False], index=0)
    elif model_type == "KMeans Clustering":
        hyperparams['n_clusters'] = st.slider("Number of clusters", 2, 10, 3)
        hyperparams['init'] = st.selectbox("Initialization method", ["k-means++", "random"], index=0)
        hyperparams['n_init'] = st.slider("Number of initializations", 10, 50, 10)
        hyperparams['max_iter'] = st.number_input("Maximum iterations", 100, 1000, 300)
        hyperparams['tol'] = st.slider("Tolerance", 1e-4, 1e-1, 1e-4)

    # Step 9: Train the model
    if st.button("Train the model"):
        if problem_type in ["Multi-class Classification", "Binary Classification", "Regression", "Multi-output Regression"]:
            X = df.drop(columns=target_columns)
            y = df[target_columns]
            if problem_type in ["Multi-class Classification", "Binary Classification"]:
                y = y.iloc[:, 0]  # Only one target column for classification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            if model_type == "Logistic Regression":
                model = LogisticRegression(C=hyperparams['C'], max_iter=hyperparams['max_iter'],
                                           solver=hyperparams['solver'], penalty=hyperparams['penalty'],
                                           class_weight=hyperparams['class_weight'])
            elif model_type == "Random Forest Classifier":
                model = RandomForestClassifier(n_estimators=hyperparams['n_estimators'],
                                               max_depth=hyperparams['max_depth'],
                                               min_samples_split=hyperparams['min_samples_split'],
                                               min_samples_leaf=hyperparams['min_samples_leaf'],
                                               bootstrap=hyperparams['bootstrap'])
            elif model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Ridge Regression":
                model = Ridge(alpha=hyperparams['alpha'], fit_intercept=hyperparams['fit_intercept'])
            elif model_type == "Lasso Regression":
                model = Lasso(alpha=hyperparams['alpha'], fit_intercept=hyperparams['fit_intercept'])
            elif model_type == "ElasticNet Regression":
                model = ElasticNet(alpha=hyperparams['alpha'], l1_ratio=hyperparams['l1_ratio'], fit_intercept=hyperparams['fit_intercept'])
            elif model_type == "Random Forest Regressor":
                model = RandomForestRegressor(n_estimators=hyperparams['n_estimators'],
                                              max_depth=hyperparams['max_depth'],
                                              min_samples_split=hyperparams['min_samples_split'],
                                              min_samples_leaf=hyperparams['min_samples_leaf'],
                                              bootstrap=hyperparams['bootstrap'])

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Step 10: Display results
            if problem_type == "Multi-class Classification":
                st.write("Training Results:")
                st.text(classification_report(y_train, y_pred_train))
                st.write("Test Results:")
                st.text(classification_report(y_test, y_pred_test))

                # Confusion Matrix
                st.write("Confusion Matrix (Test Data):")
                cm = confusion_matrix(y_test, y_pred_test)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            elif problem_type == "Binary Classification":
                st.write("Training Results:")
                st.text(classification_report(y_train, y_pred_train))
                st.write("Test Results:")
                st.text(classification_report(y_test, y_pred_test))

                # Confusion Matrix
                st.write("Confusion Matrix (Test Data):")
                cm = confusion_matrix(y_test, y_pred_test)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # ROC AUC Curve (only for linear regression)
                if model_type == "Logistic Regression":
                    st.write("ROC AUC Curve:")
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:0.2f})'))
                    st.plotly_chart(fig)

            elif problem_type in ["Regression", "Multi-output Regression"]:
                st.write("Training Results:")
                st.write(f"Train Mean Squared Error: {mean_squared_error(y_train, y_pred_train):.4f}")
                st.write(f"Train R^2 Score: {r2_score(y_train, y_pred_train):.4f}")
                st.write("Test Results:")
                st.write(f"Test Mean Squared Error: {mean_squared_error(y_test, y_pred_test):.4f}")
                st.write(f"Test R^2 Score: {r2_score(y_test, y_pred_test):.4f}")

            elif problem_type == "Clustering" and model_type == "KMeans Clustering":
                X = df.drop(columns=target_columns)
                model = KMeans(n_clusters=hyperparams['n_clusters'], init=hyperparams['init'],
                               n_init=hyperparams['n_init'], max_iter=hyperparams['max_iter'], tol=hyperparams['tol'])
                model.fit(X)
                df['Cluster'] = model.labels_
                st.write("Clustering Results:")
                st.write(df.head())

                # Cluster Plot
                st.write("Cluster Plot:")
                fig = px.scatter(df, x=X.columns[0], y=X.columns[1], color='Cluster')
                st.plotly_chart(fig)
