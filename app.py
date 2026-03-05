import io
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def show_dataset_info(df: pd.DataFrame) -> None:
    """Display dataset overview information."""
    st.subheader("1) Dataset Upload")
    st.success("Dataset loaded successfully.")
    st.write("First 5 Rows")
    st.dataframe(df.head(), width="stretch")

    rows, cols = df.shape
    c1, c2 = st.columns(2)
    c1.metric("Rows", rows)
    c2.metric("Columns", cols)

    st.write("Column Names and Data Types")
    st.dataframe(
        pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str).values}),
        width="stretch",
    )


def show_data_understanding(df: pd.DataFrame) -> None:
    """Display summary statistics, missing values and dataframe info."""
    st.subheader("2) Data Understanding")

    st.write("Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), width="stretch")

    st.write("Missing Values Per Column")
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, width="stretch")

    st.write("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform simple missing value handling based on user choice."""
    st.subheader("3) Data Cleaning")
    method = st.radio(
        "Choose missing value handling method:",
        ["Drop Missing Rows", "Fill Numeric Mean + Categorical Mode"],
        horizontal=True,
    )

    if st.button("Clean Data"):
        cleaned = df.copy()
        if method == "Drop Missing Rows":
            cleaned = cleaned.dropna()
        else:
            for col in cleaned.columns:
                if cleaned[col].isna().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
                else:
                    mode_series = cleaned[col].mode(dropna=True)
                    fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    cleaned[col] = cleaned[col].fillna(fill_value)

        st.session_state["cleaned_df"] = cleaned
        st.success("Data cleaning completed.")

    active_df = st.session_state.get("cleaned_df", df)
    st.write("Cleaned Dataset Preview")
    st.dataframe(active_df.head(), width="stretch")
    return active_df


def feature_selection(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Allow user to select target and feature columns."""
    st.subheader("4) Feature Selection")
    target_column = st.selectbox("Select Target Variable", options=df.columns)
    feature_options = [col for col in df.columns if col != target_column]
    selected_features = st.multiselect(
        "Select Input Features",
        options=feature_options,
        default=feature_options[: min(5, len(feature_options))],
    )
    return target_column, selected_features


def visualization_section(df: pd.DataFrame, target_column: str, selected_features: list[str]) -> None:
    """Show visualization options using matplotlib and seaborn."""
    st.subheader("5) Data Visualization")
    viz_choice = st.selectbox(
        "Choose Visualization",
        ["Correlation Heatmap", "Scatter Plot", "Pair Plot"],
    )

    if viz_choice == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric columns for a heatmap.")
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif viz_choice == "Scatter Plot":
        numeric_features = [
            col for col in selected_features if pd.api.types.is_numeric_dtype(df[col])
        ]
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            st.warning("Target must be numeric for scatter plot.")
            return
        if not numeric_features:
            st.warning("Select at least one numeric input feature.")
            return

        scatter_feature = st.selectbox("Choose feature for scatter plot", numeric_features)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=scatter_feature, y=target_column, ax=ax)
        ax.set_title(f"{scatter_feature} vs {target_column}")
        st.pyplot(fig)

    else:
        pair_options = [
            col for col in selected_features if pd.api.types.is_numeric_dtype(df[col])
        ]
        if pd.api.types.is_numeric_dtype(df[target_column]):
            pair_options = list(dict.fromkeys(pair_options + [target_column]))

        if len(pair_options) < 2:
            st.warning("Need at least 2 numeric columns for pair plot.")
            return

        chosen = st.multiselect(
            "Select columns for pair plot",
            options=pair_options,
            default=pair_options[: min(4, len(pair_options))],
        )
        if len(chosen) >= 2:
            pair_fig = sns.pairplot(df[chosen]).fig
            st.pyplot(pair_fig)
        else:
            st.info("Select at least 2 columns for pair plot.")


def train_and_evaluate(
    df: pd.DataFrame, target_column: str, selected_features: list[str]
) -> None:
    """Split data, train Linear Regression, predict and evaluate."""
    st.subheader("6) Train/Test Split  →  7) Model Training  →  8) Prediction  →  9) Evaluation")

    if not selected_features:
        st.warning("Please select at least one feature before training.")
        return

    if not pd.api.types.is_numeric_dtype(df[target_column]):
        st.error("Linear Regression requires a numeric target column.")
        return

    model_df = df[selected_features + [target_column]].dropna()
    X = model_df[selected_features]
    y = model_df[target_column]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train Linear Regression Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions_df = pd.DataFrame(
            {
                "Actual": y_test.values,
                "Predicted": y_pred,
            }
        )

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        st.session_state["trained_model"] = model
        st.session_state["predictions_df"] = predictions_df
        st.session_state["metrics"] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        }
        st.session_state["split"] = {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        st.success("Model training completed successfully.")

    if "metrics" in st.session_state:
        split_info = st.session_state["split"]
        st.write(
            f"Train Size: {split_info['train_rows']} rows | Test Size: {split_info['test_rows']} rows"
        )

        st.write("Predicted vs Actual Values")
        st.dataframe(st.session_state["predictions_df"], width="stretch")

        st.write("Model Evaluation Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{st.session_state['metrics']['MAE']:.4f}")
        c2.metric("MSE", f"{st.session_state['metrics']['MSE']:.4f}")
        c3.metric("RMSE", f"{st.session_state['metrics']['RMSE']:.4f}")
        c4.metric("R² Score", f"{st.session_state['metrics']['R2']:.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=st.session_state["predictions_df"],
            x="Actual",
            y="Predicted",
            ax=ax,
            alpha=0.7,
        )
        min_val = st.session_state["predictions_df"]["Actual"].min()
        max_val = st.session_state["predictions_df"]["Actual"].max()
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        model_bytes = pickle.dumps(st.session_state["trained_model"])
        st.download_button(
            "Download Trained Model (.pkl)",
            data=model_bytes,
            file_name="linear_regression_model.pkl",
            mime="application/octet-stream",
        )


def manual_user_prediction(
    df: pd.DataFrame, target_column: str, selected_features: list[str]
) -> None:
    """Take manual user inputs and generate single prediction output."""
    st.subheader("10) User Input → Output Prediction")

    if "trained_model" not in st.session_state:
        st.info("Train the model first to enable manual user prediction output.")
        return

    model = st.session_state["trained_model"]
    st.write("Enter values for selected input features:")

    input_values = {}
    for col in selected_features:
        series = df[col].dropna()
        label = col.replace("_", " ").title()

        if pd.api.types.is_numeric_dtype(df[col]):
            default_val = float(series.median()) if not series.empty else 0.0
            min_val = float(series.min()) if not series.empty else 0.0
            max_val = float(series.max()) if not series.empty else 100.0
            input_values[col] = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                key=f"manual_{col}",
            )
        else:
            options = sorted(series.astype(str).unique().tolist()) if not series.empty else ["Unknown"]
            input_values[col] = st.selectbox(label, options=options, key=f"manual_{col}")

    input_df = pd.DataFrame([input_values])
    st.dataframe(input_df, width="stretch")

    if st.button("Predict Output", type="primary"):
        prediction = float(model.predict(input_df)[0])
        st.success(f"Predicted {target_column}: **{prediction:.4f}**")


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="Linear Regression ML Pipeline", page_icon="📈", layout="wide")
    st.title("📈 Linear Regression - Complete ML Pipeline")
    st.caption(
        "Flow: Upload Dataset → Understand Data → Clean Data → Feature Selection → Visualization → "
        "Train/Test Split → Model Training → Prediction → Evaluation"
    )

    st.sidebar.title("Pipeline Navigation")
    st.sidebar.markdown(
        """
1. Upload Dataset
2. Data Understanding
3. Data Cleaning
4. Feature Selection
5. Visualization
6. Train/Test Split
7. Model Training
8. Prediction
9. Evaluation
"""
    )

    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to start the ML pipeline.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    show_dataset_info(df)
    show_data_understanding(df)
    cleaned_df = clean_data(df)
    target_column, selected_features = feature_selection(cleaned_df)
    visualization_section(cleaned_df, target_column, selected_features)
    train_and_evaluate(cleaned_df, target_column, selected_features)
    manual_user_prediction(cleaned_df, target_column, selected_features)


if __name__ == "__main__":
    main()
