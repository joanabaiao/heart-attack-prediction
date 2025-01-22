import pandas as pd
import toml
import streamlit as st
from streamlit_option_menu import option_menu
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col, lit, as_double
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time


@st.cache_resource
def create_session_object():

    config = toml.load("config.toml")
    connection_parameters = config["snowflake_connection"]

    session = Session.builder.configs(connection_parameters).create()
    session = Session.builder.configs(connection_parameters).create()
    print(
        session.sql(
            "SELECT current_warehouse(), current_database(), current_schema()"
        ).collect()
    )
    return session


def display_evaluation_metrics(metrics):
    """
    Displays evaluation metrics including the confusion matrix and key performance metrics.
    """

    if isinstance(metrics, pd.DataFrame):
        metrics = {
            "TP": metrics["TP"].iloc[0],
            "TN": metrics["TN"].iloc[0],
            "FP": metrics["FP"].iloc[0],
            "FN": metrics["FN"].iloc[0],
            "accuracy": metrics["ACCURACY"].iloc[0],
            "precision": metrics["PRECISION"].iloc[0],
            "recall": metrics["RECALL"].iloc[0],
            "f1_score": metrics["F1_SCORE"].iloc[0],
            "auc_roc": metrics["AUC_ROC"].iloc[0],
        }

    if "f1" in metrics:
        metrics["f1_score"] = metrics.pop("f1")

    TP = metrics.get("TP", 0)
    TN = metrics.get("TN", 0)
    FP = metrics.get("FP", 0)
    FN = metrics.get("FN", 0)

    # Generate confusion matrix
    confusion_matrix = [[TP, FP], [FN, TN]]
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Positive", "Predicted Negative"],
            y=["Actual Positive", "Actual Negative"],
            text=[[f"{TP}", f"{FP}"], [f"{FN}", f"{TN}"]],
            texttemplate="%{text}",
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
    )
    st.plotly_chart(fig)

    # Display metrics using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metrics.get('accuracy') * 100:.2f}%",
        )
    with col2:
        st.metric(
            label="Precision",
            value=f"{metrics.get('precision') * 100:.2f}%",
        )
    with col3:
        st.metric(label="Recall", value=f"{metrics.get('recall') * 100:.2f}%")

    with col1:
        st.metric(
            label="F1-Score",
            value=f"{metrics.get('f1_score') * 100:.2f}%",
        )
    with col2:
        st.metric(
            label="AUC-ROC",
            value=f"{metrics.get('auc_roc') * 100:.2f}%",
        )


########################################################################
# STREAMLIT APP
########################################################################
st.set_page_config(page_title="Snowpark project")
st.title("Heart attack prediction w/ Snowpark")

session = create_session_object()
model_info_tables = [
    "MODEL_CATALOG",
    "MODEL_TRAINING_INFO",
    "INFERENCE_RESULTS",
    "LAST_INFERENCE_RESULT",
]
other_tables = ["PREDICTIONS_RESULT"]

with st.sidebar:
    option_selected = option_menu(
        "Menu",
        [
            "Home",
            # "Load data",
            "Data Analysis",
            "Train model",
            "Model catalog",
            "Inference",
            "Inference runs",
        ],
        icons=[
            "info-circle",
            # "upload",
            "graph-up",
            "gear",
            "list-task",
            "play-circle",
            "lightbulb",
        ],
        menu_icon="-",
        default_index=0,
    )


######################
# OPTION 1: HOME
######################
if option_selected == "Home":
    st.header("Home", divider="red")

    st.write(
        "This app helps you analyze patient data, train machine learning models, and make predictions - all powered by **Snowflake's Snowpark**."
    )
    st.subheader("What you can do:")
    st.write(
        """
            - **Data Analysis**: Explore patient datasets and gain insights.
            - **Train Models**: Build and train machine learning models.
            - **Model Catalog**: Review details of trained models and their performance.
            - **Inference**: Use trained models to make predictions on new data.
            - **Inference runs**: Track past predictions and performance metrics."""
    )

    st.write("🚀 Use the menu on the left to get started!")


######################
# OPTION 2: DATA ANALYSIS
######################
if option_selected == "Data Analysis":
    st.header("Data Analysis", divider="red")

    # DISPLAY AVAILABLE TABLES
    with st.container():

        # Fetch table information from the session
        df_tables = (
            session.table("information_schema.tables")
            .filter(col("table_schema") == "PUBLIC")
            .filter(~col("table_name").isin(model_info_tables + other_tables))
            .select(col("table_name"), col("row_count"), col("created"))
        ).to_pandas()

        st.subheader("Tables available:")
        st.dataframe(df_tables, use_container_width=True)

    # DISPLAY SELECTED TABLE DETAILS
    with st.container():
        # Dropdown to select a table
        list_tables_names = df_tables["TABLE_NAME"].values.tolist()
        selected_table = st.selectbox("Select a table to analyse:", list_tables_names)

        st.markdown("----")

        if selected_table:
            st.subheader(f"Details for table {selected_table}:")

            # Fetch table data and statistics
            complete_table_name = f"PUBLIC.{selected_table}"
            df_table = session.table(complete_table_name)
            pd_table = df_table.to_pandas()  # Preview of the table
            pd_describe = df_table.describe().to_pandas()  # Descriptive statistics
            target_column = "TARGET"

            # Split layout for metrics
            col1, col2 = st.columns(2)

            with col1:
                positive_count = df_table.filter(col("target") == 1).count()
                st.metric(label="Positive", value=f"{positive_count:,}")

            with col2:
                negative_count = df_table.filter(col("target") == 0).count()
                st.metric(label="Negative", value=f"{negative_count:,}")

            # Display table preview
            st.write(f"**Preview table data**")
            st.dataframe(pd_table.head(5), use_container_width=True)

            # Display descriptive statistics
            st.write(f"**Data descriptive statistics**")
            st.dataframe(pd_describe, use_container_width=True)

            st.markdown("----")

            st.subheader("Distribution of data columns:")
            column_options = pd_table.columns.tolist()
            selected_column = st.selectbox(
                "Select a column to analyse:", column_options
            )
            if selected_column:
                st.write(f"**Distribution of {selected_column} values**")

                # Check if the selected column is categorical (for pie chart visualization)
                if (
                    pd_table[selected_column].dtype == "object"
                    or pd_table[selected_column].nunique() < 5
                ):

                    column_counts = (
                        pd_table[selected_column].value_counts().reset_index()
                    )
                    column_counts.columns = [selected_column, "Count"]

                    fig = px.pie(
                        column_counts,
                        names=selected_column,
                        values="Count",
                    )
                    st.plotly_chart(fig)

                    st.bar_chart(
                        pd_table[selected_column].value_counts(), horizontal=True
                    )
                else:
                    # For numerical data, display histogram
                    st.bar_chart(pd_table[selected_column].value_counts())


######################
# OPTION 3: TRAIN MODEL
######################
if option_selected == "Train model":
    st.header("Train model", divider="red")

    df_tables = (
        session.table("information_schema.tables")
        .filter(col("table_schema") == "PUBLIC")
        .filter(~col("table_name").isin(model_info_tables + other_tables))
        .select(col("table_name"))
    ).to_pandas()

    df_models = session.table("MODEL_CATALOG").select(col("MODEL_NAME")).to_pandas()

    list_tables_names = df_tables["TABLE_NAME"].values.tolist()

    training_table = st.selectbox("Choose a dataset:", options=list_tables_names)
    model_name = st.selectbox("Select a model:", options=df_models)
    optimization = st.checkbox("Apply optimization with Optuna?")

    if st.button("Train", type="primary"):

        st.markdown("----")

        with st.spinner("Training the model... please wait."):
            time.sleep(1)  # Simulate loading time
            try:
                result = session.call(
                    "TRAIN_AND_DEPLOY_MODEL", model_name, optimization, training_table
                )
                result_dict = eval(result)
                st.success("🎉 Model training complete!")
                st.write(
                    f"The model has been successfully trained and saved in Snowpark!"
                )
                st.write(f"**Model ID:** {result_dict.get('model_id')}")
            except Exception as e:
                st.error("❌ An error occurred while calling Snowpark.")
                st.write(e)

######################
# OPTION 4: MODEL CATALOG
######################
if option_selected == "Model catalog":  # Model Training Details
    st.header("Model training details", divider="red")
    with st.container():

        df_models_training = session.table("PUBLIC.MODEL_TRAINING_INFO").to_pandas()
        df_models_training.sort_values("TRAINING_DATE", inplace=True)

        st.subheader("Models available:")
        st.dataframe(df_models_training, use_container_width=True)

        # Dropdown to select a model
        df_models_training.sort_values("MODEL_ID", inplace=True)
        list_models_train = df_models_training["MODEL_ID"].values.tolist()
        selected_model = st.selectbox(
            "Select a model to see more details:", list_models_train
        )

        if selected_model and st.button("See more details", type="primary"):
            st.markdown("----")

            st.subheader(f"Details of model {selected_model}:")

            df_model = df_models_training[
                df_models_training["MODEL_ID"] == selected_model
            ]
            training_date = pd.to_datetime(df_model["TRAINING_DATE"].iloc[0])

            st.write(
                f"**Training data source:** {df_model['TRAINING_TABLE'].values[0]}"
            )
            st.write(
                f"**Model trained on:** {training_date.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            display_evaluation_metrics(df_model)

######################
# OPTION 5: INFERENCE
######################
if option_selected == "Inference":
    st.header("Inference", divider="red")

    df_models_training = session.table("MODEL_TRAINING_INFO").to_pandas()
    df_models_training.sort_values("MODEL_ID", inplace=True)
    list_models_train = df_models_training["MODEL_ID"].values.tolist()
    selected_model = st.selectbox("Select a model for inference:", list_models_train)

    df_tables = (
        session.table("information_schema.tables")
        .filter(col("table_schema") == "PUBLIC")
        .filter(~col("table_name").isin(model_info_tables + other_tables))
        .select(col("table_name"), col("row_count"), col("created"))
    ).to_pandas()

    list_tables_names = df_tables["TABLE_NAME"].values.tolist()
    selected_table = st.selectbox("Select a table for inference:", list_tables_names)

    # default_table_name = "RESULT"
    # results_table_name = st.text_input(
    #    "Enter the name to save the data with predictions (starts with 'PREDICTIONS_'):",
    #    value=default_table_name,
    #    placeholder="e.g., MODEL1_TABLE2",
    # )
    # results_table_name = f"PREDICTIONS_{results_table_name}"
    results_table_name = "PREDICTIONS_RESULT"

    if st.button("Run inference", type="primary"):
        st.markdown("----")
        with st.spinner("Predicting... please wait."):
            time.sleep(1)

            result_metrics = session.call(
                "RUN_INFERENCE", selected_table, selected_model
            )

            st.success("🎉 Inference run complete!")
            st.write(
                "The prediction process was successfully completed and results have been saved in Snowpark!"
            )

            df_predictions = session.table(results_table_name).to_pandas()
            st.subheader("Data with predicted values")
            st.dataframe(df_predictions, use_container_width=True)

            st.subheader("Evaluation metrics")
            result_metrics = eval(result_metrics)
            display_evaluation_metrics(result_metrics)


######################
# OPTION 6: INFERENCE RUNS
######################
if option_selected == "Inference runs":
    st.header("Inference runs", divider="red")

    with st.container():

        df_inference_results = session.table("INFERENCE_RESULTS").to_pandas()
        df_inference_results.sort_values("INFERENCE_DATE", inplace=True)

        # st.subheader("Models available:")
        st.dataframe(df_inference_results, use_container_width=True)
