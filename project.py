import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ==================================================
# STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# TRAIN MACHINE LEARNING MODEL
# ==================================================

@st.cache_resource
def train_model():

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    df = pd.read_csv("student_data.csv")

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # --------------------------------------------------
    # Target Column
    # --------------------------------------------------

    target_column = "final_score"

    # Check target column
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    # --------------------------------------------------
    # Separate Features and Target
    # --------------------------------------------------

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --------------------------------------------------
    # Identify Numerical and Categorical Columns
    # --------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "bool", "category"]
    ).columns.tolist()

    # --------------------------------------------------
    # Numerical Preprocessing
    # --------------------------------------------------

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    # --------------------------------------------------
    # Categorical Preprocessing
    # --------------------------------------------------

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------
    # Combine Preprocessing
    # --------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------
    # Random Forest Model
    # --------------------------------------------------

    random_forest = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------
    # Complete Machine Learning Pipeline
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                random_forest
            )
        ]
    )

    # --------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Test Model
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # Model Evaluation
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return model, df, mae, r2


# ==================================================
# LOAD MODEL
# ==================================================

try:

    model, df, mae, r2 = train_model()

except FileNotFoundError:

    st.error(
        "❌ Dataset not found!\n\n"
        "Make sure 'student_data.csv' is in the "
        "same folder as project.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading the dataset: {e}"
    )

    st.stop()


# ==================================================
# APPLICATION TITLE
# ==================================================

st.title("🎓 Student Performance Prediction")

st.write(
    "Enter student information below to predict "
    "the student's final examination score."
)

st.divider()


# ==================================================
# DATASET INFORMATION
# ==================================================

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Students",
        len(df)
    )

with col2:

    st.metric(
        "Number of Features",
        len(df.columns) - 1
    )

with col3:

    st.metric(
        "Model R² Score",
        f"{r2:.2f}"
    )


st.divider()


# ==================================================
# USER INPUT
# ==================================================

st.subheader("👨‍🎓 Enter Student Details")

# All columns except target
input_columns = [
    column
    for column in df.columns
    if column != "final_score"
]

input_data = {}

col1, col2 = st.columns(2)


# ==================================================
# CREATE INPUT FIELDS
# ==================================================

for index, column in enumerate(input_columns):

    with col1 if index % 2 == 0 else col2:

        # --------------------------------------------------
        # CATEGORICAL COLUMN
        # --------------------------------------------------

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            options = (
                df[column]
                .dropna()
                .unique()
                .tolist()
            )

            input_data[column] = st.selectbox(
                column.replace(
                    "_",
                    " "
                ).title(),
                options
            )

        # --------------------------------------------------
        # NUMERICAL COLUMN
        # --------------------------------------------------

        else:

            minimum = float(
                df[column].min()
            )

            maximum = float(
                df[column].max()
            )

            default = float(
                df[column].median()
            )

            input_data[column] = st.number_input(
                column.replace(
                    "_",
                    " "
                ).title(),

                min_value=minimum,

                max_value=maximum,

                value=default
            )


# ==================================================
# PREDICTION
# ==================================================

st.divider()

if st.button(
    "🚀 Predict Final Score",
    type="primary"
):

    # --------------------------------------------------
    # Create DataFrame from User Input
    # --------------------------------------------------

    new_student = pd.DataFrame(
        [input_data]
    )

    # Make sure columns are in same order
    new_student = new_student[
        input_columns
    ]

    try:

        # --------------------------------------------------
        # Make Prediction
        # --------------------------------------------------

        prediction = model.predict(
            new_student
        )[0]

        # Keep prediction between 0 and 100
        prediction = max(
            0,
            min(
                100,
                prediction
            )
        )

        # --------------------------------------------------
        # Display Prediction
        # --------------------------------------------------

        st.success(
            f"🎯 Predicted Final Score: "
            f"**{prediction:.2f} / 100**"
        )

        # --------------------------------------------------
        # Performance Category
        # --------------------------------------------------

        if prediction >= 90:

            performance = "Excellent 🏆"

        elif prediction >= 75:

            performance = "Very Good ⭐"

        elif prediction >= 60:

            performance = "Good 👍"

        elif prediction >= 40:

            performance = "Needs Improvement ⚠️"

        else:

            performance = "At Risk ❗"

        st.info(
            f"📚 Performance Level: "
            f"**{performance}**"
        )

        # --------------------------------------------------
        # Display Entered Details
        # --------------------------------------------------

        st.subheader(
            "📋 Student Details"
        )

        st.dataframe(
            new_student,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

st.divider()

st.subheader("🤖 Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Mean Absolute Error",
        f"{mae:.2f}"
    )

with col2:

    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )


# ==================================================
# EXPLANATION
# ==================================================

st.divider()

st.subheader("ℹ️ About the Model")

st.write(
    """
    This application uses a Random Forest Regression model
    to predict a student's final examination score.

    The model uses student information such as:

    • Age
    • Gender
    • Study Hours
    • Attendance
    • Previous Score
    • Sleep Hours
    • Assignments
    • Extracurricular Activities
    • Parental Support
    • Internet Access

    Categorical information such as Gender and Parental
    Support is automatically converted into numerical form
    using One-Hot Encoding.
    """
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    f"Model: Random Forest Regressor | "
    f"Dataset: {len(df):,} student records"
)