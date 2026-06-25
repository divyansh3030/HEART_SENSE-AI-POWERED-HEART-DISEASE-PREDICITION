from datetime import datetime
from functools import wraps
import os
from urllib.parse import urlparse

from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import joblib
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - depends on local environment
    psycopg2 = None
    RealDictCursor = None

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "heartsense-dev-key")

# ============================
# Paths and database config
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

DNN_MODEL_PATH = os.path.join(MODEL_DIR, "heartsense_dnn.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "dnn_scaler.pkl")
PIPELINE_PATH = os.path.join(MODEL_DIR, "heartsense_pipeline.pkl")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# In this dataset, target=1 behaves like lower/no disease risk,
# while target=0 behaves like disease presence.
TARGET_ONE_IS_NO_DISEASE = True

MODEL_PROBE_ROW = {
    "age": 45.0,
    "sex": 1.0,
    "cp": 2.0,
    "trestbps": 120.0,
    "chol": 220.0,
    "fbs": 0.0,
    "restecg": 0.0,
    "thalach": 150.0,
    "exang": 0.0,
    "oldpeak": 1.0,
    "slope": 1.0,
    "ca": 0.0,
    "thal": 2.0,
}

# ============================
# Load saved model (ONCE)
# ============================
model = None
scaler = None
model_type = None


# ============================
# Feature order (CRITICAL)
# ============================
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

TRAINING_BOUNDS = {
    "trestbps": (94, 200),
    "chol": (126, 564),
    "thalach": (71, 202),
    "oldpeak": (0.0, 6.2),
}

AGE_RISK_POINTS = [
    (29, 0.26),
    (40, 0.34),
    (50, 0.51),
    (60, 0.63),
    (77, 0.70),
]

SELECT_OPTIONS = {
    "sex": [("1", "Male"), ("0", "Female")],
    "cp": [
        ("0", "Typical Angina"),
        ("1", "Atypical Angina"),
        ("2", "Non-anginal Pain"),
        ("3", "Asymptomatic"),
    ],
    "fbs": [("0", "No"), ("1", "Yes")],
    "restecg": [
        ("0", "Normal"),
        ("1", "ST-T Wave Abnormality"),
        ("2", "Left Ventricular Hypertrophy"),
    ],
    "exang": [("0", "No"), ("1", "Yes")],
    "slope": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")],
    "ca": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")],
    "thal": [
        ("0", "Normal"),
        ("1", "Fixed Defect"),
        ("2", "Reversible Defect"),
        ("3", "Not Described"),
    ],
}

RISK_LEVEL_OPTIONS = ["Low", "Moderate", "High", "Very High"]

FORM_FIELDS = [
    {"name": "age", "label": "Age", "type": "number", "step": "1", "min": "1", "max": "120", "placeholder": "Enter age"},
    {"name": "sex", "label": "Sex", "type": "select"},
    {"name": "cp", "label": "Chest Pain (cp)", "type": "select"},
    {"name": "trestbps", "label": "Resting BP (trestbps)", "type": "number", "step": "1", "min": "94", "max": "200", "placeholder": "Enter value"},
    {"name": "chol", "label": "Cholesterol (chol)", "type": "number", "step": "1", "min": "126", "max": "564", "placeholder": "Enter value"},
    {"name": "fbs", "label": "Fasting BS (fbs)", "type": "select"},
    {"name": "restecg", "label": "Resting ECG (restecg)", "type": "select"},
    {"name": "thalach", "label": "Max Heart Rate (thalach)", "type": "number", "step": "1", "min": "71", "max": "202", "placeholder": "Enter value"},
    {"name": "exang", "label": "Exercise Angina (exang)", "type": "select"},
    {"name": "oldpeak", "label": "Oldpeak", "type": "number", "step": "0.1", "min": "0", "max": "6.2", "placeholder": "Enter value"},
    {"name": "slope", "label": "Slope", "type": "select"},
    {"name": "ca", "label": "CA (Major Vessels)", "type": "select"},
    {"name": "thal", "label": "Thalassemia (thal)", "type": "select"},
]

FIELD_LABELS = {
    "age": "Age",
    "trestbps": "Resting BP",
    "chol": "Cholesterol",
    "thalach": "Max Heart Rate",
    "oldpeak": "Oldpeak",
}


def probe_loaded_model(candidate_model, candidate_type, candidate_scaler=None):
    probe_df = pd.DataFrame([MODEL_PROBE_ROW], columns=FEATURE_ORDER)

    if candidate_type == "pipeline":
        candidate_model.predict_proba(probe_df)
        return

    if candidate_type == "dnn":
        scaled_probe = candidate_scaler.transform(probe_df.to_numpy(dtype=float))
        candidate_model.predict(scaled_probe, verbose=0)
        return

    raise RuntimeError(f"Unsupported model type: {candidate_type}")


def get_model_unavailable_message():
    return (
        "Prediction model is not available right now. Please verify the saved model files "
        "and installed packages, then restart the app."
    )


def load_prediction_model():
    global model, scaler, model_type

    pipeline_error = None

    if os.path.exists(PIPELINE_PATH):
        try:
            candidate_model = joblib.load(PIPELINE_PATH)
            probe_loaded_model(candidate_model, "pipeline")
            model = candidate_model
            scaler = None
            model_type = "pipeline"
            print("Pipeline model loaded")
            return
        except Exception as exc:  # pragma: no cover - depends on local model env
            pipeline_error = exc
            print("Pipeline model unavailable, trying DNN fallback:", exc)

    try:
        from tensorflow.keras.models import load_model

        candidate_model = load_model(DNN_MODEL_PATH)
        candidate_scaler = joblib.load(SCALER_PATH)
        probe_loaded_model(candidate_model, "dnn", candidate_scaler)
        model = candidate_model
        scaler = candidate_scaler
        model_type = "dnn"
        print("DNN model and scaler loaded")
    except Exception as exc:  # pragma: no cover - startup depends on local files
        print("Failed to load model:", exc)
        if pipeline_error is not None:
            print("Original pipeline error:", pipeline_error)


load_prediction_model()


def build_postgres_config():
    if DATABASE_URL:
        parsed = urlparse(DATABASE_URL)
        return {
            "dbname": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "sslmode": os.getenv("PGSSLMODE", "prefer"),
        }

    return {
        "dbname": os.getenv("PGDATABASE", "heartsense"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", ""),
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "sslmode": os.getenv("PGSSLMODE", "prefer"),
    }


def get_db_connection():
    if psycopg2 is None:
        raise RuntimeError(
            "PostgreSQL driver not installed. Run: pip install psycopg2-binary"
        )

    return psycopg2.connect(**build_postgres_config())


def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    connection = get_db_connection()
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute(query, params or ())
        if commit:
            connection.commit()
            inserted_row = cursor.fetchone()
            return inserted_row["id"] if inserted_row else None
        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        return None
    finally:
        cursor.close()
        connection.close()


def get_current_user():
    if hasattr(g, "current_user"):
        return g.current_user

    user_id = session.get("user_id")
    if not user_id:
        g.current_user = None
        return None

    try:
        g.current_user = execute_query(
            """
            SELECT id, fullname, email, created_at
            FROM users
            WHERE id = %s
            """,
            (user_id,),
            fetchone=True,
        )
    except Exception:
        g.current_user = None

    return g.current_user


@app.context_processor
def inject_template_globals():
    return {
        "current_user": get_current_user(),
    }


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if get_current_user() is None:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def normalize_patient_name(name):
    return " ".join(name.strip().split())


def normalize_email(email):
    return email.strip().lower()


def parse_input_row(source):
    return {feature: float(source[feature]) for feature in FEATURE_ORDER}


def run_prediction(input_row):
    input_df = pd.DataFrame([input_row], columns=FEATURE_ORDER)

    if model_type == "pipeline":
        try:
            target_one_prob = float(model.predict_proba(input_df)[0][1])
        except AttributeError as exc:
            if "monotonic_cst" not in str(exc):
                raise

            load_prediction_model()
            if model_type != "dnn":
                raise RuntimeError(
                    "Saved scikit-learn model is incompatible with the current environment. "
                    "Please retrain the model or restore a compatible version."
                ) from exc

            scaled_input = scaler.transform(input_df.to_numpy(dtype=float))
            target_one_prob = float(model.predict(scaled_input, verbose=0)[0][0])
    elif model_type == "dnn":
        scaled_input = scaler.transform(input_df.to_numpy(dtype=float))
        target_one_prob = float(model.predict(scaled_input, verbose=0)[0][0])
    else:
        raise RuntimeError("Unknown model type")

    disease_prob = 1 - target_one_prob if TARGET_ONE_IS_NO_DISEASE else target_one_prob
    disease_prob = calibrate_disease_probability(disease_prob, input_row["age"])

    return {
        "prediction": int(disease_prob >= 0.5),
        "probability_disease": round(disease_prob * 100, 2),
        "probability_no_disease": round((1 - disease_prob) * 100, 2),
        "risk_level": get_risk_level(disease_prob),
    }


def validate_inputs(input_row):
    age = input_row["age"]
    if age <= 0 or age > 120:
        return "Age must be between 1 and 120."

    for feature, (lower, upper) in TRAINING_BOUNDS.items():
        value = input_row[feature]
        if value < lower or value > upper:
            field_label = FIELD_LABELS.get(feature, feature)
            return (
                f"{field_label} must be between {lower} and {upper} "
                f"because the model was trained only on that range."
            )
    return None


def calibrate_disease_probability(model_prob, age):
    age_prior = interpolate_age_risk(age)
    calibrated = (model_prob * 0.5) + (age_prior * 0.5)
    return max(0.0, min(1.0, calibrated))


def interpolate_age_risk(age):
    if age <= AGE_RISK_POINTS[0][0]:
        return AGE_RISK_POINTS[0][1]
    if age >= AGE_RISK_POINTS[-1][0]:
        return AGE_RISK_POINTS[-1][1]

    for index in range(1, len(AGE_RISK_POINTS)):
        left_age, left_risk = AGE_RISK_POINTS[index - 1]
        right_age, right_risk = AGE_RISK_POINTS[index]
        if age <= right_age:
            span = right_age - left_age
            position = (age - left_age) / span
            return left_risk + position * (right_risk - left_risk)

    return AGE_RISK_POINTS[-1][1]


def get_risk_level(prob):
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Moderate"
    if prob < 0.80:
        return "High"
    return "Very High"


def build_empty_stats():
    return {
        "total_predictions": 0,
        "low_risk": 0,
        "moderate_risk": 0,
        "high_risk": 0,
        "very_high_risk": 0,
        "latest_months": [],
        "distribution": [
            {"label": "Low", "count": 0, "percent": 0, "color": "#34c759"},
            {"label": "Moderate", "count": 0, "percent": 0, "color": "#ffb020"},
            {"label": "High", "count": 0, "percent": 0, "color": "#ff5b57"},
            {"label": "Very High", "count": 0, "percent": 0, "color": "#d946ef"},
        ],
    }


def normalize_prediction_record(row):
    created_at = row["created_at"]
    if isinstance(created_at, datetime):
        created_label = created_at.strftime("%d %b %Y")
    else:
        created_label = str(created_at)

    return {
        "id": row["id"],
        "patient_name": row["patient_name"],
        "created_at": created_at,
        "created_label": created_label,
        "input": {
            "age": row["age"],
            "sex": row["sex"],
            "cp": row["cp"],
            "trestbps": row["trestbps"],
            "chol": row["chol"],
            "fbs": row["fbs"],
            "restecg": row["restecg"],
            "thalach": row["thalach"],
            "exang": row["exang"],
            "oldpeak": float(row["oldpeak"]),
            "slope": row["slope"],
            "ca": row["ca"],
            "thal": row["thal"],
        },
        "response": {
            "prediction": row["prediction"],
            "probability_disease": float(row["probability_disease"]),
            "probability_no_disease": float(row["probability_no_disease"]),
            "risk_level": row["risk_level"],
        },
    }


def fetch_predictions_for_user(user_id, limit=None):
    query = """
        SELECT id, user_id, patient_name, age, sex, cp, trestbps, chol, fbs,
               restecg, thalach, exang, oldpeak, slope, ca, thal, prediction,
               probability_disease, probability_no_disease, risk_level, created_at
        FROM predictions
    """
    conditions = ["user_id = %s"]
    params = [user_id]

    return build_predictions_query(query, conditions, params, limit)


def build_predictions_query(base_query, conditions, params, limit=None):
    query = base_query + "\nWHERE " + " AND ".join(conditions) + "\nORDER BY created_at DESC, id DESC"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    rows = execute_query(query, tuple(params), fetchall=True) or []
    return [normalize_prediction_record(row) for row in rows]


def fetch_filtered_predictions_for_user(user_id, search_term="", risk_level="", result_filter=""):
    base_query = """
        SELECT id, user_id, patient_name, age, sex, cp, trestbps, chol, fbs,
               restecg, thalach, exang, oldpeak, slope, ca, thal, prediction,
               probability_disease, probability_no_disease, risk_level, created_at
        FROM predictions
    """
    conditions = ["user_id = %s"]
    params = [user_id]

    if search_term:
        conditions.append("patient_name LIKE %s")
        params.append(f"%{search_term}%")

    if risk_level:
        conditions.append("risk_level = %s")
        params.append(risk_level)

    if result_filter == "positive":
        conditions.append("prediction = %s")
        params.append(1)
    elif result_filter == "negative":
        conditions.append("prediction = %s")
        params.append(0)

    return build_predictions_query(base_query, conditions, params)


def fetch_prediction_for_user(user_id, prediction_id):
    row = execute_query(
        """
        SELECT id, user_id, patient_name, age, sex, cp, trestbps, chol, fbs,
               restecg, thalach, exang, oldpeak, slope, ca, thal, prediction,
               probability_disease, probability_no_disease, risk_level, created_at
        FROM predictions
        WHERE user_id = %s AND id = %s
        """,
        (user_id, prediction_id),
        fetchone=True,
    )
    return normalize_prediction_record(row) if row else None


def build_dashboard_stats(records):
    if not records:
        return build_empty_stats()

    total_predictions = len(records)
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Very High": 0}
    monthly_counts = {}

    for record in records:
        risk_level = record["response"]["risk_level"]
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1

        month_key = record["created_at"].strftime("%b")
        monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1

    palette = {
        "Low": "#34c759",
        "Moderate": "#ffb020",
        "High": "#ff5b57",
        "Very High": "#d946ef",
    }

    distribution = []
    for label in ["Low", "Moderate", "High", "Very High"]:
        count = risk_counts[label]
        distribution.append(
            {
                "label": label,
                "count": count,
                "percent": round((count / total_predictions) * 100, 1),
                "color": palette[label],
            }
        )

    return {
        "total_predictions": total_predictions,
        "low_risk": risk_counts["Low"],
        "moderate_risk": risk_counts["Moderate"],
        "high_risk": risk_counts["High"],
        "very_high_risk": risk_counts["Very High"],
        "latest_months": list(monthly_counts.items())[-5:],
        "distribution": distribution,
    }


def create_prediction_for_user(user_id, patient_name, input_row, response):
    return execute_query(
        """
        INSERT INTO predictions (
            user_id, patient_name, age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal, prediction,
            probability_disease, probability_no_disease, risk_level
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING id
        """,
        (
            user_id,
            patient_name,
            int(input_row["age"]),
            int(input_row["sex"]),
            int(input_row["cp"]),
            int(input_row["trestbps"]),
            int(input_row["chol"]),
            int(input_row["fbs"]),
            int(input_row["restecg"]),
            int(input_row["thalach"]),
            int(input_row["exang"]),
            float(input_row["oldpeak"]),
            int(input_row["slope"]),
            int(input_row["ca"]),
            int(input_row["thal"]),
            int(response["prediction"]),
            float(response["probability_disease"]),
            float(response["probability_no_disease"]),
            response["risk_level"],
        ),
        commit=True,
    )


def delete_prediction_for_user(user_id, prediction_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            "DELETE FROM predictions WHERE user_id = %s AND id = %s",
            (user_id, prediction_id),
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        connection.close()


def update_user_settings(user_id, fullname, email, password_hash=None):
    if password_hash is None:
        execute_query(
            """
            UPDATE users
            SET fullname = %s, email = %s
            WHERE id = %s
            """,
            (fullname, email, user_id),
            commit=True,
        )
        return

    execute_query(
        """
        UPDATE users
        SET fullname = %s, email = %s, password_hash = %s
        WHERE id = %s
        """,
        (fullname, email, password_hash, user_id),
        commit=True,
    )


@app.route("/")
def home():
    current_user = get_current_user()
    records = fetch_predictions_for_user(current_user["id"], limit=3) if current_user else []
    stats = build_dashboard_stats(records) if current_user else build_empty_stats()
    return render_template("home.html", stats=stats, records=records)


@app.route("/about")
def about():
    current_user = get_current_user()
    records = fetch_predictions_for_user(current_user["id"]) if current_user else []
    stats = build_dashboard_stats(records) if current_user else build_empty_stats()
    return render_template("about.html", stats=stats)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if get_current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        fullname = normalize_patient_name(request.form.get("fullname", ""))
        email = normalize_email(request.form.get("email", ""))
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not fullname:
            flash("Full name is required.", "error")
        elif not email:
            flash("Email is required.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            try:
                existing_user = execute_query(
                    "SELECT id FROM users WHERE email = %s",
                    (email,),
                    fetchone=True,
                )
                if existing_user:
                    flash("An account with that email already exists.", "error")
                else:
                    execute_query(
                        """
                        INSERT INTO users (fullname, email, password_hash)
                        VALUES (%s, %s, %s)
                        """,
                        (fullname, email, generate_password_hash(password)),
                        commit=True,
                    )
                    flash("Account created. Please log in.", "success")
                    return redirect(url_for("login"))
            except Exception as exc:
                print("Could not create account:", exc)
                flash("Could not create account right now. Please try again.", "error")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_current_user():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = normalize_email(request.form.get("email", ""))
        password = request.form.get("password", "")

        try:
            user = execute_query(
                """
                SELECT id, fullname, email, password_hash
                FROM users
                WHERE email = %s
                """,
                (email,),
                fetchone=True,
            )
        except Exception as exc:
            print("Database connection failed:", exc)
            flash("Database connection failed. Please check your PostgreSQL setup.", "error")
            user = None

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    current_user = get_current_user()
    records = fetch_predictions_for_user(current_user["id"])
    stats = build_dashboard_stats(records)
    return render_template(
        "dashboard.html",
        stats=stats,
        records=records[:5],
        highlighted_name=current_user["fullname"],
        today_label=datetime.now().strftime("%d %b %Y"),
        active_page="dashboard",
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    current_user = get_current_user()

    if request.method == "POST":
        fullname = normalize_patient_name(request.form.get("fullname", ""))
        email = normalize_email(request.form.get("email", ""))
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_new_password = request.form.get("confirm_new_password", "")

        if not fullname:
            flash("Full name is required.", "error")
        elif not email:
            flash("Email is required.", "error")
        else:
            email_changed = email != current_user["email"]
            password_change_requested = bool(new_password or confirm_new_password)

            if password_change_requested and len(new_password) < 8:
                flash("New password must be at least 8 characters.", "error")
            elif password_change_requested and new_password != confirm_new_password:
                flash("New password and confirm password do not match.", "error")
            else:
                try:
                    if email_changed:
                        existing_user = execute_query(
                            "SELECT id FROM users WHERE email = %s AND id <> %s",
                            (email, current_user["id"]),
                            fetchone=True,
                        )
                        if existing_user:
                            flash("That email is already being used by another account.", "error")
                            return render_template(
                                "profile.html",
                                profile_form=request.form,
                                active_page="profile",
                            )

                    if email_changed or password_change_requested:
                        user_with_password = execute_query(
                            "SELECT password_hash FROM users WHERE id = %s",
                            (current_user["id"],),
                            fetchone=True,
                        )
                        if not current_password or not user_with_password or not check_password_hash(
                            user_with_password["password_hash"], current_password
                        ):
                            flash("Current password is required to change email or password.", "error")
                            return render_template(
                                "profile.html",
                                profile_form=request.form,
                                active_page="profile",
                            )

                    password_hash = None
                    if password_change_requested:
                        password_hash = generate_password_hash(new_password)

                    update_user_settings(current_user["id"], fullname, email, password_hash)
                    flash("Profile settings updated successfully.", "success")
                    return redirect(url_for("profile"))
                except Exception as exc:
                    print("Could not update profile settings:", exc)
                    flash("Could not update profile settings right now. Please try again.", "error")

        return render_template(
            "profile.html",
            profile_form=request.form,
            active_page="profile",
        )

    return render_template(
        "profile.html",
        profile_form={
            "fullname": current_user["fullname"],
            "email": current_user["email"],
        },
        active_page="profile",
    )


@app.route("/new-prediction")
@login_required
def new_prediction():
    return render_template(
        "new_prediction.html",
        fields=FORM_FIELDS,
        select_options=SELECT_OPTIONS,
        form_values={},
        error=None,
        active_page="new_prediction",
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        return render_template(
            "new_prediction.html",
            fields=FORM_FIELDS,
            select_options=SELECT_OPTIONS,
            form_values=request.form,
            error=get_model_unavailable_message(),
            active_page="new_prediction",
        ), 500

    current_user = get_current_user()

    try:
        patient_name = normalize_patient_name(request.form.get("patient_name", ""))
        if not patient_name:
            raise ValueError("Patient name is required.")

        input_row = parse_input_row(request.form)
        validation_error = validate_inputs(input_row)
        if validation_error:
            raise ValueError(validation_error)

        response = run_prediction(input_row)
        prediction_id = create_prediction_for_user(
            current_user["id"], patient_name, input_row, response
        )
        return redirect(url_for("prediction_result", prediction_id=prediction_id))
    except Exception as exc:
        return render_template(
            "new_prediction.html",
            fields=FORM_FIELDS,
            select_options=SELECT_OPTIONS,
            form_values=request.form,
            error=str(exc),
            active_page="new_prediction",
        ), 400


@app.route("/api/predict", methods=["POST"])
@login_required
def predict_api():
    if model is None:
        return jsonify({"error": get_model_unavailable_message()}), 500

    current_user = get_current_user()

    try:
        payload = request.get_json(silent=True) or request.form
        patient_name = normalize_patient_name(payload.get("patient_name", ""))
        if not patient_name:
            raise ValueError("Patient name is required.")

        input_row = parse_input_row(payload)
        validation_error = validate_inputs(input_row)
        if validation_error:
            raise ValueError(validation_error)

        response = run_prediction(input_row)
        prediction_id = create_prediction_for_user(
            current_user["id"], patient_name, input_row, response
        )
        response["prediction_id"] = prediction_id
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predictions")
@login_required
def predictions():
    current_user = get_current_user()
    search_term = request.args.get("search", "").strip()
    risk_level = request.args.get("risk_level", "").strip()
    result_filter = request.args.get("result", "").strip().lower()
    records = fetch_filtered_predictions_for_user(
        current_user["id"],
        search_term=search_term,
        risk_level=risk_level,
        result_filter=result_filter,
    )
    return render_template(
        "predictions.html",
        records=records,
        filters={
            "search": search_term,
            "risk_level": risk_level,
            "result": result_filter,
        },
        risk_level_options=RISK_LEVEL_OPTIONS,
        active_page="predictions",
    )


@app.route("/prediction/<int:prediction_id>")
@login_required
def prediction_result(prediction_id):
    current_user = get_current_user()
    record = fetch_prediction_for_user(current_user["id"], prediction_id)
    if record is None:
        flash("Prediction not found.", "warning")
        return redirect(url_for("predictions"))

    return render_template(
        "prediction_result.html",
        record=record,
        active_page="predictions",
    )


@app.route("/prediction/<int:prediction_id>/delete", methods=["POST"])
@login_required
def delete_prediction(prediction_id):
    current_user = get_current_user()
    deleted = delete_prediction_for_user(current_user["id"], prediction_id)
    if deleted:
        flash("Prediction deleted successfully.", "success")
    else:
        flash("Prediction not found or already removed.", "warning")
    return redirect(url_for("predictions"))


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", "5000"))
    print(f"Flask app running on http://localhost:{port}")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
