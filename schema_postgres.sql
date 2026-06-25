CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    fullname VARCHAR(100) NOT NULL,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    patient_name VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    sex SMALLINT NOT NULL,
    cp SMALLINT NOT NULL,
    trestbps INTEGER NOT NULL,
    chol INTEGER NOT NULL,
    fbs SMALLINT NOT NULL,
    restecg SMALLINT NOT NULL,
    thalach INTEGER NOT NULL,
    exang SMALLINT NOT NULL,
    oldpeak NUMERIC(4,1) NOT NULL,
    slope SMALLINT NOT NULL,
    ca SMALLINT NOT NULL,
    thal SMALLINT NOT NULL,
    prediction SMALLINT NOT NULL,
    probability_disease NUMERIC(5,2) NOT NULL,
    probability_no_disease NUMERIC(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_created
    ON predictions (user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_patient_name
    ON predictions (patient_name);

CREATE INDEX IF NOT EXISTS idx_predictions_risk_level
    ON predictions (risk_level);
