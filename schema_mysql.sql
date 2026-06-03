CREATE DATABASE IF NOT EXISTS heartsense;
USE heartsense;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fullname VARCHAR(100) NOT NULL,
    email VARCHAR(120) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_users_email (email)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    patient_name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    sex TINYINT NOT NULL,
    cp TINYINT NOT NULL,
    trestbps INT NOT NULL,
    chol INT NOT NULL,
    fbs TINYINT NOT NULL,
    restecg TINYINT NOT NULL,
    thalach INT NOT NULL,
    exang TINYINT NOT NULL,
    oldpeak DECIMAL(4,1) NOT NULL,
    slope TINYINT NOT NULL,
    ca TINYINT NOT NULL,
    thal TINYINT NOT NULL,
    prediction TINYINT NOT NULL,
    probability_disease DECIMAL(5,2) NOT NULL,
    probability_no_disease DECIMAL(5,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_predictions_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    INDEX idx_predictions_user_created (user_id, created_at),
    INDEX idx_predictions_patient_name (patient_name),
    INDEX idx_predictions_risk_level (risk_level)
);

-- If you want to rebuild cleanly from scratch in Workbench, run this first:
-- DROP TABLE IF EXISTS predictions;
-- DROP TABLE IF EXISTS users;
-- Then run this whole file again.
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM predictions;
SELECT * FROM users;
SELECT * FROM predictions;