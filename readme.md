# HeartSense

HeartSense is a Flask-based heart disease prediction app with:
- user signup, login, and logout
- user-specific prediction history saved in MySQL
- dashboard, prediction form, result page, and history pages
- a saved scikit-learn pipeline with DNN fallback support

## Setup
1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install the app requirements.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Create the MySQL database and tables from [schema_mysql.sql](E:/heartsense/schema_mysql.sql).

4. Optional: set custom environment variables if you do not want to use the defaults in [app.py](E:/heartsense/app.py).

```powershell
$env:MYSQL_HOST="127.0.0.1"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your-password"
$env:MYSQL_DATABASE="heartsense"
$env:FLASK_SECRET_KEY="replace-this"
```

5. Run the Flask app.

```powershell
python app.py
```

6. Open the app in your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Model Files
The app expects these saved files inside [models](E:/heartsense/models):
- `heartsense_pipeline.pkl`
- `heartsense_dnn.h5`
- `dnn_scaler.pkl`

If you need to retrain the scikit-learn pipeline:

```powershell
python train_model.py
```

The optional research script [train_dnn_talos.py](E:/heartsense/train_dnn_talos.py) is not required to run the Flask app and may need extra package compatibility work if you want to use it.

## Notes
- This project is for educational and demonstration use only, not medical advice.
- Prediction inputs are validated before inference and saved per logged-in user.
- If you switch Python environments, keep `scikit-learn==1.3.2` to stay compatible with the saved pipeline model.
