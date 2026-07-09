# AQI Predictor - Deploy on Streamlit Community Cloud

## Steps
1. Push this whole folder (`app.py`, `requirements.txt`, `runtime.txt`, `models/`) to GitHub.
2. Go to https://share.streamlit.io and choose "New app".
3. Select your repo, branch, and set the main file path to `app.py`.
4. Click **Deploy**.

If your GitHub repo contains this folder inside a larger repo, set the main file path to `aqi_app/app.py`.

## Run locally first
```
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- `runtime.txt` pins Streamlit Cloud to Python 3.11 so the ML dependencies install from prebuilt wheels.
- Models were pickled with scikit-learn 1.7.1, so `requirements.txt` pins that version to avoid version-mismatch warnings.
- `models/` is under GitHub's 100MB file limit, so a normal push works without Git LFS.
