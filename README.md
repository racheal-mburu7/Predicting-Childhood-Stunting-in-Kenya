# Child Stunting Risk Predictor — App

A small Streamlit app that lets a parent or a health worker enter a child's
details and get a stunting-risk prediction (Severe / Moderate / Normal
growth) plus plain-language dietary guidance. Built on the model trained in
`Predicting_Childhood_Stunting_in_Kenya.ipynb`.

## Files

- `app.py` — the app itself
- `stunting_classifier.joblib` — the trained model pipeline
- `model_metadata.json` — feature list, class labels, and test-set metrics
- `requirements.txt` — exact package versions to install

## Run it offline (e.g. on a clinic laptop with no internet)

1. Install Python 3.10+ if it isn't already installed.
2. In this folder, run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Streamlit opens automatically in a browser at `http://localhost:8501`.
   No internet connection is needed after the one-time install — everything
   (the model, the app) runs locally on the machine.

## Put it on the web (so it opens from a phone/browser link)

Easiest free option — **Streamlit Community Cloud**:
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, sign in, and point it at the repo /
   `app.py`.
3. It builds and gives you a public URL (e.g.
   `https://your-app-name.streamlit.app`) that works on any phone or laptop
   browser — no install needed on the user's end.

Any other host that runs a Python web app (Render, Railway, a hospital's own
server, etc.) works the same way — the app has no special requirements
beyond what's in `requirements.txt`.

## Important limitations to keep in front of users

- This is a **screening aid**, not a diagnosis. It's built from survey data,
  not a physical measurement of the child in front of you.
- Model performance (see the "How reliable is this prediction?" panel in the
  app) is modest, especially for the rare "Severe stunting" class — treat a
  "Normal growth" result as reassuring, not conclusive, and always prefer an
  actual height/age measurement and a health worker's judgment.
- The model was trained on national KDHS data; if you deploy this in a
  specific county or clinic population, its accuracy there hasn't been
  separately verified.
