# Term-Deposit Campaign Planner

A Streamlit academic project built around the UCI Bank Marketing **bank-additional-full.csv** dataset.

## Features
- Dashboard with customer and subscription KPIs
- Subscription rate charts by contact channel and month
- Campaign Planner with SQLite storage (create, view, delete)
- Customer segment insights by job and age group
- Random Forest subscription predictor
- Model and dataset information page

## Dataset
The included file is `data/bank-additional-full.csv`. It is the semicolon-separated UCI Bank Marketing additional dataset (41,188 rows, 21 columns).

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Run locally (Windows PowerShell)
Open this folder in VS Code, then open Terminal and run:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this in the same terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

Or use Command Prompt:
```bat
venv\Scripts\activate.bat
```

## GitHub
Create an empty GitHub repository, then run these commands in the project folder:

```bash
git init
git add .
git commit -m "Initial Term Deposit Campaign Planner"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your own repository URL.

## Deploy on Streamlit Community Cloud
1. Push the project to GitHub.
2. Open https://share.streamlit.io/
3. Sign in with GitHub and choose **New app**.
4. Select your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**.

The dataset is included in the repository, so no manual upload is needed after deployment.

## Important academic note: duration leakage
The original dataset has a `duration` column (call duration). It is deliberately excluded from the prediction model because it is only known after a phone call. Using it to decide whom to call before the call would leak information that is not available at prediction time.

## Notes
- The app trains the model on the first visit to the Predictor page and caches it for the current app session.
- SQLite data on Streamlit Community Cloud may reset when the app is restarted because the cloud app filesystem is not a permanent database service.
- Predictions are educational estimates, not guarantees or financial advice.
