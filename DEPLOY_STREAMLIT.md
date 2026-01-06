# Deploy to Streamlit Community Cloud

This project can be deployed quickly to Streamlit Community Cloud. Follow these steps.

## 1. Confirm repository
Ensure your repository is pushed to GitHub and reachable.

## 2. Choose entrypoint
When creating the app on Streamlit Cloud, set the "Main file path" to `main.py`.

## 3. Requirements
Make sure `requirements.txt` lists necessary dependencies (Streamlit, langchain/openai libs if used). Example:
```
streamlit==1.41.1
openai
python-dotenv
langchain
```

## 4. Secrets / Environment variables
Set any secrets required by the app (e.g. `OPENAI_API_KEY`) in Streamlit Cloud UI:
- Go to the deployed app → Settings → Secrets
- Add `OPENAI_API_KEY` and other keys

## 5. Data handling
- Avoid committing large or sensitive ZIPs to the repo. Prefer external storage (S3) and reference via environment variables.
- If you keep small sample files in `date/`, ensure they are in the repo or available via remote storage.

## 6. Advanced settings
- If you use external services (S3, DB), set credentials as secrets.
- For heavier workloads or scale, consider using Render or Cloud Run (containerized).

## 7. Troubleshooting
- Logs: Streamlit Cloud provides logs in the app dashboard.
- If the app fails to boot, check `requirements.txt` and the selected main file (`main.py`).

---
If you want, I can also create a `Dockerfile` or Render-specific `service` config next.