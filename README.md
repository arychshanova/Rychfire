# Rychfire — Burning Activity Calendar

Early prototype for exploring the NASA Space Apps Challenge concept.

**Important:** v0.1 uses reproducible *synthetic* data. No NASA FIRMS data or MODIS/VIIRS harmonization is included yet.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free

1. Create a public GitHub repository.
2. Add `app.py` and `requirements.txt` at the repository root.
3. In Streamlit Community Cloud, select the repository, branch `main`, and entrypoint `app.py`.
4. Deploy.

Future work: real NASA FIRMS observations, area-of-interest drawing, validated sensor harmonization, and trustworthy historical baselines.
