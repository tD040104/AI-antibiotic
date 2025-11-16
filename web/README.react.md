React frontend (no-build) for the AI-Antibiotic project

Files added:
- `react_index.html` - static page that loads React via CDN and mounts the form.
- `react_app.js` - the React form and client logic. Submits to `process_form.php` and shows the response.

How to use:
1. Place these files under `web/` (already added).
2. This repository now includes a small Flask API and a React frontend (Vite). The recommended flow is:

   - Run the Python API server (Flask) which exposes POST /api/predict and loads the trained orchestrator state.
   - Run the React dev server and open the app in the browser. The React app will POST JSON to the Flask API.

Run the backend API (from project root):

```powershell
# install Python deps (use your environment)
pip install -r requirements.txt

# run the Flask API
python web/api_server.py
```

By default the Flask server listens on port 5000. It requires `models/orchestrator_state.joblib` to be present to enable predictions (this is created after training).

Run the frontend (from `web/frontend`):

```powershell
cd web/frontend
npm install
npm run dev
```

Open the URL shown by Vite (usually http://localhost:5173). Fill the form and submit — the app will call `http://localhost:5000/api/predict`.

Notes and next steps:
- This is a lightweight, no-build prototype. For production use, consider creating a proper React app (Create React App / Vite) and an API endpoint that returns stable JSON.
- If you prefer the frontend to talk to a JSON API, you can modify `process_form.php` to always return JSON and skip redirects. The current `process_form.php` already echoes JSON when the Python script returns JSON.
- If you want the frontend to validate more fields or display recommendation cards, I can extend the UI to parse known response fields (e.g., `recommendations_json`, `prediction_json`).
Notes and migration tips:
- The old PHP-based pages remain in `web/` for reference (`index.php`, `process_form.php`). You can remove them if you fully migrate to the React + Flask stack.
- For production, consider running the Flask API behind a WSGI server (gunicorn) and building the React app (`npm run build`) and serving static files via nginx or similar.