from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback
import numpy as np
import pandas as pd


# Ensure project root is on sys.path so we can import main.MultiAgentOrchestrator
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main import MultiAgentOrchestrator
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Try to load orchestrator state at startup
ORCHESTRATOR_STATE = os.path.join(os.path.dirname(script_dir), 'models', 'orchestrator_state.joblib')
orchestrator = None
if os.path.exists(ORCHESTRATOR_STATE):
    try:
        orchestrator = MultiAgentOrchestrator.load_from_state(ORCHESTRATOR_STATE)
        print(f"Loaded orchestrator state from {ORCHESTRATOR_STATE}")
    except Exception as e:
        print(f"Failed to load orchestrator state: {e}")
        traceback.print_exc()


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'orchestrator_loaded': orchestrator is not None})


@app.route('/', methods=['GET'])
def index():
    return (
        '<h3>AI Antibiotic API</h3>'
        f'<p>orchestrator_loaded: {orchestrator is not None}</p>'
        '<p>Use <code>/api/predict</code> (POST) or <code>/api/health</code> (GET).</p>'
    )


@app.before_request
def log_request_info():
    try:
        remote = request.remote_addr
        host = request.host
        print(f"Incoming request from {remote} Host:{host} -> {request.method} {request.path}")
        # print headers summary
        hdrs = {k: request.headers.get(k) for k in ['User-Agent','Referer','Origin','Content-Type'] if request.headers.get(k)}
        if hdrs:
            print(f"  Headers: {hdrs}")
    except Exception:
        pass


@app.route('/api/predict', methods=['POST'])
def predict():
    global orchestrator
    if orchestrator is None:
        # Dev-friendly fallback: return a dummy prediction so frontend can be tested without training
        try:
            patient = request.get_json(force=True)
        except Exception:
            patient = {}

        # Build a deterministic dummy response based on input
        name = patient.get('patient_name', 'Bệnh nhân')
        souches = patient.get('souches', 'Unknown')

        dummy_probabilities = {
            'AMX/AMP': 0.12,
            'AMC': 0.34,
            'CZ': 0.45,
            'FOX': 0.7,
            'CTX/CRO': 0.22,
        }

        # Create recommendations sorted by probability desc
        recs = []
        rank = 1
        for k, v in sorted(dummy_probabilities.items(), key=lambda x: -x[1]):
            recs.append({
                'rank': rank,
                'antibiotic_name': k,
                'sensitive_probability': v,
                'confidence': 'low' if v < 0.3 else ('medium' if v < 0.6 else 'high')
            })
            rank += 1

        dummy_result = {
            'predictions': {k: (1 if v > 0.5 else 0) for k, v in dummy_probabilities.items()},
            'probabilities': dummy_probabilities,
            'recommendations': recs,
            'explanation': {'note': f'Dummy explanation for {name} with {souches}'},
            'report': f'Dummy report: patient {name}, souches {souches}. This is a placeholder response.'
        }

        return jsonify({'status': 'ok', 'data': dummy_result, 'note': 'dummy'}), 200

    try:
        patient = request.get_json(force=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Invalid JSON: {e}'}), 400

    try:
        # Map incoming generic form fields to the orchestrator's expected keys
        # Orchestrator.predict expects something like:
        # { 'age/gender': '45/F', 'Souches': 'S123 Escherichia coli', 'Diabetes': 'Yes', ... }
        mapped = {}
        # age and gender -> age/gender
        age = patient.get('age') or patient.get('age/gender') or patient.get('Age')
        gender = patient.get('gender') or patient.get('sex') or patient.get('Gender')
        if age and gender:
            mapped['age/gender'] = f"{age}/{gender}"
        elif age and '/' in str(age):
            mapped['age/gender'] = str(age)

        # Souches / strain
        if 'souches' in patient:
            mapped['Souches'] = patient.get('souches')
        elif 'Souches' in patient:
            mapped['Souches'] = patient.get('Souches')

        # Comorbidities and flags
        mapped['Diabetes'] = patient.get('diabetes', patient.get('Diabetes', 'No'))
        mapped['Hypertension'] = patient.get('hypertension', patient.get('Hypertension', 'No'))
        # hospital_before may be provided as 'hospital_before' or 'Hospital_before' or 'hospitalHistory'
        mapped['Hospital_before'] = patient.get('hospital_before', patient.get('Hospital_before', patient.get('hospitalHistory', 'No')))

        # Blood pressure (huyết áp)
        mapped['Blood_Pressure'] = patient.get('blood_pressure', patient.get('Blood_Pressure', ''))

        # Numeric / frequency
        try:
            mapped['Infection_Freq'] = float(patient.get('infection_freq', patient.get('Infection_Freq', 0) or 0))
        except Exception:
            mapped['Infection_Freq'] = 0.0

        # Date and notes
        mapped['Collection_Date'] = patient.get('collection_date', patient.get('Collection_Date', ''))
        mapped['Notes'] = patient.get('notes', patient.get('Notes', ''))

        # Pass through other fields if present
        for k in ['patient_name']:
            if k in patient:
                mapped[k] = patient[k]

        # Call orchestrator
        result = orchestrator.predict(mapped)

        # Convert numpy/pandas types into JSON-serializable Python types
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return make_serializable(obj.to_dict())
            if isinstance(obj, pd.DataFrame):
                return make_serializable(obj.to_dict(orient='records'))
            return obj

        serial = make_serializable(result)
        return jsonify({'status': 'ok', 'data': serial})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Development server
    # Turn off the reloader to avoid double-import issues with heavy ML libs.
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
