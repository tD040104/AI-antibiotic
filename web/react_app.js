const { useState } = React;

function App() {
  const [form, setForm] = useState({
    patient_name: 'Nguyễn Văn A',
    age: '',
    gender: 'F',
    souches: 'S123 Escherichia coli',
    diabetes: 'No',
    hypertension: 'No',
    hospital_before: 'No',
    infection_freq: 0,
    collection_date: '',
    notes: ''
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function onChange(e) {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  }

  async function onSubmit(e) {
    e.preventDefault();
    setResult(null);
    setError(null);

    // basic validation
    if (!form.patient_name || !form.age || !form.souches || !form.collection_date) {
      setError('Vui lòng điền tên, tuổi, chủng vi khuẩn và ngày lấy mẫu.');
      return;
    }

    setLoading(true);
    try {
      const fd = new FormData();
      // match field names expected by process_form.php
      fd.append('patient_name', form.patient_name);
      fd.append('age', form.age);
      fd.append('gender', form.gender);
      fd.append('souches', form.souches);
      fd.append('diabetes', form.diabetes);
      fd.append('hypertension', form.hypertension);
      fd.append('hospital_before', form.hospital_before);
      fd.append('infection_freq', form.infection_freq);
      fd.append('collection_date', form.collection_date);
      fd.append('notes', form.notes);

      const resp = await fetch('./process_form.php', {
        method: 'POST',
        body: fd,
        credentials: 'same-origin'
      });

      const text = await resp.text();

      // try parse JSON
      let parsed = null;
      try { parsed = JSON.parse(text); } catch (err) { parsed = null; }

      if (parsed) {
        // Most likely JSON returned by predict_patient.py or process_form
        setResult(parsed);
      } else {
        // not JSON -> show raw HTML/text (could include debug info)
        setResult({ raw: text });
      }
    } catch (err) {
      setError('Lỗi khi gửi dữ liệu: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={onSubmit} className="row g-3">
        {error && <div className="alert alert-danger">{error}</div>}

        <div className="col-md-4">
          <label className="form-label">Tên bệnh nhân</label>
          <input name="patient_name" value={form.patient_name} onChange={onChange} className="form-control" />
        </div>

        <div className="col-md-2">
          <label className="form-label">Tuổi</label>
          <input name="age" type="number" value={form.age} onChange={onChange} className="form-control" />
        </div>

        <div className="col-md-2">
          <label className="form-label">Giới tính</label>
          <select name="gender" value={form.gender} onChange={onChange} className="form-select">
            <option value="F">Nữ</option>
            <option value="M">Nam</option>
          </select>
        </div>

        <div className="col-md-6">
          <label className="form-label">Chủng vi khuẩn (Souches)</label>
          <input name="souches" value={form.souches} onChange={onChange} className="form-control" />
        </div>

        <div className="col-md-3">
          <label className="form-label">Tiểu đường</label>
          <select name="diabetes" value={form.diabetes} onChange={onChange} className="form-select">
            <option value="Yes">Có</option>
            <option value="No">Không</option>
          </select>
        </div>

        <div className="col-md-3">
          <label className="form-label">Tăng huyết áp</label>
          <select name="hypertension" value={form.hypertension} onChange={onChange} className="form-select">
            <option value="Yes">Có</option>
            <option value="No">Không</option>
          </select>
        </div>

        <div className="col-md-3">
          <label className="form-label">Tiền sử nhập viện</label>
          <select name="hospital_before" value={form.hospital_before} onChange={onChange} className="form-select">
            <option value="Yes">Có</option>
            <option value="No">Không</option>
          </select>
        </div>

        <div className="col-md-3">
          <label className="form-label">Số lần nhiễm trùng/năm</label>
          <input name="infection_freq" type="number" step="0.1" value={form.infection_freq} onChange={onChange} className="form-control" />
        </div>

        <div className="col-md-3">
          <label className="form-label">Ngày lấy mẫu</label>
          <input name="collection_date" type="date" value={form.collection_date} onChange={onChange} className="form-control" />
        </div>

        <div className="col-12">
          <label className="form-label">Ghi chú</label>
          <textarea name="notes" value={form.notes} onChange={onChange} className="form-control" rows="4"></textarea>
        </div>

        <div className="col-12 text-end">
          <button className="btn btn-primary" disabled={loading} type="submit">{loading ? 'Đang gửi...' : 'Gửi dữ liệu & Dự đoán'}</button>
        </div>
      </form>

      <hr />

      <div>
        <h5>Kết quả</h5>
        {loading && <div>Đang chờ phản hồi từ server...</div>}
        {!loading && result && (
          <div>
            {result.raw && <div dangerouslySetInnerHTML={{__html: result.raw}} />}
            {!result.raw && (
              <pre style={{whiteSpace: 'pre-wrap'}}>{JSON.stringify(result, null, 2)}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);