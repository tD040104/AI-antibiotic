import React, { useState } from 'react'

function RecommendationCard({r}){
  const prob = r.sensitive_probability ?? r.probability ?? r.resistance_probability ?? null
  const conf = r.confidence ?? r.conf ?? r.confidence_score ?? ''
  return (
    <div className="rec-card">
      <h5>{r.antibiotic_name || r.antibiotic || r.name || 'Unknown'}</h5>
      {prob !== null && <div className="rec-meta">Xác suất nhạy: {(prob*100).toFixed(1)}%</div>}
      <div className="rec-meta">Độ tin cậy: {conf || '—'}</div>
      <div className="rec-meta">Rank: {r.rank ?? r.rank_order ?? '—'}</div>
    </div>
  )
}

export default function App() {
  const [form, setForm] = useState({
    patient_name: '',
    age: '',
    gender: 'F',
    souches: '',
    diabetes: 'No',
    hypertension: 'No',
    hospital_before: 'No',
    blood_pressure: '',
    infection_freq: 0,
    collection_date: '',
    notes: ''
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  function onChange(e) {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: value }))
  }

  async function onSubmit(e) {
    e.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      })
      const data = await resp.json()
      if (!resp.ok) setError(data.message || JSON.stringify(data))
      else setResult(data.data || data)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  function loadPatientSummaryIntoForm(){
    if(!result) return
    const s = result.patient_summary || result.patient || result.patientSummary || result.patient_summary || {}
    // map common keys
    const mapped = {}
    if(s.age) mapped.age = String(s.age)
    if(s.gender) mapped.gender = s.gender === 'Male' ? 'M' : (s.gender === 'Female' ? 'F' : s.gender)
    if(s.bacteria || s.bacteria_name || s.Souches) mapped.souches = s.bacteria || s.bacteria_name || s.Souches
    if(s.diabetes) mapped.diabetes = s.diabetes
    if(s.hypertension) mapped.hypertension = s.hypertension
    if(s.blood_pressure) mapped.blood_pressure = s.blood_pressure
    if(s.infection_freq !== undefined) mapped.infection_freq = s.infection_freq
    if(s.collection_date) mapped.collection_date = s.collection_date
    if(s.hospital_before !== undefined) mapped.hospital_before = s.hospital_before
    setForm(prev => ({...prev, ...mapped}))
  }

  // helpers to extract recommendations
  function getRecommendations(res){
    if(!res) return []
    if(res.recommendations && Array.isArray(res.recommendations)) return res.recommendations
    if(res.resistance_predictions){
      // older structure: resistance_predictions: { resistant: [...], sensitive: [...] }
      const rp = res.resistance_predictions
      const arr = []
      if(Array.isArray(rp.resistant)) arr.push(...rp.resistant)
      if(Array.isArray(rp.sensitive)) arr.push(...rp.sensitive)
      return arr
    }
    if(res.predictions && res.probabilities){
      // synthesize list from probabilities
      return Object.keys(res.probabilities).map((k,i)=>({antibiotic_name:k, probability:res.probabilities[k], rank:i+1}))
    }
    return []
  }

  const recs = getRecommendations(result)

  // Compute top antibiotic according to model probabilities (from resistance_predictions or probabilities)
  function getTopByProbability(res){
    if(!res) return null
    // prefer structured sensitive list
    try{
      if(res.resistance_predictions && Array.isArray(res.resistance_predictions.sensitive) && res.resistance_predictions.sensitive.length>0){
        const arr = res.resistance_predictions.sensitive.slice().map(x=>({
          name: (typeof x==='object') ? (x.antibiotic || x.name || x.code) : x,
          prob: (typeof x==='object') ? (x.probability ?? x.sensitive_probability ?? x.resistance_probability ?? null) : null
        }))
        arr.sort((a,b)=> (b.prob ?? 0) - (a.prob ?? 0))
        return arr[0]
      }
      // fallback to probabilities map
      if(res.probabilities && typeof res.probabilities === 'object'){
        const keys = Object.keys(res.probabilities)
        if(keys.length===0) return null
        keys.sort((a,b)=> res.probabilities[b] - res.probabilities[a])
        return { name: keys[0], prob: res.probabilities[keys[0]] }
      }
    }catch(e){
      return null
    }
    return null
  }

  const topByProb = getTopByProbability(result)
  const reportRecMatch = (result && result.report) ? ( // try to extract medicine name from report heuristically
    (result.report.match(/[Kk]huyến nghị:\s*Xem xét sử dụng\s*([^\(\n]+)/) || result.report.match(/Khuyến nghị:\s*([^\(\n]+)/) || [])[1]
  ) : null

  // Helper: render explanation object into readable text (not raw JSON)
  function renderKeyFactors(kf){
    if(!Array.isArray(kf)) return null
    return (
      <div style={{marginBottom:8}}>
        <strong>Yếu tố chính:</strong>
        <ul>
          {kf.map((it,i)=>(<li key={i}>{it.feature || it.name || JSON.stringify(it)}: {String(it.value ?? it.score ?? '')}</li>))}
        </ul>
      </div>
    )
  }

  function renderPatientSummary(ps){
    if(!ps || typeof ps !== 'object') return null
    const pairs = []
    for(const k of ['age','gender','bacteria','blood_pressure','diabetes','hypertension','infection_freq','hospital_before','collection_date']){
      if(ps[k] !== undefined && ps[k] !== null && ps[k] !== '') pairs.push([k, ps[k]])
    }
    if(pairs.length===0) return null
    return (
      <div style={{marginBottom:8}}>
        <strong>Tóm tắt (từ explanation)</strong>
        <div>
          {pairs.map(([k,v],i)=>(<div key={i} className="rec-meta">{k}: {String(v)}</div>))}
        </div>
      </div>
    )
  }

  function renderResistancePred(rp){
    if(!rp || typeof rp !== 'object') return null
    return (
      <div style={{marginBottom:8}}>
        <strong>Dự đoán kháng/nhạy:</strong>
        {Array.isArray(rp.resistant) && rp.resistant.length>0 && (
          <div>
            <em>Resistant:</em>
            <ul>
              {rp.resistant.map((x,i)=>{
                const name = (typeof x === 'object') ? (x.antibiotic || x.name || x.code || JSON.stringify(x)) : x
                const prob = (typeof x === 'object') ? (x.resistance_probability ?? x.probability ?? x.resistance ?? null) : null
                return (<li key={i}>{name}{prob !== null ? ` — P(resistant): ${(prob*100).toFixed(1)}%` : ''}</li>)
              })}
            </ul>
          </div>
        )}
        {Array.isArray(rp.sensitive) && rp.sensitive.length>0 && (
          <div>
            <em>Sensitive:</em>
            <ul>
              {rp.sensitive.map((x,i)=>{
                const name = (typeof x === 'object') ? (x.antibiotic || x.name || x.code || JSON.stringify(x)) : x
                const prob = (typeof x === 'object') ? (x.probability ?? x.sensitive_probability ?? x.resistance_probability ?? null) : null
                return (<li key={i}>{name}{prob !== null ? ` — P(sensitive): ${(prob*100).toFixed(1)}%` : ''}</li>)
              })}
            </ul>
          </div>
        )}
      </div>
    )
  }

  function renderExplanation(expl){
    if(!expl) return null
    if(typeof expl === 'string') return (<div style={{whiteSpace:'pre-wrap'}}>{expl}</div>)
    const parts = []
    if(expl.report) parts.push(<div key="r" style={{marginBottom:8}}><strong>Báo cáo:</strong><div style={{whiteSpace:'pre-wrap', marginTop:6}}>{expl.report}</div></div>)
    if(expl.note) parts.push(<div key="n" style={{marginBottom:8}}>{expl.note}</div>)
    if(expl.key_factors) parts.push(<div key="kf">{renderKeyFactors(expl.key_factors)}</div>)
    if(expl.patient_summary) parts.push(<div key="ps">{renderPatientSummary(expl.patient_summary)}</div>)
    if(expl.resistance_predictions) parts.push(<div key="rp">{renderResistancePred(expl.resistance_predictions)}</div>)
    if(parts.length===0) return (<pre className="json-out">{JSON.stringify(expl, null, 2)}</pre>)
    return (<div>{parts}</div>)
  }

  return (
    <div className="container">
      <div className="card">
        <h2>AI Antibiotic - Form</h2>
        {error && <div className="alert alert-error">{error}</div>}
        <form onSubmit={onSubmit}>
          <div className="form-grid">
            <div className="form-group">
              <label>Tên bệnh nhân</label>
              <input name="patient_name" value={form.patient_name} onChange={onChange} />
            </div>
            <div className="form-group">
              <label>Tuổi</label>
              <input name="age" value={form.age} onChange={onChange} />
            </div>
            <div className="form-group">
              <label>Giới tính</label>
              <select name="gender" value={form.gender} onChange={onChange}><option value="F">Nữ</option><option value="M">Nam</option></select>
            </div>
            <div className="form-group">
              <label>Chủng vi khuẩn</label>
              <input name="souches" value={form.souches} onChange={onChange} />
            </div>
            <div className="form-group">
              <label>Huyết áp (vd. 120/80)</label>
              <input name="blood_pressure" value={form.blood_pressure} onChange={onChange} />
            </div>
            <div className="form-group">
              <label>Tiểu đường</label>
              <select name="diabetes" value={form.diabetes} onChange={onChange}><option value="No">Không</option><option value="Yes">Có</option></select>
            </div>
            <div className="form-group">
              <label>Tăng huyết áp</label>
              <select name="hypertension" value={form.hypertension} onChange={onChange}><option value="No">Không</option><option value="Yes">Có</option></select>
            </div>
            <div className="form-group">
              <label>Số lần nhiễm trùng / năm</label>
              <input type="number" min="0" name="infection_freq" value={form.infection_freq} onChange={onChange} />
            </div>
            <div className="form-group">
              <label>Tiền sử nhập viện</label>
              <select name="hospital_before" value={form.hospital_before} onChange={onChange}><option value="No">Không</option><option value="Yes">Có</option></select>
            </div>
            <div className="form-group">
              <label>Ngày lấy mẫu</label>
              <input type="date" name="collection_date" value={form.collection_date} onChange={onChange} />
            </div>
            <div className="form-group" style={{gridColumn:'1 / -1'}}>
              <label>Ghi chú</label>
              <textarea name="notes" value={form.notes} onChange={onChange} />
            </div>
          </div>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginTop:12}}>
            <div>
              <button className="btn-primary" type="submit" disabled={loading}>{loading ? 'Đang gửi...' : 'Gửi'}</button>
            </div>
            <div>
              <button type="button" onClick={loadPatientSummaryIntoForm} style={{marginLeft:8}}>Load into form</button>
            </div>
          </div>
        </form>
        <div className="divider" />

        <div>
          <h4>Kết quả</h4>

          {result && (
            <div style={{display:'grid', gridTemplateColumns:'1fr 320px', gap:18}}>
              <div>
                {/* Patient summary */}
                { (result.patient_summary || result.patient) && (
                  <div style={{marginBottom:12}} className="rec-card">
                    <h5>Tóm tắt bệnh nhân</h5>
                    <div className="rec-meta">Tuổi: {result.patient_summary?.age ?? result.patient?.age ?? '—'}</div>
                    <div className="rec-meta">Giới tính: {result.patient_summary?.gender ?? result.patient?.gender ?? '—'}</div>
                    <div className="rec-meta">Vi khuẩn: {result.patient_summary?.bacteria ?? result.patient?.bacteria ?? result.Souches ?? '—'}</div>
                    <div className="rec-meta">Huyết áp: {(result.patient_summary?.blood_pressure ?? result.patient?.blood_pressure) || form.blood_pressure || '—'}</div>
                    <div className="rec-meta">Tiểu đường: {result.patient_summary?.diabetes ?? result.patient?.diabetes ?? (form.diabetes === 'Yes' ? 'Có' : (form.diabetes === 'No' ? 'Không' : form.diabetes)) }</div>
                    <div className="rec-meta">Tăng huyết áp: {result.patient_summary?.hypertension ?? result.patient?.hypertension ?? (form.hypertension === 'Yes' ? 'Có' : (form.hypertension === 'No' ? 'Không' : form.hypertension)) }</div>
                    <div className="rec-meta">Số lần nhiễm trùng / năm: {result.patient_summary?.infection_freq ?? result.patient?.infection_freq ?? form.infection_freq ?? '—'}</div>
                    <div className="rec-meta">Tiền sử nhập viện: {result.patient_summary?.hospital_before ?? result.patient?.hospital_before ?? (form.hospital_before === 'Yes' ? 'Có' : (form.hospital_before === 'No' ? 'Không' : form.hospital_before))}</div>
                    <div className="rec-meta">Ngày lấy mẫu: {result.patient_summary?.collection_date ?? result.patient?.collection_date ?? form.collection_date ?? '—'}</div>
                    <div style={{marginTop:8}}><strong>Báo cáo</strong><div style={{whiteSpace:'pre-wrap', marginTop:6}}>{result.report ?? result.report_text ?? ''}</div></div>
                  </div>
                )}

                {/* Report / explanation */}
                { result.explanation && (
                  <div style={{marginBottom:12}} className="rec-card">
                    <div style={{marginTop:6}}>{renderExplanation(result.explanation)}</div>
                  </div>
                )}

                {/* Raw fallback */}
                { !result.recommendations && !result.probabilities && !result.report && !result.explanation && (
                  <pre className="json-out">{JSON.stringify(result, null, 2)}</pre>
                )}
              </div>

              <div>
                <h5>Khuyến nghị</h5>
                {recs.length === 0 && <div>Không có khuyến nghị</div>}
                {/* show report recommendation vs model top-by-probability */}
                {result && result.report && reportRecMatch && (
                  <div style={{marginBottom:8}} className="rec-card">
                    <strong>Khuyến nghị từ báo cáo:</strong>
                    <div className="rec-meta">{reportRecMatch.trim()}</div>
                    {topByProb && reportRecMatch && topByProb.name && topByProb.name.toLowerCase().includes(String(reportRecMatch).toLowerCase())===false && (
                      <div style={{marginTop:6, color:'#a66'}} className="rec-meta">Lưu ý: Báo cáo đề xuất <strong>{reportRecMatch.trim()}</strong> nhưng model cho xác suất cao nhất với <strong>{topByProb.name}</strong> ({topByProb.prob ? (Math.round(topByProb.prob*1000)/10)+'%' : 'n/a'}). Có thể do quy tắc chọn thuốc trong báo cáo khác với chỉ số xác suất.</div>
                    )}
                  </div>
                )}

                <div className="rec-list">
                  {recs.map((r,i)=>(<RecommendationCard key={i} r={r} />))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
