import { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  FileSearch,
  Fingerprint,
  LoaderCircle,
  Shield,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";

const EMPTY_POSTING = {
  title: "",
  company_profile: "",
  description: "",
  requirements: "",
  benefits: "",
  telecommuting: false,
  has_company_logo: false,
  has_questions: false,
  employment_type: "Full-time",
  required_experience: "Entry level",
  required_education: "Unspecified",
};

const PRESETS = {
  "Equipment payment request": {
    title: "Immediate Data Entry Clerk - Work From Home",
    company_profile: "",
    description: "Urgent recruitment! No prior experience required. Candidates receive an initial check to purchase home office equipment from our vendor via wire transfer. Start immediately and contact our hiring manager through Telegram.",
    requirements: "Must have an active bank account to deposit the equipment check. Immediate start without interview.",
    benefits: "Earn $2,000 to $4,000 weekly. Flexible hours, fully remote.",
    telecommuting: true,
    has_company_logo: false,
    has_questions: false,
    employment_type: "Part-time",
    required_experience: "Entry level",
    required_education: "High School or equivalent",
  },
  "Established engineering role": {
    title: "Senior Python Backend Engineer",
    company_profile: "CloudScale Systems is an enterprise cloud observability company founded in 2016. We provide telemetry and analytics for global technology enterprises.",
    description: "Design distributed microservices using Python, Redis, and Kafka. Collaborate with DevOps engineers to ensure high availability and scale our data ingestion service.",
    requirements: "5+ years of software engineering experience. Strong Python, PostgreSQL, and AWS or GCP skills. Bachelor's degree or equivalent experience.",
    benefits: "Competitive salary, 401(k) matching, medical and dental coverage, parental leave, and paid time off.",
    telecommuting: true,
    has_company_logo: true,
    has_questions: true,
    employment_type: "Full-time",
    required_experience: "Mid-Senior level",
    required_education: "Bachelor's Degree",
  },
  "Personal email contact": {
    title: "Executive Personal Assistant - Flexible Schedule",
    company_profile: "Private confidential client looking for a trusted assistant.",
    description: "Busy executive looking for a dependable personal assistant to handle errands and confidential tasks. Send your resume along with a copy of your driver's license to executive.manager2024@gmail.com for prompt review.",
    requirements: "Trustworthy, prompt communicator. Must possess a clean driving record.",
    benefits: "Weekly allowance plus performance bonuses.",
    telecommuting: false,
    has_company_logo: false,
    has_questions: false,
    employment_type: "Contract",
    required_experience: "Not Applicable",
    required_education: "Unspecified",
  },
};

const FIELD_GROUPS = [
  { key: "company_profile", label: "Company profile", placeholder: "Who is hiring? Add the company background or leave blank if it was not provided.", rows: 3 },
  { key: "description", label: "Job description", placeholder: "Paste the responsibilities, contact instructions, and any payment details...", rows: 6, required: true },
  { key: "requirements", label: "Requirements", placeholder: "Experience, qualifications, or required documents", rows: 3 },
  { key: "benefits", label: "Benefits & compensation", placeholder: "Salary, bonuses, equipment, or other benefits", rows: 3 },
];

const OPTIONS = {
  employment_type: ["Full-time", "Part-time", "Contract", "Temporary", "Other", "Unknown"],
  required_experience: ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable", "Unknown"],
  required_education: ["Bachelor's Degree", "Master's Degree", "High School or equivalent", "Associate Degree", "Unspecified", "Unknown"],
};

function App() {
  const [posting, setPosting] = useState(EMPTY_POSTING);
  const [model, setModel] = useState("Linear SVM");
  const [lowThreshold, setLowThreshold] = useState(0.25);
  const [highThreshold, setHighThreshold] = useState(0.65);
  const [models, setModels] = useState([]);
  const [apiReady, setApiReady] = useState(null);
  const [assessment, setAssessment] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then(async (response) => {
        if (!response.ok) throw new Error("The prediction service is unavailable.");
        return response.json();
      })
      .then((health) => {
        setModels(health.models);
        setApiReady(health.ready);
        if (health.models.includes("Linear SVM")) setModel("Linear SVM");
      })
      .catch(() => setApiReady(false));
  }, []);

  function updateField(key, value) {
    setPosting((current) => ({ ...current, [key]: value }));
  }

  function loadPreset(name) {
    if (name === "blank") {
      setPosting(EMPTY_POSTING);
    } else {
      setPosting({ ...PRESETS[name] });
    }
    setAssessment(null);
    setError("");
  }

  async function runScreening(event) {
    event.preventDefault();
    if (!posting.title.trim() && !posting.description.trim()) {
      setError("Add a job title or description to run the screening.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...posting,
          model_name: model,
          low_threshold: Number(lowThreshold),
          high_threshold: Number(highThreshold),
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Screening failed. Please try again.");
      setAssessment(result);
    } catch (requestError) {
      setError(requestError.message || "Could not connect to the prediction service.");
    } finally {
      setBusy(false);
    }
  }

  const highRisk = assessment?.badge_style === "danger";
  const moderateRisk = assessment?.badge_style === "warning";
  const riskClass = highRisk ? "high" : moderateRisk ? "moderate" : "low";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Risk Guard home">
          <span className="brand-mark"><Shield size={20} strokeWidth={2.2} /></span>
          <span>risk<span className="brand-light">guard</span></span>
        </a>
        <div className="topbar-right">
          <span className={`service-status ${apiReady ? "online" : apiReady === false ? "offline" : "checking"}`}>
            <span className="status-dot" />{apiReady ? "Model service online" : apiReady === false ? "Service unavailable" : "Connecting"}
          </span>
          <span className="topbar-divider" />
          <span className="topbar-caption">Recruitment integrity tools</span>
          <CircleHelp size={18} className="help-icon" aria-label="Screening is a preliminary aid, not employer verification" />
        </div>
      </header>

      <main id="top" className="workspace">
        <section className="intro-row">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> TRUST & SAFETY <span className="eyebrow-dot">/</span> POSTING REVIEW</div>
            <h1>Job posting <em>screening</em></h1>
            <p className="intro-copy">Assess a listing for recruitment fraud signals, then verify the details with the employer directly.</p>
          </div>
          <div className="dataset-stamp"><Fingerprint size={17} /><span>EMSCAD-trained models</span><span className="stamp-dot">·</span><span>4 classifiers</span></div>
        </section>

        <div className="main-grid">
          <section className="form-panel" aria-labelledby="posting-heading">
            <div className="panel-heading">
              <div className="section-index">01 <span>/</span> INPUT</div>
              <div className="sample-select-wrap">
                <Sparkles size={14} />
                <select aria-label="Load a sample posting" defaultValue="blank" onChange={(event) => loadPreset(event.target.value)}>
                  <option value="blank">Load sample</option>
                  {Object.keys(PRESETS).map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
                <ChevronDown size={14} className="select-chevron" />
              </div>
            </div>

            <form onSubmit={runScreening}>
              <div className="title-field">
                <label htmlFor="title">Position title <span className="required-mark">*</span></label>
                <input id="title" value={posting.title} onChange={(event) => updateField("title", event.target.value)} placeholder="e.g. Operations Coordinator" maxLength={500} />
              </div>

              {FIELD_GROUPS.map((field) => (
                <div className="form-field" key={field.key}>
                  <label htmlFor={field.key}>{field.label}{field.required && <span className="required-mark"> *</span>}</label>
                  <textarea id={field.key} rows={field.rows} value={posting[field.key]} onChange={(event) => updateField(field.key, event.target.value)} placeholder={field.placeholder} maxLength={field.key === "description" ? 20000 : 10000} />
                </div>
              ))}

              <div className="metadata-heading">
                <div><SlidersHorizontal size={15} /><span>Posting details</span></div>
                <span className="optional-label">STRUCTURED SIGNALS</span>
              </div>

              <div className="toggle-grid">
                <Toggle label="Remote / telecommuting" checked={posting.telecommuting} onChange={(value) => updateField("telecommuting", value)} />
                <Toggle label="Company logo shown" checked={posting.has_company_logo} onChange={(value) => updateField("has_company_logo", value)} />
                <Toggle label="Screening questions" checked={posting.has_questions} onChange={(value) => updateField("has_questions", value)} />
              </div>

              <div className="select-grid">
                {Object.entries(OPTIONS).map(([key, options]) => (
                  <label className="select-field" key={key}>
                    <span>{key === "employment_type" ? "Employment type" : key === "required_experience" ? "Experience" : "Education"}</span>
                    <div className="native-select"><select value={posting[key]} onChange={(event) => updateField(key, event.target.value)}>{options.map((option) => <option key={option}>{option}</option>)}</select><ChevronDown size={14} /></div>
                  </label>
                ))}
              </div>

              <label className="model-field">
                <span>Screening classifier</span>
                <div className="native-select"><select value={model} onChange={(event) => setModel(event.target.value)}>{models.map((availableModel) => <option key={availableModel}>{availableModel}</option>)}</select><ChevronDown size={14} /></div>
              </label>

              <div className="threshold-panel">
                <div className="threshold-title"><span>Risk thresholds</span><span className="threshold-note">Adjust classification sensitivity</span></div>
                <Threshold label="Moderate alert" value={lowThreshold} min={0.15} max={0.45} step={0.05} tone="amber" onChange={setLowThreshold} />
                <Threshold label="High risk" value={highThreshold} min={0.5} max={0.85} step={0.05} tone="red" onChange={setHighThreshold} />
              </div>

              {error && <div className="error-message" role="alert"><AlertTriangle size={16} />{error}</div>}
              {apiReady === false && <div className="service-message" role="status">Start the API with <code>uvicorn api:app --reload</code> to enable screening.</div>}

              <button className="screen-button" type="submit" disabled={busy || !apiReady}>
                {busy ? <LoaderCircle size={17} className="spin" /> : <ShieldCheck size={17} />}
                {busy ? "Screening posting..." : "Run risk screening"}
                {!busy && <ArrowUpRight size={16} className="button-arrow" />}
              </button>
              <p className="form-footnote"><Shield size={12} /> Listing details are processed locally by this application.</p>
            </form>
          </section>

          <section className={`results-panel ${assessment ? "has-result" : ""}`} aria-live="polite" aria-labelledby="results-heading">
            <div className="panel-heading result-heading">
              <div className="section-index">02 <span>/</span> ASSESSMENT</div>
              {assessment && <span className="model-tag">{assessment.model_used}</span>}
            </div>

            {!assessment ? (
              <div className="empty-state">
                <div className="empty-art"><div className="art-ring ring-one" /><div className="art-ring ring-two" /><div className="art-core"><FileSearch size={28} strokeWidth={1.4} /></div><span className="art-tick tick-a" /><span className="art-tick tick-b" /><span className="art-tick tick-c" /></div>
                <span className="empty-kicker">AWAITING LISTING</span>
                <h2 id="results-heading">Your assessment<br />will appear here.</h2>
                <p>Complete the posting details and run a screening to see the model score and any detected warning signals.</p>
                <div className="empty-divider" />
                <div className="pipeline-row"><span className="pipeline-number">01</span><span>Text & metadata analysis</span><span className="pipeline-arrow">→</span><span className="pipeline-number">02</span><span>Risk signals</span></div>
              </div>
            ) : (
              <div className="assessment-content">
                <div className={`score-card ${riskClass}`}>
                  <div className="score-card-top"><span>ESTIMATED FRAUD PROBABILITY</span><span className="score-model">{assessment.is_calibrated ? "CALIBRATED" : "MODEL SCORE"}</span></div>
                  <div className="score-row">
                    <div className="score-number">{(assessment.ml_probability * 100).toFixed(1)}<small>%</small></div>
                    <div className={`risk-badge ${riskClass}`}>{highRisk ? <ShieldAlert size={14} /> : moderateRisk ? <AlertTriangle size={14} /> : <ShieldCheck size={14} />}{assessment.risk_level}</div>
                  </div>
                  <div className="score-track"><span style={{ width: `${Math.max(0, Math.min(100, assessment.ml_probability * 100))}%` }} /></div>
                  <div className="score-scale"><span>0% · Lower signal</span><span>{Math.round(highThreshold * 100)}% · High risk</span><span>100%</span></div>
                  <p className="score-summary">{assessment.summary}</p>
                </div>

                <div className="signals-section">
                  <div className="subsection-heading"><div><span className="subsection-marker">A</span><h3>Heuristic signals</h3></div><span className="signal-count">{assessment.flags_count} FOUND</span></div>
                  {assessment.heuristic_flags.length ? (
                    <div className="flag-list">
                      {assessment.heuristic_flags.map((flag, index) => <article className={`flag-item severity-${flag.severity.toLowerCase()}`} key={`${flag.category}-${index}`}>
                        <div className="flag-icon">{flag.severity === "HIGH" ? <AlertTriangle size={16} /> : <Fingerprint size={16} />}</div>
                        <div className="flag-copy"><div className="flag-title-row"><h4>{flag.category}</h4><span>{flag.severity}</span></div><p>{flag.description}</p><div className="evidence-list">{flag.evidence.map((item) => <span key={item}>{item}</span>)}</div></div>
                      </article>)}
                    </div>
                  ) : <div className="no-flags"><CheckCircle2 size={17} /><span>No heuristic warning signals detected.</span></div>}
                </div>

                <div className="verification-section">
                  <div className="subsection-heading"><div><span className="subsection-marker green">B</span><h3>Applicant checklist</h3></div><span className="checklist-label">BEFORE YOU APPLY</span></div>
                  <ul>{assessment.recommendations.map((recommendation) => <li key={recommendation}><span><Check size={12} /></span>{recommendation}</li>)}</ul>
                </div>

                <div className="disclaimer"><AlertTriangle size={15} /><p>{assessment.disclaimer}</p></div>
              </div>
            )}
          </section>
        </div>

        <footer className="page-footer"><span>RISK GUARD <span className="footer-dot">·</span> PRELIMINARY SCREENING ONLY</span><span>Historical patterns are not proof of fraud. Always verify independently.</span></footer>
      </main>
    </div>
  );
}

function Toggle({ label, checked, onChange }) {
  return <label className={`toggle-control ${checked ? "checked" : ""}`}><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span className="toggle-box">{checked && <Check size={11} />}</span><span>{label}</span></label>;
}

function Threshold({ label, value, min, max, step, tone, onChange }) {
  return <label className="threshold-row"><span className={`threshold-dot ${tone}`} /><span className="threshold-label">{label}</span><input aria-label={`${label} threshold`} type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} /><output>{Math.round(value * 100)}%</output></label>;
}

export default App;