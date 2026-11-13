import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// How often the live camera sends a frame for recognition (milliseconds).
// Recognition runs on the CPU and takes roughly half a second per image,
// so values below about 1000 will not make results appear any faster.
const LIVE_INTERVAL_MS = 1500;

// Today's date as YYYY-MM-DD in the browser's local time.
function todayString() {
  return new Date().toLocaleDateString("en-CA");
}

// "2026-11-12T09:05:00" -> "09:05:00"
function timeOf(iso) {
  return iso ? iso.slice(11, 19) : "—";
}

const GALLERY_OPTIONS = [
  { id: "registered", label: "Registered people" },
  { id: "lfw", label: "Dataset (LFW)" },
  { id: "both", label: "Both" },
];

function GalleryToggle({ value, onChange }) {
  return (
    <div className="mode-toggle" role="radiogroup" aria-label="Gallery to compare against">
      {GALLERY_OPTIONS.map((option) => (
        <button
          key={option.id}
          type="button"
          role="radio"
          aria-checked={value === option.id}
          className={`mode-option ${value === option.id ? "active" : ""}`}
          onClick={() => onChange(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API}${path}`, options);
  } catch {
    throw new Error(
      "Cannot connect to the backend. Make sure FastAPI is running on port 8000. " +
        "If the API status shows connected, the backend may have hit an error on this request: look for a traceback in the backend window."
    );
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail || data.message || `Request failed (${response.status})`
    );
  }

  return data;
}

function Icon({ children }) {
  return <span className="nav-icon" aria-hidden="true">{children}</span>;
}

function App() {
  const [page, setPage] = useState("dashboard");
  const [backend, setBackend] = useState("checking");
  const [health, setHealth] = useState(null);
  const [people, setPeople] = useState([]);
  const [loadingPeople, setLoadingPeople] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);

  const [recognitionFile, setRecognitionFile] = useState(null);
  const [recognitionPreview, setRecognitionPreview] = useState("");
  const [threshold, setThreshold] = useState(0.45);
  const [galleryMode, setGalleryMode] = useState("both");
  const [recognitionResult, setRecognitionResult] = useState(null);

  const [personName, setPersonName] = useState("");
  const [registrationFile, setRegistrationFile] = useState(null);
  const [registrationPreview, setRegistrationPreview] = useState("");
  const [history, setHistory] = useState([]);

  const recognizeInput = useRef(null);
  const registerInput = useRef(null);

  // ---- Live camera -------------------------------------------------------
  const [liveOn, setLiveOn] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [liveFrame, setLiveFrame] = useState(null); // {w, h} of the analysed frame
  const [liveError, setLiveError] = useState("");

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  // `id` changes on every start so an old loop can never keep running
  // after the camera was stopped and started again.
  const loopRef = useRef({ running: false, id: 0, timer: null });
  const [markAttendance, setMarkAttendance] = useState(true);
  const settingsRef = useRef({ threshold, galleryMode, markAttendance });

  useEffect(() => {
    settingsRef.current = { threshold, galleryMode, markAttendance };
  }, [threshold, galleryMode, markAttendance]);

  const stopCamera = useCallback(() => {
    loopRef.current.running = false;
    clearTimeout(loopRef.current.timer);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setLiveOn(false);
    setLiveResult(null);
    setLiveFrame(null);
  }, []);

  async function runLiveLoop() {
    const state = loopRef.current;
    const runId = ++state.id;
    const alive = () => state.running && state.id === runId;

    while (alive()) {
      const startedAt = Date.now();
      const video = videoRef.current;

      if (video && video.videoWidth > 0) {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

          const blob = await new Promise((resolve) =>
            canvas.toBlob(resolve, "image/jpeg", 0.85)
          );

          if (!alive()) break;
          if (!blob) throw new Error("Could not capture a frame from the camera.");

          const { threshold: t, galleryMode: m, markAttendance: mark } = settingsRef.current;
          const form = new FormData();
          form.append("file", blob, "frame.jpg");

          // Attendance is only ever marked for registered people, so it is
          // skipped when the dataset-only gallery is selected.
          const markFlag = mark && m !== "lfw";

          const data = await request(
            `/compare-galleries?threshold=${encodeURIComponent(t)}&mode=${encodeURIComponent(m)}&mark_attendance=${markFlag}`,
            { method: "POST", body: form }
          );

          if (!alive()) break;
          setLiveResult(data);
          setLiveFrame({ w: canvas.width, h: canvas.height });
          setLiveError("");
        } catch (err) {
          if (!alive()) break;
          setLiveError(err.message);
        }
      }

      // Wait for the rest of the interval. The next frame is only sent after
      // the previous request finished, so requests never pile up.
      const wait = Math.max(0, LIVE_INTERVAL_MS - (Date.now() - startedAt));
      await new Promise((resolve) => {
        state.timer = setTimeout(resolve, wait);
      });
    }
  }

  async function startCamera() {
    setLiveError("");
    setLiveResult(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setLiveError(
        "This browser cannot use the camera. Use a recent Chrome, Edge or Firefox on localhost or HTTPS."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      video.srcObject = stream;
      await video.play();

      setLiveOn(true);
      loopRef.current.running = true;
      runLiveLoop();
    } catch (err) {
      stopCamera();
      setLiveError(
        err.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser and try again."
          : err.name === "NotFoundError"
            ? "No camera was found on this computer."
            : err.name === "NotReadableError"
              ? "The camera is being used by another application."
              : `Could not start the camera: ${err.message}`
      );
    }
  }

  // Turn the camera off when leaving the Live camera page or closing the app.
  useEffect(() => {
    if (page !== "live") stopCamera();
  }, [page, stopCamera]);

  useEffect(() => stopCamera, [stopCamera]);

  // ---- Attendance --------------------------------------------------------
  const [attendanceDay, setAttendanceDay] = useState(todayString);
  const [attendance, setAttendance] = useState(null);
  const [attendanceLoading, setAttendanceLoading] = useState(false);

  const loadAttendance = useCallback(async (day) => {
    setAttendanceLoading(true);

    try {
      const data = await request(`/attendance?day=${encodeURIComponent(day)}`);
      setAttendance(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setAttendanceLoading(false);
    }
  }, []);

  useEffect(() => {
    if (page !== "attendance" || !attendanceDay) return undefined;

    loadAttendance(attendanceDay);

    // Keep today's list fresh while the page is open.
    if (attendanceDay !== todayString()) return undefined;
    const timer = setInterval(() => loadAttendance(attendanceDay), 10000);
    return () => clearInterval(timer);
  }, [page, attendanceDay, loadAttendance]);

  const checkBackend = useCallback(async () => {
    setBackend("checking");

    try {
      const data = await request("/health");
      setHealth(data);
      setBackend("online");
    } catch (err) {
      setHealth(null);
      setBackend("offline");
      setError(err.message);
    }
  }, []);

  const loadPeople = useCallback(async () => {
    setLoadingPeople(true);

    try {
      const data = await request("/registered");
      setPeople(Array.isArray(data) ? data : data.identities || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingPeople(false);
    }
  }, []);

  useEffect(() => {
    checkBackend();
    loadPeople();
  }, [checkBackend, loadPeople]);

  function chooseImage(file, mode) {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("The image must be smaller than 10 MB.");
      return;
    }

    setError("");
    setNotice("");

    const preview = URL.createObjectURL(file);

    if (mode === "recognize") {
      if (recognitionPreview) {
        URL.revokeObjectURL(recognitionPreview);
      }
      setRecognitionFile(file);
      setRecognitionPreview(preview);
      setRecognitionResult(null);
      // Allow the same file to be selected again.
      if (recognizeInput.current) {
        recognizeInput.current.value = "";
      }
    }

    else {
      if (registrationPreview) URL.revokeObjectURL(registrationPreview);
      setRegistrationFile(file);
      setRegistrationPreview(preview);
    }
  }
  
  function startNewRecognition() {
    if (recognitionPreview) {
      URL.revokeObjectURL(recognitionPreview);
    }

    setRecognitionFile(null);
    setRecognitionPreview("");
    setRecognitionResult(null);
    setError("");
    setNotice("");

    if (recognizeInput.current) {
      recognizeInput.current.value = "";
    }
  }

  function keepImageAndTryAgain() {
    // Keep the selected image and clear the previous result.
    setRecognitionResult(null);
    setError("");
    setNotice("");
  }


  async function recognizeFace(event) {
    event.preventDefault();
    if (!recognitionFile) {
      setError("Please upload an image first.");
      return;
    }

    setBusy(true);
    setError("");
    setNotice("");
    setRecognitionResult(null);

    try {
      const form = new FormData();
      form.append("file", recognitionFile);

      const data = await request(
        `/compare-galleries?threshold=${encodeURIComponent(threshold)}&mode=${encodeURIComponent(galleryMode)}`,
        { method: "POST", body: form }
      );

      setRecognitionResult(data);
      setHistory((previous) => [
        {
          id: Date.now(),
          time: new Date().toLocaleString(),
          filename: recognitionFile.name,
          mode: galleryMode,
          data,
        },
        ...previous,
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function registerPerson(event) {
    event.preventDefault();

    const name = personName.trim();
    if (!name || !registrationFile) {
      setError("Enter a person's name and choose an image.");
      return;
    }

    setBusy(true);
    setError("");
    setNotice("");

    try {
      const form = new FormData();
      form.append("file", registrationFile);

      const data = await request(
        `/register?name=${encodeURIComponent(name)}`,
        { method: "POST", body: form }
      );

      setNotice(data.message || `${name} was registered successfully.`);
      setPersonName("");
      setRegistrationFile(null);

      if (registrationPreview) URL.revokeObjectURL(registrationPreview);
      setRegistrationPreview("");

      if (registerInput.current) registerInput.current.value = "";

      await loadPeople();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function deletePerson(person) {
    const id = person.id || person.person_id;
    if (!id) {
      setError("This record has no person ID, so it cannot be deleted.");
      return;
    }

    if (!window.confirm(`Remove ${person.name || "this person"} from the registry?`)) {
      return;
    }

    setError("");
    setNotice("");

    try {
      const data = await request(
        `/registered/${encodeURIComponent(id)}`,
        { method: "DELETE" }
      );
      setNotice(data.message || "Person removed from the registry.");
      await loadPeople();
    } catch (err) {
      setError(err.message);
    }
  }

  const navItems = [
    { id: "dashboard", icon: "▦", label: "Overview" },
    { id: "recognize", icon: "◎", label: "Face recognition" },
    { id: "live", icon: "◉", label: "Live camera" },
    { id: "attendance", icon: "▤", label: "Webcam Activity" },
    { id: "register", icon: "＋", label: "Register a person" },
    { id: "people", icon: "♙", label: "Registered people" },
    { id: "history", icon: "◷", label: "Recent activity" },
  ];

  const resultStatus = recognitionResult?.status;
  const resultIsKnown = resultStatus === "recognized";
  const resultIsUnknown = resultStatus === "unknown";
  const resultHasNoFace = resultStatus === "no_face";

  const liveStatus = liveResult?.status;
  const liveKnown = liveStatus === "recognized";
  const liveBox =
    liveOn && liveFrame && Array.isArray(liveResult?.bbox) && liveResult.bbox.length === 4
      ? {
          left: `${(liveResult.bbox[0] / liveFrame.w) * 100}%`,
          top: `${(liveResult.bbox[1] / liveFrame.h) * 100}%`,
          width: `${((liveResult.bbox[2] - liveResult.bbox[0]) / liveFrame.w) * 100}%`,
          height: `${((liveResult.bbox[3] - liveResult.bbox[1]) / liveFrame.h) * 100}%`,
        }
      : null;

  function goTo(id) {
    setPage(id);
    setError("");
    setNotice("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">F</div>
          <div>
            <div className="brand-name">Face<span>Detect</span></div>
            <div className="brand-subtitle">ARCFACE RECOGNITION</div>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>
        <nav className="navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => goTo(item.id)}
            >
              <Icon>{item.icon}</Icon>
              <span>{item.label}</span>
              {item.id === "people" && people.length > 0 && (
                <span className="nav-count">{people.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="model-card">
            <div className="model-icon">✳</div>
            <div>
              <strong>ArcFace engine</strong>
              <span>InsightFace · buffalo_l</span>
            </div>
          </div>
          <div className="sidebar-footer">LOCAL WORKSPACE · V1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span>{" "}
            {navItems.find((item) => item.id === page)?.label}
          </div>
          <div className="topbar-right">
            <span className={`status-dot ${backend}`} />
            <span className="backend-label">
              API {backend === "online" ? "connected" : backend === "checking" ? "checking" : "offline"}
            </span>
            <button className="icon-button" title="Refresh backend status" onClick={checkBackend}>
              ↻
            </button>
          </div>
        </header>

        <div className="page-content">
          {error && (
            <div className="alert alert-error">
              <span>!</span><div>{error}</div>
              <button onClick={() => setError("")}>×</button>
            </div>
          )}

          {notice && (
            <div className="alert alert-success">
              <span>✓</span><div>{notice}</div>
              <button onClick={() => setNotice("")}>×</button>
            </div>
          )}

          {page === "dashboard" && (
            <>
              <section className="page-heading">
                <div>
                  <div className="eyebrow">YOUR WORKSPACE</div>
                  <h1>Recognition overview</h1>
                  <p>Manage identities and recognize faces with your local ArcFace engine.</p>
                </div>
                <button className="primary-button" onClick={() => goTo("recognize")}>
                  <span>＋</span> New recognition
                </button>
              </section>

              <section className="stats-grid">
                <div className="stat-card">
                  <div className="stat-top"><span>Registered people</span><span className="stat-icon purple">♙</span></div>
                  <div className="stat-number">{people.length}</div>
                  <div className="stat-note">In your local registry</div>
                </div>
                <div className="stat-card">
                  <div className="stat-top"><span>Recognition checks</span><span className="stat-icon blue">◎</span></div>
                  <div className="stat-number">{history.length}</div>
                  <div className="stat-note">This browser session</div>
                </div>
                <div className="stat-card">
                  <div className="stat-top"><span>API status</span><span className="stat-icon green">⌁</span></div>
                  <div className="stat-number status-number">{backend === "online" ? "Online" : backend === "checking" ? "Checking" : "Offline"}</div>
                  <div className="stat-note">{health?.model || "ArcFace recognition API"}</div>
                </div>
              </section>

              <section className="content-grid">
                <div className="panel welcome-panel">
                  <div className="panel-heading">
                    <div><div className="eyebrow">GET STARTED</div><h2>What would you like to do?</h2></div>
                  </div>
                  <div className="action-list">
                    <button className="action-row" onClick={() => goTo("recognize")}>
                      <div className="action-icon purple">◎</div>
                      <div className="action-copy"><strong>Recognize a face</strong><span>Upload a photo and identify a face.</span></div>
                      <span className="action-arrow">→</span>
                    </button>
                    <button className="action-row" onClick={() => goTo("register")}>
                      <div className="action-icon blue">＋</div>
                      <div className="action-copy"><strong>Register a person</strong><span>Add a person to your local registry.</span></div>
                      <span className="action-arrow">→</span>
                    </button>
                    <button className="action-row" onClick={() => goTo("people")}>
                      <div className="action-icon green">♙</div>
                      <div className="action-copy"><strong>Manage identities</strong><span>View or remove registered people.</span></div>
                      <span className="action-arrow">→</span>
                    </button>
                  </div>
                </div>

                <div className="panel system-panel">
                  <div className="panel-heading"><div><div className="eyebrow">SYSTEM</div><h2>Engine status</h2></div><span className={`pill ${backend}`}>{backend}</span></div>
                  <div className="system-row"><span>API endpoint</span><strong>127.0.0.1:8000</strong></div>
                  <div className="system-row"><span>Recognition model</span><strong>{health?.model || "buffalo_l"}</strong></div>
                  <div className="system-row"><span>Execution provider</span><strong>{health?.provider || "CPU"}</strong></div>
                  <div className="system-row"><span>Dataset gallery</span><strong>{health?.gallery_size ?? "—"}</strong></div>
                  <button className="secondary-button full-button" onClick={checkBackend}>↻ Refresh status</button>
                </div>
              </section>

              <section className="panel activity-panel">
                <div className="panel-heading">
                  <div><div className="eyebrow">LATEST</div><h2>Recent recognition</h2></div>
                  <button className="text-button" onClick={() => goTo("history")}>View activity →</button>
                </div>
                {history.length === 0 ? (
                  <div className="empty-state compact-empty"><div className="empty-icon">◷</div><strong>No recognition checks yet</strong><span>Your latest checks will appear here.</span></div>
                ) : (
                  <div className="table-wrap"><table><thead><tr><th>Image</th><th>Result</th><th>Similarity</th><th>Time</th></tr></thead><tbody>
                    {history.slice(0, 5).map((item) => <tr key={item.id}><td>{item.filename}</td><td>{item.data.identity || item.data.status || "Complete"}</td><td>{typeof item.data.similarity === "number" ? item.data.similarity.toFixed(4) : "—"}</td><td>{item.time}</td></tr>)}
                  </tbody></table></div>
                )}
              </section>
            </>
          )}

          {page === "recognize" && (
            <>
              <section className="page-heading">
                <div><div className="eyebrow">ARCFACE ENGINE</div><h1>Face recognition</h1><p>Upload an image to compare its detected face against your recognition galleries.</p></div>
              </section>
              <section className="recognition-layout">
                <form className="panel form-panel" onSubmit={recognizeFace}>
                  <div className="panel-heading"><div><h2>Input image</h2><p>Use a clear, well-lit photo with a visible face.</p></div></div>
                  <input ref={recognizeInput} className="hidden-input" type="file" accept="image/*" onChange={(e) => chooseImage(e.target.files?.[0], "recognize")} />
                  <button type="button" className={`upload-zone ${recognitionPreview ? "has-preview" : ""}`} onClick={() => recognizeInput.current?.click()}>
                    {recognitionPreview ? <img src={recognitionPreview} alt="Selected recognition input" /> : <><span className="upload-icon">↑</span><strong>Choose an image</strong><span>Click to browse your computer</span><small>JPG, PNG, WEBP · Maximum 10 MB</small></>}
                  </button>
                  {recognitionFile && <div className="file-selected">✓ {recognitionFile.name}</div>}
                  <div className="field-group">
                    <div className="field-label"><label>Compare against</label></div>
                    <div className="mode-toggle" role="radiogroup" aria-label="Gallery to compare against">
                      {[
                        { id: "registered", label: "Registered people" },
                        { id: "lfw", label: "Dataset (LFW)" },
                        { id: "both", label: "Both" },
                      ].map((option) => (
                        <button
                          key={option.id}
                          type="button"
                          role="radio"
                          aria-checked={galleryMode === option.id}
                          className={`mode-option ${galleryMode === option.id ? "active" : ""}`}
                          onClick={() => { setGalleryMode(option.id); setRecognitionResult(null); }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="field-group">
                    <div className="field-label"><label htmlFor="threshold">Similarity threshold</label><strong>{Number(threshold).toFixed(2)}</strong></div>
                    <input id="threshold" type="range" min="0.2" max="0.8" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
                    <div className="range-labels"><span>0.20 · More permissive</span><span>0.80 · More strict</span></div>
                    <p className="field-help">A higher threshold makes acceptance stricter. The best value should be calibrated for your use case.</p>
                  </div>
                  <button className="primary-button full-button" disabled={busy || backend !== "online" || !recognitionFile}>
                    {busy ? "Processing image…" : "◎  Recognize face"}
                  </button>
                </form>

                <div className="panel result-panel">
                  <div className="panel-heading"><div><h2>Recognition result</h2><p>The API response will appear here.</p></div></div>
                  {!recognitionResult ? (
                    <div className="empty-state result-empty"><div className="empty-icon">◎</div><strong>Waiting for an image</strong><span>Upload a photo and run recognition to see the result.</span></div>
                  ) : (
                    <div className="result-content">
                      <div className={`result-banner ${resultIsKnown ? "known" : resultIsUnknown ? "unknown" : "neutral"}`}>
                        <span className="result-symbol">{resultIsKnown ? "✓" : resultIsUnknown ? "?" : "!"}</span>
                        <div><strong>{resultIsKnown ? "Face recognized" : resultIsUnknown ? "Unknown identity" : resultHasNoFace ? "No face detected" : "Recognition complete"}</strong><span>{resultIsKnown ? "A matching identity was returned by the API." : resultIsUnknown ? "No accepted identity was returned." : "Review the details below."}</span></div>
                      </div>
                      <div className="result-identity"><span>IDENTITY</span><strong>{recognitionResult.identity || "Not identified"}</strong></div>
                      <div className="result-metrics">
                        <div><span>Similarity</span><strong>{typeof recognitionResult.similarity === "number" ? recognitionResult.similarity.toFixed(4) : "—"}</strong></div>
                        <div><span>Detection score</span><strong>{typeof recognitionResult.detection_score === "number" ? recognitionResult.detection_score.toFixed(4) : "—"}</strong></div>
                        <div><span>Faces detected</span><strong>{recognitionResult.face_count ?? "—"}</strong></div>
                        <div><span>Match source</span><strong>{recognitionResult.source || "—"}</strong></div>
                      </div>
                      {Array.isArray(recognitionResult.candidates) && recognitionResult.candidates.length > 0 && (
                        <div className="candidates">
                          <div className="candidates-title">Top matches ({recognitionResult.mode})</div>
                          <div className="table-wrap"><table><thead><tr><th>#</th><th>Identity</th><th>Source</th><th>Similarity</th><th>Result</th></tr></thead><tbody>
                            {recognitionResult.candidates.map((c, i) => (
                              <tr key={`${c.source}-${c.identity}-${i}`}>
                                <td>{i + 1}</td>
                                <td>{c.identity}</td>
                                <td><span className={`pill ${c.source === "registered" ? "online" : "checking"}`}>{c.source === "lfw" ? "dataset" : c.source}</span></td>
                                <td>{c.similarity.toFixed(4)}</td>
                                <td>{c.accepted ? "Accepted" : "Below threshold"}</td>
                              </tr>
                            ))}
                          </tbody></table></div>
                        </div>
                      )}
                      <details className="raw-details"><summary>View full API response</summary><pre>{JSON.stringify(recognitionResult, null, 2)}</pre></details>
                      <div className="result-actions" style={{ display: "flex", gap: "10px", flexWrap: "wrap", marginTop: "18px", }} > <button type="button" className="secondary-button" onClick={keepImageAndTryAgain} > ↻ Keep Image &amp; Try Again </button> <button type="button" className="primary-button" onClick={startNewRecognition} > ← Recognize Another Image </button> </div>
                    </div>
                  )}
                </div>
              </section>
            </>
          )}

          {page === "live" && (
            <>
              <section className="page-heading">
                <div>
                  <div className="eyebrow">ARCFACE ENGINE</div>
                  <h1>Live camera</h1>
                  <p>Recognize faces from your webcam. A frame is checked about every {LIVE_INTERVAL_MS / 1000} seconds.</p>
                </div>
              </section>

              <section className="recognition-layout">
                <div className="panel form-panel">
                  <div className="panel-heading">
                    <div><h2>Camera</h2><p>Only the largest face in view is identified.</p></div>
                    {liveOn && <span className="pill online">● Live</span>}
                  </div>

                  <div className="live-stage">
                    <video
                      ref={videoRef}
                      className="live-video"
                      playsInline
                      muted
                      style={{ display: liveOn ? "block" : "none" }}
                    />
                    {!liveOn && (
                      <div className="live-placeholder">
                        <span className="upload-icon">◉</span>
                        <strong>Camera is off</strong>
                        <span>Press Start camera and allow access when your browser asks.</span>
                      </div>
                    )}
                    {liveBox && (
                      <div className={`live-box ${liveKnown ? "known" : "unknown"}`} style={liveBox}>
                        <span className="live-label">
                          {liveKnown ? liveResult.identity : "Unknown"}
                          {typeof liveResult.similarity === "number" && ` · ${liveResult.similarity.toFixed(2)}`}
                        </span>
                      </div>
                    )}
                  </div>

                  <div className="field-group">
                    <div className="field-label"><label>Compare against</label></div>
                    <GalleryToggle value={galleryMode} onChange={setGalleryMode} />
                  </div>

                  <div className="field-group">
                    <div className="field-label"><label htmlFor="live-threshold">Similarity threshold</label><strong>{Number(threshold).toFixed(2)}</strong></div>
                    <input id="live-threshold" type="range" min="0.2" max="0.8" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
                    <div className="range-labels"><span>0.20 · More permissive</span><span>0.80 · More strict</span></div>
                  </div>

                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={markAttendance}
                      onChange={(e) => setMarkAttendance(e.target.checked)}
                    />
                    <span>
                      <strong>Mark attendance</strong>
                      <small>Registered people recognised with similarity 0.50 or higher are marked present for today. Not used in dataset-only mode.</small>
                    </span>
                  </label>

                  {liveOn ? (
                    <button type="button" className="secondary-button full-button" onClick={stopCamera}>■  Stop camera</button>
                  ) : (
                    <button type="button" className="primary-button full-button" disabled={backend !== "online"} onClick={startCamera}>
                      ◉  Start camera
                    </button>
                  )}

                  <div className="privacy-note">
                    <span>⌂</span>
                    <div>
                      <strong>Camera privacy</strong>
                      <p>Frames are sent only to your local backend and are not saved. The camera stops when you leave this page. Only point it at people who have agreed.</p>
                    </div>
                  </div>
                </div>

                <div className="panel result-panel">
                  <div className="panel-heading"><div><h2>Live result</h2><p>Updates after each checked frame.</p></div></div>

                  {liveError && (
                    <div className="alert alert-error">
                      <span>!</span><div>{liveError}</div>
                      <button onClick={() => setLiveError("")}>×</button>
                    </div>
                  )}

                  {!liveResult ? (
                    <div className="empty-state result-empty">
                      <div className="empty-icon">◉</div>
                      <strong>{liveOn ? "Looking for a face…" : "Camera is off"}</strong>
                      <span>{liveOn ? "The first result appears in a moment." : "Start the camera to see live results."}</span>
                    </div>
                  ) : (
                    <div className="result-content">
                      <div className={`result-banner ${liveKnown ? "known" : liveStatus === "unknown" ? "unknown" : "neutral"}`}>
                        <span className="result-symbol">{liveKnown ? "✓" : liveStatus === "unknown" ? "?" : "!"}</span>
                        <div>
                          <strong>{liveKnown ? "Face recognized" : liveStatus === "unknown" ? "Unknown identity" : liveStatus === "no_face" ? "No face in view" : "Checking…"}</strong>
                          <span>{liveStatus === "no_face" ? "Move into the camera view." : liveKnown ? "A matching identity was found." : "No accepted identity."}</span>
                        </div>
                      </div>
                      <div className="result-identity"><span>IDENTITY</span><strong>{liveResult.identity || "Not identified"}</strong></div>
                      {markAttendance && liveResult.attendance && liveResult.attendance.status !== "not_applicable" && (
                        <div className={`attendance-note ${liveResult.attendance.status}`}>
                          {liveResult.attendance.status === "checked_in" &&
                            `✓ Attendance marked · first seen ${timeOf(liveResult.attendance.first_seen)}`}
                          {liveResult.attendance.status === "updated" &&
                            `✓ Present today · first ${timeOf(liveResult.attendance.first_seen)} · last ${timeOf(liveResult.attendance.last_seen)}`}
                          {liveResult.attendance.status === "low_confidence" &&
                            `Attendance not marked: similarity is below ${Number(liveResult.attendance.min_similarity).toFixed(2)}`}
                          {liveResult.attendance.status === "error" &&
                            (liveResult.attendance.message || "Attendance could not be saved. Check the backend window.")}
                        </div>
                      )}
                      <div className="result-metrics">
                        <div><span>Similarity</span><strong>{typeof liveResult.similarity === "number" ? liveResult.similarity.toFixed(4) : "—"}</strong></div>
                        <div><span>Detection score</span><strong>{typeof liveResult.detection_score === "number" ? liveResult.detection_score.toFixed(4) : "—"}</strong></div>
                        <div><span>Faces in view</span><strong>{liveResult.face_count ?? "—"}</strong></div>
                        <div><span>Match source</span><strong>{liveResult.source === "lfw" ? "dataset" : liveResult.source || "—"}</strong></div>
                      </div>
                      {Array.isArray(liveResult.candidates) && liveResult.candidates.length > 0 && (
                        <div className="candidates">
                          <div className="candidates-title">Top matches ({liveResult.mode})</div>
                          <div className="table-wrap"><table><thead><tr><th>#</th><th>Identity</th><th>Source</th><th>Similarity</th></tr></thead><tbody>
                            {liveResult.candidates.slice(0, 3).map((c, i) => (
                              <tr key={`${c.source}-${c.identity}-${i}`}>
                                <td>{i + 1}</td>
                                <td>{c.identity}</td>
                                <td><span className={`pill ${c.source === "registered" ? "online" : "checking"}`}>{c.source === "lfw" ? "dataset" : c.source}</span></td>
                                <td>{c.similarity.toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody></table></div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </section>
            </>
          )}

          {page === "attendance" && (
            <>
              <section className="page-heading">
                <div>
                  <div className="eyebrow">DAILY RECORD</div>
                  <h1>Webcam Activity</h1>
                  <p>First and last time each registered person was seen on the Live camera page.</p>
                </div>
              </section>

              <section className="stats-grid">
                <div className="stat-card">
                  <div className="stat-top"><span>Present</span><span className="stat-icon green">✓</span></div>
                  <div className="stat-number">{attendance?.present.length ?? "—"}</div>
                  <div className="stat-note">{attendanceDay}</div>
                </div>
                <div className="stat-card">
                  <div className="stat-top"><span>Absent</span><span className="stat-icon purple">○</span></div>
                  <div className="stat-number">{attendance?.absent.length ?? "—"}</div>
                  <div className="stat-note">Registered, not seen that day</div>
                </div>
                <div className="stat-card">
                  <div className="stat-top"><span>Registered</span><span className="stat-icon blue">♙</span></div>
                  <div className="stat-number">{attendance?.registered_count ?? "—"}</div>
                  <div className="stat-note">Registered by that date</div>
                </div>
              </section>

              <section className="panel stack">
                <div className="panel-heading">
                  <div><h2>Day</h2><p>Pick a date to see who was present and absent.</p></div>
                </div>
                <div className="attendance-toolbar">
                  <input
                    type="date"
                    className="text-input"
                    value={attendanceDay}
                    max={todayString()}
                    onChange={(e) => e.target.value && setAttendanceDay(e.target.value)}
                    aria-label="Attendance date"
                  />
                  <button type="button" className="secondary-button" onClick={() => setAttendanceDay(todayString())}>Today</button>
                  <button type="button" className="secondary-button" onClick={() => loadAttendance(attendanceDay)} disabled={attendanceLoading}>
                    {attendanceLoading ? "Refreshing…" : "↻ Refresh"}
                  </button>
                  <a
                    className="primary-button link-button"
                    href={`${API}/attendance/export?start=${encodeURIComponent(attendanceDay)}&end=${encodeURIComponent(attendanceDay)}`}
                  >
                    ↓ CSV for this day
                  </a>
                  <a className="secondary-button link-button" href={`${API}/attendance/export`}>↓ CSV for all days</a>
                </div>
              </section>

              <section className="panel people-panel stack">
                <div className="panel-heading">
                  <div><h2>Present</h2><p>{attendance ? `${attendance.present.length} ${attendance.present.length === 1 ? "person" : "people"} on ${attendance.day}` : "Loading…"}</p></div>
                </div>
                {!attendance || attendance.present.length === 0 ? (
                  <div className="empty-state compact-empty">
                    <div className="empty-icon">▤</div>
                    <strong>No one marked present</strong>
                    <span>Start the Live camera with “Mark attendance” on, and registered people are recorded here.</span>
                  </div>
                ) : (
                  <div className="table-wrap"><table><thead><tr><th>Person</th><th>First seen</th><th>Last seen</th><th>Visits</th></tr></thead><tbody>
                    {attendance.present.map((row) => (
                      <tr key={row.person_id}>
                        <td><div className="person-cell"><span className="person-avatar">{(row.name || "?").charAt(0).toUpperCase()}</span><strong>{row.name}</strong></div></td>
                        <td>{timeOf(row.first_seen)}</td>
                        <td>{timeOf(row.last_seen)}</td>
                        <td>{row.visits}</td>
                      </tr>
                    ))}
                  </tbody></table></div>
                )}
              </section>

              <section className="panel people-panel stack">
                <div className="panel-heading">
                  <div><h2>Absent</h2><p>Registered people not seen on this day.</p></div>
                </div>
                {!attendance || attendance.absent.length === 0 ? (
                  <div className="empty-state compact-empty">
                    <strong>{attendance ? "Nobody is absent" : "Loading…"}</strong>
                  </div>
                ) : (
                  <div className="table-wrap"><table><thead><tr><th>Person</th><th>Status</th></tr></thead><tbody>
                    {attendance.absent.map((row) => (
                      <tr key={row.person_id}>
                        <td><div className="person-cell"><span className="person-avatar">{(row.name || "?").charAt(0).toUpperCase()}</span><strong>{row.name}</strong></div></td>
                        <td><span className="pill offline">absent</span></td>
                      </tr>
                    ))}
                  </tbody></table></div>
                )}
              </section>
            </>
          )}

          {page === "register" && (
            <>
              <section className="page-heading"><div><div className="eyebrow">IDENTITY MANAGEMENT</div><h1>Register a person</h1><p>Create a local identity using a name and a photo containing a clearly visible face.</p></div></section>
              <section className="registration-layout">
                <form className="panel form-panel" onSubmit={registerPerson}>
                  <div className="panel-heading"><div><h2>Person details</h2><p>The image will be processed by your local API.</p></div></div>
                  <div className="field-group"><label className="field-label" htmlFor="person-name">Full name</label><input id="person-name" className="text-input" value={personName} onChange={(e) => setPersonName(e.target.value)} placeholder="e.g. Alex Morgan" maxLength={100} required /></div>
                  <div className="field-group"><label className="field-label">Face image</label><input ref={registerInput} className="hidden-input" type="file" accept="image/*" onChange={(e) => chooseImage(e.target.files?.[0], "register")} /><button type="button" className={`upload-zone register-upload ${registrationPreview ? "has-preview" : ""}`} onClick={() => registerInput.current?.click()}>{registrationPreview ? <img src={registrationPreview} alt="Selected registration input" /> : <><span className="upload-icon">↑</span><strong>Upload a face photo</strong><span>Click to browse your computer</span><small>JPG, PNG, WEBP · Maximum 10 MB</small></>}</button>{registrationFile && <div className="file-selected">✓ {registrationFile.name}</div>}</div>
                  <div className="privacy-note"><span>⌂</span><div><strong>Local storage</strong><p>Registration is sent to your configured local backend, which stores the face embedding and image in its registry.</p></div></div>
                  <button className="primary-button full-button" disabled={busy || backend !== "online" || !personName.trim() || !registrationFile}>{busy ? "Registering…" : "＋  Register person"}</button>
                </form>
                <div className="panel guidance-panel"><div className="guidance-art">◎</div><h2>For best results</h2><ul><li>Use a clear image with one visible face.</li><li>Prefer good lighting and a forward-facing view.</li><li>Avoid heavy blur, filters, or extreme angles.</li><li>Only register images you have permission to use.</li></ul><div className="guidance-divider" /><div className="guidance-stat"><span>People registered</span><strong>{people.length}</strong></div><button className="text-button" onClick={() => goTo("people")}>View registered people →</button></div>
              </section>
            </>
          )}

          {page === "people" && (
            <>
              <section className="page-heading"><div><div className="eyebrow">IDENTITY MANAGEMENT</div><h1>Registered people</h1><p>Manage the identities saved in your local registry.</p></div><button className="primary-button" onClick={() => goTo("register")}>＋ Register person</button></section>
              <section className="panel people-panel">
                <div className="panel-heading"><div><h2>Identity registry</h2><p>{people.length} registered {people.length === 1 ? "person" : "people"}</p></div><button className="secondary-button" onClick={loadPeople} disabled={loadingPeople}>{loadingPeople ? "Refreshing…" : "↻ Refresh"}</button></div>
                {loadingPeople && people.length === 0 ? <div className="empty-state"><span>Loading registry…</span></div> : people.length === 0 ? <div className="empty-state"><div className="empty-icon">♙</div><strong>No people registered yet</strong><span>Register a person to begin building your local identity registry.</span><button className="primary-button" onClick={() => goTo("register")}>Register first person</button></div> : (
                  <div className="table-wrap"><table><thead><tr><th>Person</th><th>Registry ID</th><th>Action</th></tr></thead><tbody>{people.map((person, index) => { const id = person.id || person.person_id; return <tr key={id || `${person.name}-${index}`}><td><div className="person-cell"><span className="person-avatar">{(person.name || "?").charAt(0).toUpperCase()}</span><strong>{person.name || "Unnamed person"}</strong></div></td><td><code>{id || "—"}</code></td><td><button className="delete-button" disabled={!id} onClick={() => deletePerson(person)}>Remove</button></td></tr>; })}</tbody></table></div>
                )}
              </section>
            </>
          )}

          {page === "history" && (
            <>
              <section className="page-heading"><div><div className="eyebrow">SESSION LOG</div><h1>Recent activity</h1><p>Recognition results recorded in this browser session.</p></div></section>
              <section className="panel activity-panel">
                <div className="panel-heading"><div><h2>Recognition history</h2><p>{history.length} {history.length === 1 ? "check" : "checks"} in this session</p></div><button className="secondary-button" onClick={() => setHistory([])} disabled={history.length === 0}>Clear history</button></div>
                {history.length === 0 ? <div className="empty-state"><div className="empty-icon">◷</div><strong>No activity yet</strong><span>Your recognition attempts will appear here.</span><button className="primary-button" onClick={() => goTo("recognize")}>Start recognition</button></div> : <div className="table-wrap"><table><thead><tr><th>Image</th><th>Status</th><th>Identity</th><th>Similarity</th><th>Time</th></tr></thead><tbody>{history.map((item) => <tr key={item.id}><td>{item.filename}</td><td><span className={`pill ${item.data.status === "recognized" ? "online" : "offline"}`}>{item.data.status || "complete"}</span></td><td>{item.data.identity || "—"}</td><td>{typeof item.data.similarity === "number" ? item.data.similarity.toFixed(4) : "—"}</td><td>{item.time}</td></tr>)}</tbody></table></div>}
              </section>
            </>
          )}

          <footer className="page-footer"><span>FacialIQ · Local facial-recognition workspace</span><span>ArcFace · InsightFace · FastAPI</span></footer>
        </div>
      </main>
    </div>
  );
}

export default App;