import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type UIEvent } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { desktop } from "./api";
import { GlassButton, Glyph } from "./components/GlassPrimitives";
import type { GuiValues, RuntimePhase, RuntimeSnapshot } from "./types";

const starterScript = `// Permalloy relaxation example
SetGridSize(256, 128, 1)
SetCellSize(4e-9, 4e-9, 4e-9)

Msat = 800e3
Aex = 13e-12
alpha = 0.02

m = Uniform(1, 0.08, 0)
TableAutoSave(5e-12)
AutoSave(m, 50e-12)

Run(1e-9)
Save(m)
`;

const idleRuntime: RuntimeSnapshot = {
  phase: "idle",
  processAlive: false,
  viewerAvailable: false,
  guiUrl: "",
  renderUrl: "",
  outputDirectory: "",
  scriptPath: "",
  logs: [],
  lastError: "",
  exitCode: null,
};

const phaseLabels: Record<RuntimePhase, string> = {
  idle: "Engine idle",
  building: "Building Metal engine",
  ready: "Engine ready",
  starting: "Starting simulation",
  running: "Simulation running",
  completed: "Script complete · interactive viewer ready",
  failed: "Simulation failed",
  stopped: "Simulation stopped",
};

function CodeEditor({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const gutterRef = useRef<HTMLPreElement>(null);
  const lineNumbers = useMemo(
    () => Array.from({ length: value.split("\n").length }, (_, index) => index + 1).join("\n"),
    [value],
  );

  const syncScroll = (event: UIEvent<HTMLTextAreaElement>) => {
    if (gutterRef.current) gutterRef.current.scrollTop = event.currentTarget.scrollTop;
  };

  const insertTab = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Tab") return;
    event.preventDefault();
    const target = event.currentTarget;
    const next = `${value.slice(0, target.selectionStart)}  ${value.slice(target.selectionEnd)}`;
    const cursor = target.selectionStart + 2;
    onChange(next);
    requestAnimationFrame(() => target.setSelectionRange(cursor, cursor));
  };

  return (
    <div className="code-editor">
      <pre ref={gutterRef} className="line-numbers" aria-hidden="true">{lineNumbers}</pre>
      <textarea
        aria-label="mumax3 script editor"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onScroll={syncScroll}
        onKeyDown={insertTab}
        spellCheck={false}
      />
    </div>
  );
}

function metricValue(values: GuiValues, key: string) {
  const value = values[key];
  if (value === undefined || value === null || value === "") return "—";
  return String(value);
}

export default function App() {
  const [script, setScript] = useState(starterScript);
  const [workingDirectory, setWorkingDirectory] = useState("");
  const [fileName, setFileName] = useState("relaxation.mx3");
  const [engineBuilt, setEngineBuilt] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeSnapshot>(idleRuntime);
  const [guiValues, setGuiValues] = useState<GuiValues>({});
  const [busyAction, setBusyAction] = useState("");
  const [message, setMessage] = useState("Connecting to the desktop runtime…");
  const [dirty, setDirty] = useState(true);
  const [previewReady, setPreviewReady] = useState(false);
  const [frame, setFrame] = useState(0);
  const consoleRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    void desktop.info()
      .then((info) => {
        setWorkingDirectory(info.defaultWorkingDirectory);
        setEngineBuilt(info.engineBuilt);
        setMessage(info.engineBuilt ? "Metal engine ready" : "Build the Metal engine to begin");
      })
      .catch((error) => setMessage(`Desktop runtime unavailable: ${String(error)}`));
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void desktop.runtimeSnapshot()
        .then((next) => {
          setRuntime(next);
          if (next.viewerAvailable) setFrame((value) => value + 1);
          if (next.lastError) setMessage(next.lastError);
        })
        .catch(() => undefined);
    }, 550);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!runtime.guiUrl || !runtime.viewerAvailable) return;
    const timer = window.setInterval(() => {
      void desktop.guiSnapshot()
        .then((values) => setGuiValues((current) => ({ ...current, ...values })))
        .catch(() => undefined);
    }, 650);
    return () => window.clearInterval(timer);
  }, [runtime.guiUrl, runtime.viewerAvailable]);

  useEffect(() => {
    if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [runtime.logs.length]);

  const chooseFolder = async () => {
    const selected = await open({ directory: true, multiple: false, title: "Choose a simulation folder" });
    if (typeof selected === "string") setWorkingDirectory(selected);
  };

  const openScript = async () => {
    const selected = await open({
      directory: false,
      multiple: false,
      title: "Open a mumax3 script",
      filters: [{ name: "mumax3 script", extensions: ["mx3"] }],
    });
    if (typeof selected !== "string") return;
    setBusyAction("open");
    try {
      const document = await desktop.loadScript(selected);
      setScript(document.contents);
      setWorkingDirectory(document.workingDirectory);
      setFileName(document.fileName);
      setDirty(false);
      setMessage(`Opened ${document.fileName}`);
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusyAction("");
    }
  };

  const saveScript = async () => {
    if (!workingDirectory) {
      setMessage("Choose a working folder first.");
      return false;
    }
    setBusyAction("save");
    try {
      const document = await desktop.saveScript(workingDirectory, fileName, script);
      setFileName(document.fileName);
      setDirty(false);
      setMessage(`Saved ${document.path}`);
      return true;
    } catch (error) {
      setMessage(String(error));
      return false;
    } finally {
      setBusyAction("");
    }
  };

  const buildEngine = async () => {
    setBusyAction("build");
    setMessage("Compiling the native Metal engine…");
    try {
      const result = await desktop.buildEngine();
      setEngineBuilt(true);
      setMessage(`Engine built at ${result.binaryPath}`);
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusyAction("");
    }
  };

  const runSimulation = async () => {
    if (!workingDirectory || !fileName.trim()) {
      setMessage("Choose a working folder and script name first.");
      return;
    }
    setBusyAction("run");
    setPreviewReady(false);
    setGuiValues({});
    setMessage("Saving the script and starting mumax3-ultrafast…");
    try {
      const next = await desktop.startSimulation({ workingDirectory, fileName, contents: script });
      setRuntime(next);
      setDirty(false);
      setMessage(`Running ${fileName}`);
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusyAction("");
    }
  };

  const stopSimulation = async () => {
    setBusyAction("stop");
    try {
      setRuntime(await desktop.stopSimulation());
      setMessage("Simulation stopped.");
    } catch (error) {
      setMessage(String(error));
    } finally {
      setBusyAction("");
    }
  };

  const metrics = [
    { label: "Step", value: metricValue(guiValues, "nsteps"), unit: "iterations" },
    { label: "Time", value: metricValue(guiValues, "time"), unit: "s" },
    { label: "Δt", value: metricValue(guiValues, "dt"), unit: "s" },
    { label: "Max torque", value: metricValue(guiValues, "maxtorque"), unit: "T" },
    { label: "Error / step", value: metricValue(guiValues, "lasterr"), unit: "relative" },
    { label: "Progress", value: metricValue(guiValues, "progress") === "—" ? "0" : metricValue(guiValues, "progress"), unit: "%" },
  ];

  const actionBusy = Boolean(busyAction);
  const phase = busyAction === "build" ? "building" : runtime.phase;
  const running = runtime.processAlive && runtime.phase !== "completed";
  const renderSource = runtime.renderUrl ? `${runtime.renderUrl}?frame=${frame}` : "";

  return (
    <main className="app-shell">
      <div className="title-drag" data-tauri-drag-region>
        <span className="brand-mark">m³</span>
        <strong>mumax³ ultrafast</strong>
        <span>Desktop</span>
      </div>

      <header className="topbar">
        <div className="project-heading">
          <span className="eyebrow">APPLE SILICON MICROMAGNETICS</span>
          <h1>Simulation workspace</h1>
        </div>
        <div className={`status-chip glass phase-${phase}`}><i /><span>{busyAction === "build" ? phaseLabels.building : phaseLabels[runtime.phase]}</span></div>
        <div className="topbar-actions">
          <GlassButton disabled={!runtime.viewerAvailable} onClick={() => void openUrl(runtime.guiUrl)}>Open full viewer ↗</GlassButton>
        </div>
      </header>

      <section className="workspace-grid">
        <div className="visual-column">
          <section className="preview-panel glass">
            <div className="panel-heading">
              <div><span className="eyebrow">LIVE MAGNETIZATION</span><strong>m · vector field</strong></div>
              <span className={`live-state ${runtime.viewerAvailable ? "active" : ""}`}><i /> {runtime.viewerAvailable ? runtime.phase === "completed" ? "Complete" : "Live" : "Waiting"}</span>
            </div>
            <div className={`preview-stage ${previewReady ? "ready" : ""}`}>
              {renderSource && <img src={renderSource} alt="Live mumax3 magnetization" onLoad={() => setPreviewReady(true)} onError={() => setPreviewReady(false)} />}
              {!previewReady && (
                <div className="preview-empty">
                  <div className="field-orb"><span /><span /><span /></div>
                  <strong>{runtime.viewerAvailable ? "Preparing the first field frame" : "Your simulation will appear here"}</strong>
                  <p>{runtime.viewerAvailable ? "The Metal renderer is warming up." : "Build the Metal engine, then run the script from the editor."}</p>
                </div>
              )}
            </div>
          </section>

          <section className="metrics-grid">
            {metrics.map((metric) => (
              <article className="metric-card glass" key={metric.label}>
                <span>{metric.label}</span>
                <strong title={metric.value}>{metric.value}</strong>
                <small>{metric.unit}</small>
              </article>
            ))}
          </section>

          <section className="console-panel glass">
            <div className="panel-heading compact">
              <div><span className="eyebrow">ENGINE OUTPUT</span><strong>Runtime log</strong></div>
              <span className="mono">{runtime.outputDirectory || runtime.phase}</span>
            </div>
            <pre ref={consoleRef}>{runtime.logs.length ? runtime.logs.join("\n") : <span className="muted">// Build and run output will stream here.</span>}</pre>
          </section>
        </div>

        <section className="editor-panel glass">
          <div className="editor-heading">
            <div><span className="eyebrow">MX3 EDITOR</span><strong>{fileName || "Untitled.mx3"}</strong></div>
            <span className="language-badge">mumax³</span>
          </div>

          <div className="path-stack">
            <label>
              <span>Working folder</span>
              <button className="path-field" type="button" onClick={() => void chooseFolder()}>
                <Glyph>⌁</Glyph><b title={workingDirectory}>{workingDirectory || "Choose a working folder"}</b><em>Choose…</em>
              </button>
            </label>
            <label>
              <span>Script file</span>
              <div className="filename-field">
                <Glyph>⌘</Glyph>
                <input value={fileName} onChange={(event) => { setFileName(event.target.value); setDirty(true); }} spellCheck={false} />
              </div>
            </label>
          </div>

          <div className="editor-toolbar">
            <GlassButton disabled={actionBusy || running} onClick={() => void openScript()}>{busyAction === "open" ? "Opening…" : "Open"}</GlassButton>
            <GlassButton disabled={actionBusy || running} onClick={() => void saveScript()}>{busyAction === "save" ? "Saving…" : "Save"}</GlassButton>
            <span />
            <GlassButton disabled={actionBusy || runtime.processAlive} onClick={() => void buildEngine()}>{busyAction === "build" ? "Building…" : engineBuilt ? "Rebuild engine" : "Build engine"}</GlassButton>
            {runtime.processAlive ? (
              <button className="stop-button" disabled={actionBusy} type="button" onClick={() => void stopSimulation()}>{busyAction === "stop" ? "Stopping…" : "Stop"} <b>■</b></button>
            ) : (
              <button className="primary-button" disabled={actionBusy || !engineBuilt || !workingDirectory} type="button" onClick={() => void runSimulation()}>{busyAction === "run" ? "Starting…" : "Run simulation"} <b>▶</b></button>
            )}
          </div>

          <CodeEditor value={script} onChange={(value) => { setScript(value); setDirty(true); }} />

          <footer className="editor-footer">
            <span className="editor-message" title={message}>{message}</span>
            <span>{script.split("\n").length} lines</span>
            <span>UTF-8</span>
            <span className="save-state">{dirty ? "Not saved" : "Saved"}</span>
          </footer>
        </section>
      </section>
    </main>
  );
}
