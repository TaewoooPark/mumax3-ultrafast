import { useMemo, useRef, useState, type KeyboardEvent, type UIEvent } from "react";
import { GlassButton, Glyph } from "./components/GlassPrimitives";

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

const sampleMetrics = [
  { label: "Step", value: "—", unit: "iterations" },
  { label: "Time", value: "—", unit: "s" },
  { label: "Δt", value: "—", unit: "s" },
  { label: "Max torque", value: "—", unit: "T" },
  { label: "Error / step", value: "—", unit: "relative" },
  { label: "Progress", value: "0", unit: "%" },
];

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

export default function App() {
  const [script, setScript] = useState(starterScript);
  const [folder] = useState("Choose a working folder");
  const [fileName, setFileName] = useState("relaxation.mx3");

  return (
    <main className="app-shell">
      <div className="title-drag" data-tauri-drag-region>
        <span className="brand-mark">m³</span>
        <strong>mumax³ ultrafast</strong>
        <span>Desktop preview</span>
      </div>

      <header className="topbar">
        <div className="project-heading">
          <span className="eyebrow">APPLE SILICON MICROMAGNETICS</span>
          <h1>Simulation workspace</h1>
        </div>
        <div className="status-chip glass"><i /><span>Engine not built</span></div>
        <div className="topbar-actions">
          <GlassButton disabled>Open full viewer ↗</GlassButton>
        </div>
      </header>

      <section className="workspace-grid">
        <div className="visual-column">
          <section className="preview-panel glass">
            <div className="panel-heading">
              <div><span className="eyebrow">LIVE MAGNETIZATION</span><strong>m · vector field</strong></div>
              <span className="live-state"><i /> Waiting</span>
            </div>
            <div className="preview-empty">
              <div className="field-orb"><span /><span /><span /></div>
              <strong>Your simulation will appear here</strong>
              <p>Build the Metal engine, then run the script from the editor.</p>
            </div>
          </section>

          <section className="metrics-grid">
            {sampleMetrics.map((metric) => (
              <article className="metric-card glass" key={metric.label}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
                <small>{metric.unit}</small>
              </article>
            ))}
          </section>

          <section className="console-panel glass">
            <div className="panel-heading compact">
              <div><span className="eyebrow">ENGINE OUTPUT</span><strong>Runtime log</strong></div>
              <span className="mono">idle</span>
            </div>
            <pre><span className="muted">// Build and run output will stream here.</span></pre>
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
              <button className="path-field" type="button">
                <Glyph>⌁</Glyph><b>{folder}</b><em>Choose…</em>
              </button>
            </label>
            <label>
              <span>Script file</span>
              <div className="filename-field">
                <Glyph>⌘</Glyph>
                <input value={fileName} onChange={(event) => setFileName(event.target.value)} spellCheck={false} />
              </div>
            </label>
          </div>

          <div className="editor-toolbar">
            <GlassButton>Open</GlassButton>
            <GlassButton>Save</GlassButton>
            <span />
            <GlassButton>Build engine</GlassButton>
            <button className="primary-button" type="button">Run simulation <b>▶</b></button>
          </div>

          <CodeEditor value={script} onChange={setScript} />

          <footer className="editor-footer">
            <span>{script.split("\n").length} lines</span>
            <span>UTF-8</span>
            <span className="save-state">Not saved</span>
          </footer>
        </section>
      </section>
    </main>
  );
}
