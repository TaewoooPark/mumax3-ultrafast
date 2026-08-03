use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, VecDeque},
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{Manager, State, WindowEvent};

const MAX_LOG_LINES: usize = 400;
const MAX_RESULT_GLYPHS: usize = 12_000;

#[derive(Clone)]
struct DesktopRuntime {
    repository_root: PathBuf,
    binary_path: PathBuf,
    state: Arc<Mutex<ProcessState>>,
}

struct ProcessState {
    child: Option<Child>,
    phase: String,
    gui_url: String,
    render_url: String,
    output_directory: String,
    script_path: String,
    logs: VecDeque<String>,
    last_error: String,
    exit_code: Option<i32>,
}

impl Default for ProcessState {
    fn default() -> Self {
        Self {
            child: None,
            phase: "idle".to_string(),
            gui_url: String::new(),
            render_url: String::new(),
            output_directory: String::new(),
            script_path: String::new(),
            logs: VecDeque::new(),
            last_error: String::new(),
            exit_code: None,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AppInfo {
    repository_root: String,
    default_working_directory: String,
    binary_path: String,
    engine_built: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BuildResult {
    binary_path: String,
    output: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ScriptDocument {
    path: String,
    working_directory: String,
    file_name: String,
    contents: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StartRequest {
    working_directory: String,
    file_name: String,
    contents: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeSnapshot {
    phase: String,
    process_alive: bool,
    viewer_available: bool,
    gui_url: String,
    render_url: String,
    output_directory: String,
    script_path: String,
    logs: Vec<String>,
    last_error: String,
    exit_code: Option<i32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResultFrameInfo {
    file_name: String,
    quantity: String,
    size_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResultSet {
    output_directory: String,
    frames: Vec<ResultFrameInfo>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VectorFrame {
    file_name: String,
    title: String,
    time_label: String,
    dimensions: [usize; 3],
    sample_step: [usize; 3],
    glyphs: Vec<f32>,
    magnitude_min: f32,
    magnitude_max: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct GuiCall {
    f: String,
    args: Vec<Value>,
}

fn find_repository_root() -> Result<PathBuf, String> {
    if let Some(configured) = env::var_os("MUMAX3_ULTRAFAST_HOME").map(PathBuf::from)
        && configured.join("go.mod").is_file()
        && configured.join("cmd/mumax3").is_dir()
    {
        return configured.canonicalize().map_err(|error| error.to_string());
    }

    if let Ok(current) = env::current_dir()
        && let Some(root) = current
            .ancestors()
            .find(|path| path.join("go.mod").is_file() && path.join("cmd/mumax3").is_dir())
    {
        return Ok(root.to_path_buf());
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .find(|path| path.join("go.mod").is_file() && path.join("cmd/mumax3").is_dir())
        .map(Path::to_path_buf)
        .ok_or_else(|| "Could not locate the mumax3-ultrafast repository.".to_string())
}

fn checked_working_directory(value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err("The working directory must be an absolute path.".to_string());
    }
    if !path.is_dir() {
        return Err(format!(
            "The working directory does not exist: {}",
            path.display()
        ));
    }
    path.canonicalize().map_err(|error| error.to_string())
}

fn checked_file_name(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("Enter a script file name.".to_string());
    }
    let path = Path::new(trimmed);
    if path.components().count() != 1
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(
            "The script name must be a file name without directory components.".to_string(),
        );
    }
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mx3"))
    {
        Ok(trimmed.to_string())
    } else {
        Ok(format!("{trimmed}.mx3"))
    }
}

fn save_script_document(
    working_directory: &str,
    file_name: &str,
    contents: &str,
) -> Result<ScriptDocument, String> {
    let directory = checked_working_directory(working_directory)?;
    let file_name = checked_file_name(file_name)?;
    let path = directory.join(&file_name);
    fs::write(&path, contents)
        .map_err(|error| format!("Could not save {}: {error}", path.display()))?;
    Ok(ScriptDocument {
        path: path.to_string_lossy().into_owned(),
        working_directory: directory.to_string_lossy().into_owned(),
        file_name,
        contents: contents.to_string(),
    })
}

fn push_log(shared: &Arc<Mutex<ProcessState>>, line: String) {
    if let Ok(mut state) = shared.lock() {
        if let Some(gui_url) = line.strip_prefix("//starting GUI at ")
            && let Ok(gui_url) = normalized_loopback_gui_url(gui_url.trim())
        {
            state.gui_url = gui_url;
            state.render_url = format!("{}/render/m", state.gui_url);
            state.phase = "running".to_string();
        }
        if line.contains("//entering interactive mode") {
            state.phase = "completed".to_string();
        }
        state.logs.push_back(line);
        while state.logs.len() > MAX_LOG_LINES {
            state.logs.pop_front();
        }
    }
}

fn mark_runtime_failed(runtime: &DesktopRuntime, message: &str) {
    if let Ok(mut state) = runtime.state.lock() {
        state.phase = "failed".to_string();
        state.last_error = message.to_string();
        state.logs.push_back(format!("[wrapper] {message}"));
        while state.logs.len() > MAX_LOG_LINES {
            state.logs.pop_front();
        }
    }
}

fn capture_output<R: Read + Send + 'static>(
    reader: R,
    shared: Arc<Mutex<ProcessState>>,
    prefix: &'static str,
) {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            match line {
                Ok(line) => push_log(&shared, format!("{prefix}{line}")),
                Err(error) => {
                    push_log(
                        &shared,
                        format!("[wrapper] Could not read engine output: {error}"),
                    );
                    break;
                }
            }
        }
    });
}

fn snapshot_locked(state: &mut ProcessState) -> RuntimeSnapshot {
    let exit = state
        .child
        .as_mut()
        .and_then(|child| child.try_wait().ok().flatten());
    if let Some(status) = exit {
        state.child.take();
        state.exit_code = status.code();
        state.gui_url.clear();
        state.render_url.clear();
        if !status.success() && state.phase != "stopped" {
            state.phase = "failed".to_string();
            state.last_error = format!("mumax3-ultrafast exited with {status}.");
        } else if state.phase != "stopped" {
            state.phase = "completed".to_string();
        }
    }
    let process_alive = state.child.is_some();
    RuntimeSnapshot {
        phase: state.phase.clone(),
        process_alive,
        viewer_available: process_alive && !state.gui_url.is_empty(),
        gui_url: state.gui_url.clone(),
        render_url: state.render_url.clone(),
        output_directory: state.output_directory.clone(),
        script_path: state.script_path.clone(),
        logs: state.logs.iter().cloned().collect(),
        last_error: state.last_error.clone(),
        exit_code: state.exit_code,
    }
}

fn snapshot(runtime: &DesktopRuntime) -> Result<RuntimeSnapshot, String> {
    let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
    Ok(snapshot_locked(&mut state))
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn normalized_loopback_gui_url(value: &str) -> Result<String, String> {
    let parsed = reqwest::Url::parse(value).map_err(|error| error.to_string())?;
    if parsed.scheme() != "http"
        || parsed.host_str() != Some("127.0.0.1")
        || parsed.port().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(
            "The viewer URL must be an origin on the local loopback interface.".to_string(),
        );
    }
    Ok(format!(
        "http://127.0.0.1:{}",
        parsed.port().expect("port was checked above")
    ))
}

fn find_go_executable() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(configured) = env::var_os("MUMAX3_GO") {
        candidates.push(PathBuf::from(configured));
    }
    if let Some(path) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&path).map(|directory| directory.join("go")));
    }
    if let Some(go_root) = env::var_os("GOROOT") {
        candidates.push(PathBuf::from(go_root).join("bin/go"));
    }
    candidates.extend(
        ["/opt/homebrew/bin/go", "/usr/local/bin/go", "/usr/bin/go"]
            .into_iter()
            .map(PathBuf::from),
    );
    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            "Could not find Go. Install Go or set MUMAX3_GO to the Go executable.".to_string()
        })
}

#[derive(Clone, Copy)]
enum OvfDataKind {
    Text,
    Binary4,
    Binary8,
}

struct GlyphCollector {
    dimensions: [usize; 3],
    sample_step: [usize; 3],
    value_dim: usize,
    component: usize,
    cell: usize,
    vector: [f32; 3],
    glyphs: Vec<f32>,
    magnitude_min: f32,
    magnitude_max: f32,
}

impl GlyphCollector {
    fn new(dimensions: [usize; 3], sample_step: [usize; 3], value_dim: usize) -> Self {
        Self {
            dimensions,
            sample_step,
            value_dim,
            component: 0,
            cell: 0,
            vector: [0.0; 3],
            glyphs: Vec::new(),
            magnitude_min: f32::INFINITY,
            magnitude_max: 0.0,
        }
    }

    fn push(&mut self, value: f32) {
        if self.component < 3 {
            self.vector[self.component] = value;
        }
        self.component += 1;
        if self.component != self.value_dim {
            return;
        }

        let nx = self.dimensions[0];
        let ny = self.dimensions[1];
        let x = self.cell % nx;
        let y = (self.cell / nx) % ny;
        let z = self.cell / (nx * ny);
        if x.is_multiple_of(self.sample_step[0])
            && y.is_multiple_of(self.sample_step[1])
            && z.is_multiple_of(self.sample_step[2])
        {
            let magnitude = self
                .vector
                .iter()
                .map(|component| component * component)
                .sum::<f32>()
                .sqrt();
            if magnitude > 1e-12 && magnitude.is_finite() {
                self.glyphs.extend([
                    x as f32,
                    y as f32,
                    z as f32,
                    self.vector[0],
                    self.vector[1],
                    self.vector[2],
                ]);
                self.magnitude_min = self.magnitude_min.min(magnitude);
                self.magnitude_max = self.magnitude_max.max(magnitude);
            }
        }
        self.cell += 1;
        self.component = 0;
        self.vector = [0.0; 3];
    }
}

fn sampled_steps(dimensions: [usize; 3]) -> [usize; 3] {
    let total = dimensions.iter().copied().product::<usize>();
    if total <= MAX_RESULT_GLYPHS {
        return [1; 3];
    }
    let active_dimensions = dimensions
        .iter()
        .filter(|dimension| **dimension > 1)
        .count()
        .max(1);
    let factor = ((total as f64 / MAX_RESULT_GLYPHS as f64)
        .powf(1.0 / active_dimensions as f64)
        .ceil() as usize)
        .max(1);
    let mut steps = dimensions.map(|dimension| if dimension > 1 { factor } else { 1 });
    let sampled_count = |steps: [usize; 3]| {
        dimensions
            .iter()
            .zip(steps)
            .map(|(dimension, step)| dimension.div_ceil(step))
            .product::<usize>()
    };
    while sampled_count(steps) > MAX_RESULT_GLYPHS {
        let axis = (0..3)
            .filter(|axis| dimensions[*axis] > 1)
            .max_by_key(|axis| dimensions[*axis].div_ceil(steps[*axis]))
            .unwrap_or(0);
        steps[axis] += 1;
    }
    steps
}

fn parse_header_value<T: std::str::FromStr>(
    header: &HashMap<String, String>,
    key: &str,
) -> Result<T, String> {
    header
        .get(key)
        .ok_or_else(|| format!("OVF header is missing {key}."))?
        .parse::<T>()
        .map_err(|_| format!("OVF header contains an invalid {key}."))
}

fn read_ovf_header<R: BufRead>(
    reader: &mut R,
) -> Result<(HashMap<String, String>, OvfDataKind), String> {
    let mut header = HashMap::new();
    loop {
        let mut line = Vec::new();
        if reader
            .read_until(b'\n', &mut line)
            .map_err(|error| error.to_string())?
            == 0
        {
            return Err("OVF file ended before its data section.".to_string());
        }
        let line = String::from_utf8_lossy(&line);
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        let data_kind = if lower.starts_with("# begin: data text") {
            Some(OvfDataKind::Text)
        } else if lower.starts_with("# begin: data binary 4") {
            Some(OvfDataKind::Binary4)
        } else if lower.starts_with("# begin: data binary 8") {
            Some(OvfDataKind::Binary8)
        } else {
            None
        };
        if let Some(data_kind) = data_kind {
            return Ok((header, data_kind));
        }
        if let Some(content) = trimmed.strip_prefix('#')
            && let Some((key, value)) = content.trim().split_once(':')
        {
            header.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
}

fn binary_is_little_endian<const N: usize>(
    bytes: [u8; N],
    little: f64,
    big: f64,
    expected: f64,
) -> Result<bool, String> {
    let tolerance = expected.abs() * 1e-9;
    if (little - expected).abs() <= tolerance {
        Ok(true)
    } else if (big - expected).abs() <= tolerance {
        Ok(false)
    } else {
        Err(format!("OVF binary check value is invalid: {bytes:?}"))
    }
}

fn parse_ovf_reader<R: BufRead>(reader: &mut R, file_name: String) -> Result<VectorFrame, String> {
    let (header, data_kind) = read_ovf_header(reader)?;
    let dimensions = [
        parse_header_value(&header, "xnodes")?,
        parse_header_value(&header, "ynodes")?,
        parse_header_value(&header, "znodes")?,
    ];
    if dimensions.contains(&0) {
        return Err("OVF dimensions must be greater than zero.".to_string());
    }
    let value_dim = parse_header_value::<usize>(&header, "valuedim")?;
    if value_dim != 3 {
        return Err(format!(
            "The 3D result viewer supports 3-component OVF vector fields; this file has valuedim {value_dim}."
        ));
    }
    let total_cells = dimensions
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| "OVF dimensions are too large.".to_string())?;
    let total_values = total_cells
        .checked_mul(value_dim)
        .ok_or_else(|| "OVF value count is too large.".to_string())?;
    let sample_step = sampled_steps(dimensions);
    let mut collector = GlyphCollector::new(dimensions, sample_step, value_dim);

    match data_kind {
        OvfDataKind::Text => {
            let mut line = String::new();
            while collector.cell < total_cells {
                line.clear();
                if reader
                    .read_line(&mut line)
                    .map_err(|error| error.to_string())?
                    == 0
                {
                    break;
                }
                if line.trim_start().starts_with('#') {
                    continue;
                }
                for token in line.split_whitespace() {
                    let value = token
                        .parse::<f32>()
                        .map_err(|_| format!("Invalid OVF text value: {token}"))?;
                    collector.push(value);
                    if collector.cell == total_cells {
                        break;
                    }
                }
            }
        }
        OvfDataKind::Binary4 => {
            let mut check = [0u8; 4];
            reader
                .read_exact(&mut check)
                .map_err(|error| error.to_string())?;
            let little_endian = binary_is_little_endian(
                check,
                f32::from_le_bytes(check) as f64,
                f32::from_be_bytes(check) as f64,
                1_234_567.0,
            )?;
            for _ in 0..total_values {
                let mut bytes = [0u8; 4];
                reader
                    .read_exact(&mut bytes)
                    .map_err(|error| error.to_string())?;
                collector.push(if little_endian {
                    f32::from_le_bytes(bytes)
                } else {
                    f32::from_be_bytes(bytes)
                });
            }
        }
        OvfDataKind::Binary8 => {
            let mut check = [0u8; 8];
            reader
                .read_exact(&mut check)
                .map_err(|error| error.to_string())?;
            let little_endian = binary_is_little_endian(
                check,
                f64::from_le_bytes(check),
                f64::from_be_bytes(check),
                123_456_789_012_345.0,
            )?;
            for _ in 0..total_values {
                let mut bytes = [0u8; 8];
                reader
                    .read_exact(&mut bytes)
                    .map_err(|error| error.to_string())?;
                collector.push(if little_endian {
                    f64::from_le_bytes(bytes) as f32
                } else {
                    f64::from_be_bytes(bytes) as f32
                });
            }
        }
    }
    if collector.cell != total_cells {
        return Err(format!(
            "OVF data ended after {} of {total_cells} cells.",
            collector.cell
        ));
    }
    if collector.magnitude_min == f32::INFINITY {
        collector.magnitude_min = 0.0;
    }
    Ok(VectorFrame {
        file_name,
        title: header.get("title").cloned().unwrap_or_default(),
        time_label: header.get("desc").cloned().unwrap_or_default(),
        dimensions,
        sample_step,
        glyphs: collector.glyphs,
        magnitude_min: collector.magnitude_min,
        magnitude_max: collector.magnitude_max,
    })
}

fn current_output_directory(runtime: &DesktopRuntime) -> Result<PathBuf, String> {
    let output_directory = runtime
        .state
        .lock()
        .map_err(|error| error.to_string())?
        .output_directory
        .clone();
    if output_directory.is_empty() {
        return Err("Run a simulation before opening its result viewer.".to_string());
    }
    let path = PathBuf::from(output_directory);
    if !path.is_dir() {
        return Err(format!(
            "The result folder does not exist: {}",
            path.display()
        ));
    }
    path.canonicalize().map_err(|error| error.to_string())
}

fn ovf_quantity(file_name: &str) -> String {
    let stem = file_name.strip_suffix(".ovf").unwrap_or(file_name);
    let quantity = stem.trim_end_matches(|character: char| character.is_ascii_digit());
    quantity.trim_end_matches(['_', '-']).to_string()
}

#[tauri::command]
fn app_info(runtime: State<'_, DesktopRuntime>) -> AppInfo {
    AppInfo {
        repository_root: runtime.repository_root.to_string_lossy().into_owned(),
        default_working_directory: String::new(),
        binary_path: runtime.binary_path.to_string_lossy().into_owned(),
        engine_built: runtime.binary_path.is_file(),
    }
}

#[tauri::command]
fn build_engine(runtime: State<'_, DesktopRuntime>) -> Result<BuildResult, String> {
    {
        let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
        if state.child.is_some() {
            return Err("Stop the active simulation before rebuilding the engine.".to_string());
        }
        state.phase = "building".to_string();
        state.last_error.clear();
        state
            .logs
            .push_back("[wrapper] Building the native Metal engine…".to_string());
    }

    if let Some(parent) = runtime.binary_path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        let message = format!("Could not prepare the engine build directory: {error}");
        mark_runtime_failed(&runtime, &message);
        return Err(message);
    }
    let go_executable = match find_go_executable() {
        Ok(executable) => executable,
        Err(message) => {
            mark_runtime_failed(&runtime, &message);
            return Err(message);
        }
    };
    let output = Command::new(go_executable)
        .args(["build", "-o"])
        .arg(&runtime.binary_path)
        .arg("./cmd/mumax3")
        .current_dir(&runtime.repository_root)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(error) => {
            let message = format!("Could not start the Go compiler: {error}");
            mark_runtime_failed(&runtime, &message);
            return Err(message);
        }
    };
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
    for line in combined.lines() {
        state.logs.push_back(format!("[build] {line}"));
    }
    if !output.status.success() {
        state.phase = "failed".to_string();
        state.last_error = format!("Engine build failed with {}.", output.status);
        return Err(format!("{}\n{}", state.last_error, combined.trim()));
    }
    state.phase = "ready".to_string();
    state.logs.push_back(format!(
        "[wrapper] Engine ready: {}",
        runtime.binary_path.display()
    ));
    Ok(BuildResult {
        binary_path: runtime.binary_path.to_string_lossy().into_owned(),
        output: combined,
    })
}

#[tauri::command]
fn save_script(
    working_directory: String,
    file_name: String,
    contents: String,
) -> Result<ScriptDocument, String> {
    save_script_document(&working_directory, &file_name, &contents)
}

#[tauri::command]
fn load_script(path: String) -> Result<ScriptDocument, String> {
    let path = PathBuf::from(path);
    if !path.is_absolute() || !path.is_file() {
        return Err("Choose an existing .mx3 file.".to_string());
    }
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_none_or(|extension| !extension.eq_ignore_ascii_case("mx3"))
    {
        return Err("Only .mx3 scripts can be opened.".to_string());
    }
    let canonical = path.canonicalize().map_err(|error| error.to_string())?;
    let contents = fs::read_to_string(&canonical)
        .map_err(|error| format!("Could not read {}: {error}", canonical.display()))?;
    let directory = canonical
        .parent()
        .ok_or_else(|| "The selected script has no parent directory.".to_string())?;
    let file_name = canonical
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "The selected script name is not valid UTF-8.".to_string())?;
    Ok(ScriptDocument {
        path: canonical.to_string_lossy().into_owned(),
        working_directory: directory.to_string_lossy().into_owned(),
        file_name: file_name.to_string(),
        contents,
    })
}

#[tauri::command]
fn start_simulation(
    request: StartRequest,
    runtime: State<'_, DesktopRuntime>,
) -> Result<RuntimeSnapshot, String> {
    if !runtime.binary_path.is_file() {
        return Err("Build the mumax3-ultrafast engine before starting a simulation.".to_string());
    }
    {
        let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
        if state.child.is_some() {
            return Err("A simulation is already active.".to_string());
        }
        state.logs.clear();
        state.last_error.clear();
        state.exit_code = None;
    }

    let document = save_script_document(
        &request.working_directory,
        &request.file_name,
        &request.contents,
    )?;
    let working_directory = PathBuf::from(&document.working_directory);
    let stem = Path::new(&document.file_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("simulation");
    let output_directory = working_directory.join(format!("{stem}-{}.out", unix_millis()));

    let mut command = Command::new(&runtime.binary_path);
    command
        .arg("-i=false")
        .arg("-openbrowser=false")
        .arg("-http=127.0.0.1:0")
        .arg(format!("-o={}", output_directory.display()))
        .arg(&document.path)
        .current_dir(&working_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Could not start mumax3-ultrafast: {error}");
            mark_runtime_failed(&runtime, &message);
            return Err(message);
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
        state.child = Some(child);
        state.phase = "starting".to_string();
        state.gui_url.clear();
        state.render_url.clear();
        state.output_directory = output_directory.to_string_lossy().into_owned();
        state.script_path = document.path.clone();
        state
            .logs
            .push_back(format!("[wrapper] Running {}", document.path));
    }
    if let Some(stdout) = stdout {
        capture_output(stdout, Arc::clone(&runtime.state), "");
    }
    if let Some(stderr) = stderr {
        capture_output(stderr, Arc::clone(&runtime.state), "[stderr] ");
    }
    snapshot(&runtime)
}

#[tauri::command]
fn stop_simulation(runtime: State<'_, DesktopRuntime>) -> Result<RuntimeSnapshot, String> {
    let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
    if let Some(child) = state.child.as_mut() {
        child
            .kill()
            .map_err(|error| format!("Could not stop the simulation: {error}"))?;
        let _ = child.wait();
    }
    state.child.take();
    state.phase = "stopped".to_string();
    state.gui_url.clear();
    state.render_url.clear();
    state
        .logs
        .push_back("[wrapper] Simulation stopped.".to_string());
    Ok(snapshot_locked(&mut state))
}

#[tauri::command]
fn runtime_snapshot(runtime: State<'_, DesktopRuntime>) -> Result<RuntimeSnapshot, String> {
    snapshot(&runtime)
}

#[tauri::command]
async fn gui_snapshot(
    runtime: State<'_, DesktopRuntime>,
) -> Result<HashMap<String, Value>, String> {
    let gui_url = runtime
        .state
        .lock()
        .map_err(|error| error.to_string())?
        .gui_url
        .clone();
    let gui_url = normalized_loopback_gui_url(&gui_url)?;
    let calls = reqwest::Client::new()
        .post(gui_url)
        .timeout(Duration::from_millis(900))
        .header("content-type", "application/x-www-form-urlencoded")
        .body("id=mumax3-ultrafast-desktop")
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json::<Vec<GuiCall>>()
        .await
        .map_err(|error| error.to_string())?;
    Ok(extract_gui_values(calls))
}

#[tauri::command]
fn list_result_frames(runtime: State<'_, DesktopRuntime>) -> Result<ResultSet, String> {
    let output_directory = current_output_directory(&runtime)?;
    let mut frames = fs::read_dir(&output_directory)
        .map_err(|error| error.to_string())?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let is_ovf = path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("ovf"));
            if !is_ovf || !path.is_file() {
                return None;
            }
            let file_name = path.file_name()?.to_str()?.to_string();
            let quantity = ovf_quantity(&file_name);
            let size_bytes = entry.metadata().ok()?.len();
            Some(ResultFrameInfo {
                file_name,
                quantity,
                size_bytes,
            })
        })
        .collect::<Vec<_>>();
    frames.sort_by(|left, right| left.file_name.cmp(&right.file_name));
    Ok(ResultSet {
        output_directory: output_directory.to_string_lossy().into_owned(),
        frames,
    })
}

#[tauri::command]
async fn load_result_frame(
    file_name: String,
    runtime: State<'_, DesktopRuntime>,
) -> Result<VectorFrame, String> {
    let requested = Path::new(&file_name);
    if requested.components().count() != 1
        || requested
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || !requested
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("ovf"))
    {
        return Err("Choose an OVF file from the current result folder.".to_string());
    }
    let output_directory = current_output_directory(&runtime)?;
    let path = output_directory
        .join(requested)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !path.starts_with(&output_directory) || !path.is_file() {
        return Err("The requested OVF frame is outside the current result folder.".to_string());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let file = File::open(&path).map_err(|error| error.to_string())?;
        parse_ovf_reader(&mut BufReader::new(file), file_name)
    })
    .await
    .map_err(|error| error.to_string())?
}

fn extract_gui_values(calls: Vec<GuiCall>) -> HashMap<String, Value> {
    let mut values = HashMap::new();
    for call in calls {
        if call.f != "setAttr" || call.args.len() < 3 {
            continue;
        }
        let Some(id) = call.args[0].as_str() else {
            continue;
        };
        let Some(attribute) = call.args[1].as_str() else {
            continue;
        };
        if attribute == "innerHTML" || attribute == "value" {
            values.insert(id.to_string(), call.args[2].clone());
        }
    }
    values
}

fn stop_child(runtime: &DesktopRuntime) {
    if let Ok(mut state) = runtime.state.lock() {
        if let Some(child) = state.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        state.child.take();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let repository_root =
        find_repository_root().expect("Could not locate the mumax3-ultrafast repository");
    let runtime = DesktopRuntime {
        binary_path: repository_root.join(".mumax3-ultrafast/bin/mumax3-ultrafast"),
        repository_root,
        state: Arc::new(Mutex::new(ProcessState::default())),
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(runtime)
        .invoke_handler(tauri::generate_handler![
            app_info,
            build_engine,
            save_script,
            load_script,
            start_simulation,
            stop_simulation,
            runtime_snapshot,
            gui_snapshot,
            list_result_frames,
            load_result_frame,
        ])
        .on_window_event(|window, event| {
            if matches!(
                event,
                WindowEvent::CloseRequested { .. } | WindowEvent::Destroyed
            ) {
                stop_child(&window.state::<DesktopRuntime>());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running the mumax3-ultrafast desktop application");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Cursor;

    #[test]
    fn script_names_are_normalized_and_confined_to_one_component() {
        assert_eq!(checked_file_name("sample").unwrap(), "sample.mx3");
        assert_eq!(checked_file_name("sample.mx3").unwrap(), "sample.mx3");
        assert_eq!(checked_file_name("sample.MX3").unwrap(), "sample.MX3");
        assert!(checked_file_name("../sample.mx3").is_err());
        assert!(checked_file_name("folder/sample.mx3").is_err());
        assert!(checked_file_name("").is_err());
    }

    #[test]
    fn gui_values_keep_only_supported_dom_updates() {
        let values = extract_gui_values(vec![
            GuiCall {
                f: "setAttr".to_string(),
                args: vec![json!("nsteps"), json!("innerHTML"), json!(42)],
            },
            GuiCall {
                f: "setAttr".to_string(),
                args: vec![json!("progress"), json!("value"), json!(75)],
            },
            GuiCall {
                f: "setTextbox".to_string(),
                args: vec![json!("Msat"), json!(800000)],
            },
        ]);
        assert_eq!(values.get("nsteps"), Some(&json!(42)));
        assert_eq!(values.get("progress"), Some(&json!(75)));
        assert!(!values.contains_key("Msat"));
    }

    #[test]
    fn engine_output_advances_runtime_phases() {
        let state = Arc::new(Mutex::new(ProcessState {
            phase: "starting".to_string(),
            ..ProcessState::default()
        }));
        push_log(
            &state,
            "//starting GUI at http://127.0.0.1:35367".to_string(),
        );
        {
            let state = state.lock().unwrap();
            assert_eq!(state.phase, "running");
            assert_eq!(state.gui_url, "http://127.0.0.1:35367");
            assert_eq!(state.render_url, "http://127.0.0.1:35367/render/m");
        }
        push_log(&state, "//entering interactive mode".to_string());
        assert_eq!(state.lock().unwrap().phase, "completed");
    }

    #[test]
    fn viewer_urls_are_confined_to_a_loopback_origin() {
        assert_eq!(
            normalized_loopback_gui_url("http://127.0.0.1:35367").unwrap(),
            "http://127.0.0.1:35367"
        );
        assert!(normalized_loopback_gui_url("http://example.com:35367").is_err());
        assert!(normalized_loopback_gui_url("http://127.0.0.1:35367/admin").is_err());
        assert!(normalized_loopback_gui_url("http://127.0.0.1:35367?next=/").is_err());
    }

    #[test]
    fn binary_ovf_frames_are_parsed_into_vector_glyphs() {
        let mut bytes = b"# OOMMF OVF 2.0\n\
# Begin: Header\n\
# Title: m\n\
# Desc: Total simulation time: 1e-09 s\n\
# xnodes: 2\n\
# ynodes: 1\n\
# znodes: 1\n\
# valuedim: 3\n\
# End: Header\n\
# Begin: Data Binary 4\n"
            .to_vec();
        bytes.extend(1_234_567.0f32.to_le_bytes());
        for value in [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bytes.extend(value.to_le_bytes());
        }
        let frame = parse_ovf_reader(&mut Cursor::new(bytes), "m000001.ovf".to_string()).unwrap();
        assert_eq!(frame.dimensions, [2, 1, 1]);
        assert_eq!(frame.glyphs.len(), 12);
        assert_eq!(&frame.glyphs[0..6], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(&frame.glyphs[6..12], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn text_ovf_frames_and_large_mesh_sampling_are_supported() {
        let text = b"# OOMMF OVF 1.0\n\
# Begin: Header\n\
# xnodes: 1\n\
# ynodes: 1\n\
# znodes: 1\n\
# valuedim: 3\n\
# End: Header\n\
# Begin: Data Text\n\
0 0 1\n\
# End: Data Text\n";
        let frame = parse_ovf_reader(&mut Cursor::new(text), "m.ovf".to_string()).unwrap();
        assert_eq!(frame.glyphs, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let dimensions = [4096, 4096, 1];
        let steps = sampled_steps(dimensions);
        let sampled = dimensions
            .iter()
            .zip(steps)
            .map(|(dimension, step)| dimension.div_ceil(step))
            .product::<usize>();
        assert!(sampled <= MAX_RESULT_GLYPHS);
    }

    #[test]
    fn ovf_frame_names_are_grouped_by_quantity() {
        assert_eq!(ovf_quantity("m000010.ovf"), "m");
        assert_eq!(ovf_quantity("B_demag-0012.ovf"), "B_demag");
        assert_eq!(ovf_quantity("final.ovf"), "final");
    }

    #[test]
    fn scalar_ovf_frames_are_not_misrepresented_as_vectors() {
        let text = b"# OOMMF OVF 2.0\n\
# Begin: Header\n\
# xnodes: 1\n\
# ynodes: 1\n\
# znodes: 1\n\
# valuedim: 1\n\
# End: Header\n\
# Begin: Data Text\n\
1\n";
        let error = match parse_ovf_reader(&mut Cursor::new(text), "energy.ovf".to_string()) {
            Ok(_) => panic!("scalar data must not be rendered as a vector field"),
            Err(error) => error,
        };
        assert!(error.contains("3-component OVF vector fields"));
    }
}
