use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, VecDeque},
    env, fs,
    io::{BufRead, BufReader, Read},
    net::TcpListener,
    path::{Component, Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{Manager, State, WindowEvent};

const MAX_LOG_LINES: usize = 400;

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

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct GuiCall {
    f: String,
    args: Vec<Value>,
}

fn find_repository_root() -> Result<PathBuf, String> {
    if let Some(configured) = env::var_os("MUMAX3_ULTRAFAST_HOME").map(PathBuf::from) {
        if configured.join("go.mod").is_file() && configured.join("cmd/mumax3").is_dir() {
            return configured.canonicalize().map_err(|error| error.to_string());
        }
    }

    if let Ok(current) = env::current_dir() {
        if let Some(root) = current
            .ancestors()
            .find(|path| path.join("go.mod").is_file() && path.join("cmd/mumax3").is_dir())
        {
            return Ok(root.to_path_buf());
        }
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
    if path.extension().is_some_and(|extension| extension == "mx3") {
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
        if line.contains("//starting GUI at") && state.phase == "starting" {
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

#[tauri::command]
fn app_info(runtime: State<'_, DesktopRuntime>) -> AppInfo {
    AppInfo {
        repository_root: runtime.repository_root.to_string_lossy().into_owned(),
        default_working_directory: runtime.repository_root.to_string_lossy().into_owned(),
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

    if let Some(parent) = runtime.binary_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let output = Command::new("go")
        .args(["build", "-o"])
        .arg(&runtime.binary_path)
        .arg("./cmd/mumax3")
        .current_dir(&runtime.repository_root)
        .output()
        .map_err(|error| format!("Could not start the Go compiler: {error}"))?;
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
    if !path.extension().is_some_and(|extension| extension == "mx3") {
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

    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|error| format!("Could not reserve a local viewer port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    drop(listener);
    let gui_url = format!("http://127.0.0.1:{port}");

    let mut command = Command::new(&runtime.binary_path);
    command
        .arg("-i=true")
        .arg("-openbrowser=false")
        .arg(format!("-http=127.0.0.1:{port}"))
        .arg(format!("-o={}", output_directory.display()))
        .arg(&document.path)
        .current_dir(&working_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("Could not start mumax3-ultrafast: {error}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    {
        let mut state = runtime.state.lock().map_err(|error| error.to_string())?;
        state.child = Some(child);
        state.phase = "starting".to_string();
        state.gui_url = gui_url.clone();
        state.render_url = format!("{gui_url}/render/m");
        state.output_directory = output_directory.to_string_lossy().into_owned();
        state.script_path = document.path.clone();
        state
            .logs
            .push_back(format!("[wrapper] Running {}", document.path));
        state
            .logs
            .push_back(format!("[wrapper] Full viewer: {gui_url}"));
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
async fn gui_snapshot(gui_url: String) -> Result<HashMap<String, Value>, String> {
    if !gui_url.starts_with("http://127.0.0.1:") {
        return Err("The viewer URL must use the local loopback interface.".to_string());
    }
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
    Ok(values)
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
        ])
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::Destroyed) {
                stop_child(&window.state::<DesktopRuntime>());
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running the mumax3-ultrafast desktop application");
}
