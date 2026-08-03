import { invoke } from "@tauri-apps/api/core";
import type { AppInfo, BuildResult, GuiValues, RuntimeSnapshot, ScriptDocument, StartRequest } from "./types";

export const desktop = {
  info: () => invoke<AppInfo>("app_info"),
  buildEngine: () => invoke<BuildResult>("build_engine"),
  saveScript: (workingDirectory: string, fileName: string, contents: string) =>
    invoke<ScriptDocument>("save_script", { workingDirectory, fileName, contents }),
  loadScript: (path: string) => invoke<ScriptDocument>("load_script", { path }),
  startSimulation: (request: StartRequest) => invoke<RuntimeSnapshot>("start_simulation", { request }),
  stopSimulation: () => invoke<RuntimeSnapshot>("stop_simulation"),
  runtimeSnapshot: () => invoke<RuntimeSnapshot>("runtime_snapshot"),
  guiSnapshot: (guiUrl: string) => invoke<GuiValues>("gui_snapshot", { guiUrl }),
};
