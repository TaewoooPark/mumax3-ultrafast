import { invoke } from "@tauri-apps/api/core";
import type { AppInfo, BuildResult, GuiValues, ResultSet, RuntimeSnapshot, ScriptDocument, StartRequest, VectorFrame } from "./types";

export const desktop = {
  info: () => invoke<AppInfo>("app_info"),
  buildEngine: () => invoke<BuildResult>("build_engine"),
  saveScript: (workingDirectory: string, fileName: string, contents: string) =>
    invoke<ScriptDocument>("save_script", { workingDirectory, fileName, contents }),
  loadScript: (path: string) => invoke<ScriptDocument>("load_script", { path }),
  startSimulation: (request: StartRequest) => invoke<RuntimeSnapshot>("start_simulation", { request }),
  stopSimulation: () => invoke<RuntimeSnapshot>("stop_simulation"),
  runtimeSnapshot: () => invoke<RuntimeSnapshot>("runtime_snapshot"),
  guiSnapshot: () => invoke<GuiValues>("gui_snapshot"),
  listResultFrames: () => invoke<ResultSet>("list_result_frames"),
  loadResultFrame: (fileName: string) => invoke<VectorFrame>("load_result_frame", { fileName }),
};
