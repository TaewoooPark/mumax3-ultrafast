export type RuntimePhase = "idle" | "building" | "ready" | "starting" | "running" | "completed" | "failed" | "stopped";

export interface AppInfo {
  repositoryRoot: string;
  defaultWorkingDirectory: string;
  binaryPath: string;
  engineBuilt: boolean;
}

export interface BuildResult {
  binaryPath: string;
  output: string;
}

export interface ScriptDocument {
  path: string;
  workingDirectory: string;
  fileName: string;
  contents: string;
}

export interface StartRequest {
  workingDirectory: string;
  fileName: string;
  contents: string;
}

export interface RuntimeSnapshot {
  phase: RuntimePhase;
  processAlive: boolean;
  viewerAvailable: boolean;
  guiUrl: string;
  renderUrl: string;
  outputDirectory: string;
  scriptPath: string;
  logs: string[];
  lastError: string;
  exitCode: number | null;
}

export type GuiValues = Record<string, string | number | boolean>;
