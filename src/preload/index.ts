import { contextBridge, ipcRenderer } from "electron";
import type {
  AppSettings,
  CivitaiModelCard,
  DeleteModelRequest,
  DownloadRequest,
  DownloadResult,
  RuntimePaths,
  RuntimeStatus,
  TestGenerationRequest,
  TestGenerationResult
} from "../main/runtime";

type OperationLog = {
  id: string;
  level: "info" | "stdout" | "stderr";
  line: string;
};

const api = {
  settings: {
    get: (): Promise<AppSettings> => ipcRenderer.invoke("settings:get"),
    save: (settings: AppSettings): Promise<AppSettings> => ipcRenderer.invoke("settings:save", settings)
  },
  runtime: {
    paths: (): Promise<RuntimePaths> => ipcRenderer.invoke("runtime:paths"),
    ensure: (): Promise<RuntimePaths> => ipcRenderer.invoke("runtime:ensure"),
    status: (comfyUrl?: string): Promise<RuntimeStatus> => ipcRenderer.invoke("runtime:status", comfyUrl),
    startDockerEngine: (): Promise<string> => ipcRenderer.invoke("runtime:start-docker-engine"),
    stopDockerEngine: (): Promise<string> => ipcRenderer.invoke("runtime:stop-docker-engine"),
    build: (): Promise<string> => ipcRenderer.invoke("runtime:build"),
    start: (): Promise<string> => ipcRenderer.invoke("runtime:start"),
    stop: (): Promise<string> => ipcRenderer.invoke("runtime:stop"),
    runOperation: (request: { id: string; operation: string }): Promise<string> =>
      ipcRenderer.invoke("runtime:run-operation", request),
    onOperationLog: (callback: (event: OperationLog) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, payload: OperationLog) => callback(payload);
      ipcRenderer.on("runtime:operation-log", listener);
      return () => ipcRenderer.removeListener("runtime:operation-log", listener);
    },
    downloadModel: (request: DownloadRequest): Promise<DownloadResult> =>
      ipcRenderer.invoke("runtime:download-model", request),
    deleteModel: (request: DeleteModelRequest): Promise<string> => ipcRenderer.invoke("runtime:delete-model", request),
    searchCivitai: (options: {
      query?: string;
      sort?: "Most Downloaded" | "Highest Rated" | "Newest";
      period?: "AllTime" | "Year" | "Month" | "Week" | "Day";
      limit?: number;
      nsfw?: boolean;
      baseModels?: string[];
    }): Promise<CivitaiModelCard[]> => ipcRenderer.invoke("runtime:search-civitai", options),
    testGenerate: (request: TestGenerationRequest): Promise<TestGenerationResult> =>
      ipcRenderer.invoke("runtime:test-generate", request)
  },
  app: {
    openPath: (path: string): Promise<string> => ipcRenderer.invoke("app:open-path", path)
  }
};

contextBridge.exposeInMainWorld("illustMK", api);

export type IllustMKApi = typeof api;
