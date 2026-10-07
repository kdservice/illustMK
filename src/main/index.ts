import { app, BrowserWindow, ipcMain, shell } from "electron";
import { join } from "node:path";
import {
  buildRuntime,
  buildRuntimeWithLogs,
  cleanupDockerRuntime,
  deleteModelFile,
  downloadModel,
  ensureRuntimeFiles,
  getRuntimePaths,
  getStatus,
  getAppSettings,
  initNativeRuntime,
  installNativeRuntime,
  runTestGeneration,
  searchCivitaiModels,
  saveAppSettings,
  showComfyContainerLogs,
  showNativeLogs,
  startNativeRuntime,
  startDockerEngine,
  startRuntime,
  startRuntimeWithLogs,
  stopDockerEngine,
  stopNativeRuntime,
  stopRuntime,
  stopRuntimeWithLogs,
  type AppSettings,
  type DeleteModelRequest,
  type DownloadRequest,
  type TestGenerationRequest
} from "./runtime";

let mainWindow: BrowserWindow | null = null;

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 980,
    minWidth: 1120,
    minHeight: 760,
    title: "illustMK",
    backgroundColor: "#071020",
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  const devServerUrl = process.env.VITE_DEV_SERVER_URL;
  if (devServerUrl) {
    await mainWindow.loadURL(devServerUrl);
  } else {
    await mainWindow.loadFile(join(__dirname, "../../renderer/index.html"));
  }
}

app.whenReady().then(async () => {
  registerIpc();
  await createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    await createWindow();
  }
});

function registerIpc(): void {
  ipcMain.handle("runtime:paths", () => getRuntimePaths());
  ipcMain.handle("settings:get", () => getAppSettings());
  ipcMain.handle("settings:save", (_event, settings: AppSettings) => saveAppSettings(settings));
  ipcMain.handle("runtime:ensure", () => ensureRuntimeFiles());
  ipcMain.handle("runtime:status", (_event, comfyUrl?: string) => getStatus(comfyUrl));
  ipcMain.handle("runtime:start-docker-engine", () => startDockerEngine());
  ipcMain.handle("runtime:stop-docker-engine", () => stopDockerEngine());
  ipcMain.handle("runtime:build", () => buildRuntime());
  ipcMain.handle("runtime:start", () => startRuntime());
  ipcMain.handle("runtime:stop", () => stopRuntime());
  ipcMain.handle("runtime:run-operation", async (event, request: { id: string; operation: string }) => {
    const sendLog = (level: "info" | "stdout" | "stderr", line: string) => {
      event.sender.send("runtime:operation-log", { id: request.id, level, line });
    };

    try {
      if (request.operation === "build") {
        return await buildRuntimeWithLogs((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "start") {
        return await startRuntimeWithLogs((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "stop") {
        return await stopRuntimeWithLogs((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "logs") {
        return await showComfyContainerLogs((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "native-init") {
        return await initNativeRuntime((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "native-install") {
        return await installNativeRuntime((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "native-start") {
        return await startNativeRuntime((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "native-stop") {
        return await stopNativeRuntime((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "native-logs") {
        return await showNativeLogs((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "docker-cleanup") {
        return await cleanupDockerRuntime((log) => sendLog(log.level, log.line));
      }
      throw new Error(`Unknown operation: ${request.operation}`);
    } catch (error) {
      sendLog("stderr", error instanceof Error ? error.message : String(error));
      throw error;
    }
  });
  ipcMain.handle("runtime:download-model", (event, request: DownloadRequest) =>
    downloadModel(request, (log) => {
      event.sender.send("runtime:operation-log", {
        id: request.operationId ?? "download",
        level: log.level,
        line: log.line
      });
    })
  );
  ipcMain.handle("runtime:delete-model", (_event, request: DeleteModelRequest) => deleteModelFile(request));
  ipcMain.handle("runtime:search-civitai", (_event, options) => searchCivitaiModels(options));
  ipcMain.handle("runtime:test-generate", (_event, request: TestGenerationRequest) => runTestGeneration(request));
  ipcMain.handle("app:open-path", (_event, path: string) => shell.openPath(path));
}
