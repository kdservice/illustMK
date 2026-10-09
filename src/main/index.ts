import { app, BrowserWindow, dialog, ipcMain, screen, shell, type OpenDialogOptions, type Rectangle } from "electron";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  buildRuntime,
  buildRuntimeWithLogs,
  cleanupDockerRuntime,
  deleteLoraTrainingProject,
  deleteModelFile,
  deleteIllustrationTag,
  deleteOllamaModelStorage,
  deleteOllamaModel,
  downloadCivitaiLoraProject,
  downloadModel,
  enhancePromptWithOllama,
  ensureRuntimeFiles,
  getRuntimePaths,
  getStatus,
  getAppSettings,
  getIllustrationTags,
  getLlmChatHistory,
  getLoraCaptionWorkspace,
  getLoraTrainingProjects,
  getOllamaStatus,
  getSavedPromptHistory,
  generateTagJapaneseDescription,
  initNativeRuntime,
  installLoraRuntime,
  importModelAsset,
  importLoraFile,
  installOllama,
  installNativeRuntime,
  pullOllamaModel,
  pauseLoraTrainingProject,
  runTestGeneration,
  searchCivitaiModelPage,
  searchCivitaiModels,
  searchHuggingFaceLlmModels,
  saveAppSettings,
  saveIllustrationTag,
  saveLoraCaptionWorkspace,
  saveLoraTrainingProject,
  savePromptHistoryEntry,
  sendLlmChatPrompt,
  showComfyContainerLogs,
  showNativeLogs,
  startNativeRuntime,
  startDockerEngine,
  startOllama,
  startRuntime,
  startRuntimeWithLogs,
  startLoraTrainingProject,
  stopAllLoraTrainingProcesses,
  stopDockerEngine,
  stopLoraTrainingProject,
  stopNativeRuntime,
  stopOllama,
  uninstallOllama,
  translateTagDescription,
  trainLoraTrainingProject,
  updateLoraTrainingProjectThumbnail,
  clearPromptHistory,
  clearLoraTrainingProjects,
  clearLlmChatHistory,
  stopRuntime,
  stopRuntimeWithLogs,
  type AppSettings,
  type DeleteModelRequest,
  type DownloadCivitaiLoraProjectRequest,
  type DownloadRequest,
  type ImportModelAssetRequest,
  type LoraTrainingRunParams,
  type LlmChatRequest,
  type PromptEnhanceRequest,
  type SaveLoraCaptionWorkspaceRequest,
  type SaveIllustrationTagRequest,
  type SaveLoraTrainingProjectRequest,
  type SavePromptHistoryRequest,
  type TagDescriptionRequest,
  type TagTranslationRequest,
  type TestGenerationRequest,
  type UpdateLoraThumbnailRequest
} from "./runtime";

let mainWindow: BrowserWindow | null = null;
let quittingAfterCleanup = false;

type WindowState = {
  x?: number;
  y?: number;
  width: number;
  height: number;
};

function windowStatePath(): string {
  return join(app.getPath("userData"), "window-state.json");
}

async function loadWindowState(): Promise<WindowState | null> {
  try {
    const parsed = JSON.parse(await readFile(windowStatePath(), "utf8")) as Partial<WindowState>;
    if (!Number.isFinite(parsed.width) || !Number.isFinite(parsed.height)) return null;
    return {
      x: Number.isFinite(parsed.x) ? parsed.x : undefined,
      y: Number.isFinite(parsed.y) ? parsed.y : undefined,
      width: Math.max(1120, Number(parsed.width)),
      height: Math.max(760, Number(parsed.height))
    };
  } catch {
    return null;
  }
}

function isVisibleWindowState(state: WindowState): state is Required<WindowState> {
  if (!Number.isFinite(state.x) || !Number.isFinite(state.y)) return false;
  const bounds: Rectangle = { x: Number(state.x), y: Number(state.y), width: state.width, height: state.height };
  return screen.getAllDisplays().some((display) => {
    const area = display.workArea;
    const right = bounds.x + bounds.width;
    const bottom = bounds.y + bounds.height;
    return bounds.x < area.x + area.width && right > area.x && bounds.y < area.y + area.height && bottom > area.y;
  });
}

async function saveWindowState(window: BrowserWindow): Promise<void> {
  if (window.isDestroyed() || window.isMinimized()) return;
  const bounds = window.getBounds();
  await mkdir(app.getPath("userData"), { recursive: true });
  await writeFile(windowStatePath(), JSON.stringify(bounds, null, 2), "utf8");
}

async function createWindow(): Promise<void> {
  const savedState = await loadWindowState();
  const restoredState = savedState && isVisibleWindowState(savedState) ? savedState : null;
  mainWindow = new BrowserWindow({
    x: restoredState?.x,
    y: restoredState?.y,
    width: restoredState?.width ?? 1440,
    height: restoredState?.height ?? 980,
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

  let saveTimer: NodeJS.Timeout | null = null;
  const scheduleWindowStateSave = () => {
    if (!mainWindow) return;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (mainWindow) void saveWindowState(mainWindow);
    }, 300);
  };
  mainWindow.on("resize", scheduleWindowStateSave);
  mainWindow.on("move", scheduleWindowStateSave);
  mainWindow.on("close", () => {
    if (mainWindow) void saveWindowState(mainWindow);
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

app.on("before-quit", (event) => {
  if (quittingAfterCleanup) return;
  event.preventDefault();
  quittingAfterCleanup = true;
  void stopAllLoraTrainingProcesses().finally(() => app.quit());
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
  ipcMain.handle("runtime:run-operation", async (event, request: { id: string; operation: string; model?: string; ollamaUrl?: string; projectId?: string; sourceLora?: string; runParams?: Partial<LoraTrainingRunParams> }) => {
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
      if (request.operation === "lora-install") {
        return await installLoraRuntime((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "lora-train") {
        return await trainLoraTrainingProject(request.projectId ?? "", (log) => sendLog(log.level, log.line), { sourceLora: request.sourceLora, runParams: request.runParams });
      }
      if (request.operation === "lora-resume") {
        return await trainLoraTrainingProject(request.projectId ?? "", (log) => sendLog(log.level, log.line), { resume: true, sourceLora: request.sourceLora, runParams: request.runParams });
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
      if (request.operation === "ollama-install") {
        return await installOllama((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "ollama-uninstall") {
        return await uninstallOllama((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "ollama-delete-storage") {
        return await deleteOllamaModelStorage((log) => sendLog(log.level, log.line));
      }
      if (request.operation === "ollama-pull") {
        return await pullOllamaModel(request.ollamaUrl ?? "http://127.0.0.1:11434", request.model ?? "", (log) =>
          sendLog(log.level, log.line)
        );
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
  ipcMain.handle("lora-projects:download-civitai", (event, request: DownloadCivitaiLoraProjectRequest) =>
    downloadCivitaiLoraProject(request, (log) => {
      event.sender.send("runtime:operation-log", {
        id: request.operationId ?? "lora-civitai-download",
        level: log.level,
        line: log.line
      });
    })
  );
  ipcMain.handle("runtime:delete-model", (_event, request: DeleteModelRequest) => deleteModelFile(request));
  ipcMain.handle("runtime:import-model-asset", async (_event, request: Omit<ImportModelAssetRequest, "sourcePath">) => {
    const label = request.targetType === "vae" ? "VAE" : "CLIP/text encoder";
    const options: OpenDialogOptions = {
      title: `${label} safetensorsをインポート`,
      properties: ["openFile"],
      filters: [{ name: `${label} safetensors`, extensions: ["safetensors"] }]
    };
    const result = mainWindow ? await dialog.showOpenDialog(mainWindow, options) : await dialog.showOpenDialog(options);
    if (result.canceled || !result.filePaths[0]) return null;
    return importModelAsset({ targetType: request.targetType, sourcePath: result.filePaths[0] });
  });
  ipcMain.handle("runtime:search-civitai", (_event, options) => searchCivitaiModels(options));
  ipcMain.handle("runtime:search-civitai-page", (_event, options) => searchCivitaiModelPage(options));
  ipcMain.handle("runtime:search-huggingface-llm", (_event, options) => searchHuggingFaceLlmModels(options));
  ipcMain.handle("runtime:test-generate", (_event, request: TestGenerationRequest) => runTestGeneration(request));
  ipcMain.handle("prompt-history:list", () => getSavedPromptHistory());
  ipcMain.handle("prompt-history:save", (_event, request: SavePromptHistoryRequest) => savePromptHistoryEntry(request));
  ipcMain.handle("prompt-history:clear", () => clearPromptHistory());
  ipcMain.handle("tags:list", () => getIllustrationTags());
  ipcMain.handle("tags:generate-ja", (_event, request: TagDescriptionRequest) => generateTagJapaneseDescription(request));
  ipcMain.handle("tags:translate-en", (_event, request: TagTranslationRequest) => translateTagDescription(request));
  ipcMain.handle("tags:save", (_event, request: SaveIllustrationTagRequest) => saveIllustrationTag(request));
  ipcMain.handle("tags:delete", (_event, id: string) => deleteIllustrationTag(id));
  ipcMain.handle("lora-projects:list", () => getLoraTrainingProjects());
  ipcMain.handle("lora-projects:save", (_event, request: SaveLoraTrainingProjectRequest) => saveLoraTrainingProject(request));
  ipcMain.handle("lora-projects:update-thumbnail", (_event, request: UpdateLoraThumbnailRequest) => updateLoraTrainingProjectThumbnail(request));
  ipcMain.handle("lora-projects:start", (_event, id: string) => startLoraTrainingProject(id));
  ipcMain.handle("lora-projects:pause", (_event, id: string) => pauseLoraTrainingProject(id));
  ipcMain.handle("lora-projects:stop", (_event, id: string) => stopLoraTrainingProject(id));
  ipcMain.handle("lora-projects:delete", (_event, id: string) => deleteLoraTrainingProject(id));
  ipcMain.handle("lora-projects:clear", () => clearLoraTrainingProjects());
  ipcMain.handle("lora-projects:import", async () => {
    const options: OpenDialogOptions = {
      title: "LoRA safetensorsをインポート",
      properties: ["openFile"],
      filters: [{ name: "LoRA safetensors", extensions: ["safetensors"] }]
    };
    const result = mainWindow ? await dialog.showOpenDialog(mainWindow, options) : await dialog.showOpenDialog(options);
    if (result.canceled || !result.filePaths[0]) return getLoraTrainingProjects();
    return importLoraFile(result.filePaths[0]);
  });
  ipcMain.handle("lora-captions:get", (_event, projectId: string) => getLoraCaptionWorkspace(projectId));
  ipcMain.handle("lora-captions:save", (_event, request: SaveLoraCaptionWorkspaceRequest) => saveLoraCaptionWorkspace(request));
  ipcMain.handle("ollama:status", (_event, ollamaUrl?: string) => getOllamaStatus(ollamaUrl));
  ipcMain.handle("ollama:start", () => startOllama());
  ipcMain.handle("ollama:stop", () => stopOllama());
  ipcMain.handle("ollama:delete-model", (_event, request: { ollamaUrl: string; model: string }) =>
    deleteOllamaModel(request.ollamaUrl, request.model)
  );
  ipcMain.handle("ollama:enhance-prompt", (_event, request: PromptEnhanceRequest) => enhancePromptWithOllama(request));
  ipcMain.handle("ollama:chat-history", () => getLlmChatHistory());
  ipcMain.handle("ollama:clear-chat-history", () => clearLlmChatHistory());
  ipcMain.handle("ollama:chat", (_event, request: LlmChatRequest) => sendLlmChatPrompt(request));
  ipcMain.handle("app:open-path", async (_event, path: string) => {
    const pathStat = await stat(path).catch(() => null);
    if (pathStat?.isDirectory()) {
      return shell.openPath(path);
    }
    shell.showItemInFolder(path);
    return "";
  });
  ipcMain.handle("app:open-external", (_event, url: string) => shell.openExternal(url));
}
