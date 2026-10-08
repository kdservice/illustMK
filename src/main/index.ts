import { app, BrowserWindow, ipcMain, shell } from "electron";
import { join } from "node:path";
import {
  buildRuntime,
  buildRuntimeWithLogs,
  cleanupDockerRuntime,
  deleteModelFile,
  deleteIllustrationTag,
  deleteOllamaModelStorage,
  deleteOllamaModel,
  downloadModel,
  enhancePromptWithOllama,
  ensureRuntimeFiles,
  getRuntimePaths,
  getStatus,
  getAppSettings,
  getIllustrationTags,
  getLlmChatHistory,
  getOllamaStatus,
  getSavedPromptHistory,
  generateTagJapaneseDescription,
  initNativeRuntime,
  installOllama,
  installNativeRuntime,
  pullOllamaModel,
  runTestGeneration,
  searchCivitaiModels,
  searchHuggingFaceLlmModels,
  saveAppSettings,
  saveIllustrationTag,
  savePromptHistoryEntry,
  sendLlmChatPrompt,
  showComfyContainerLogs,
  showNativeLogs,
  startNativeRuntime,
  startDockerEngine,
  startOllama,
  startRuntime,
  startRuntimeWithLogs,
  stopDockerEngine,
  stopNativeRuntime,
  stopOllama,
  uninstallOllama,
  translateTagDescription,
  clearPromptHistory,
  clearLlmChatHistory,
  stopRuntime,
  stopRuntimeWithLogs,
  type AppSettings,
  type DeleteModelRequest,
  type DownloadRequest,
  type LlmChatRequest,
  type PromptEnhanceRequest,
  type SaveIllustrationTagRequest,
  type SavePromptHistoryRequest,
  type TagDescriptionRequest,
  type TagTranslationRequest,
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
  ipcMain.handle("runtime:run-operation", async (event, request: { id: string; operation: string; model?: string; ollamaUrl?: string }) => {
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
  ipcMain.handle("runtime:delete-model", (_event, request: DeleteModelRequest) => deleteModelFile(request));
  ipcMain.handle("runtime:search-civitai", (_event, options) => searchCivitaiModels(options));
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
  ipcMain.handle("app:open-path", (_event, path: string) => shell.openPath(path));
}
