import { contextBridge, ipcRenderer } from "electron";
import type {
  AppSettings,
  CivitaiModelCard,
  DeleteModelRequest,
  DownloadRequest,
  DownloadResult,
  HuggingFaceLlmModelCard,
  LlmChatMessage,
  LlmChatRequest,
  LoraTrainingProject,
  IllustrationTag,
  OllamaStatus,
  PromptEnhanceRequest,
  PromptEnhanceResult,
  PromptHistoryEntry,
  RuntimePaths,
  RuntimeStatus,
  SaveIllustrationTagRequest,
  SaveLoraTrainingProjectRequest,
  SavePromptHistoryRequest,
  TagDescriptionRequest,
  TagTranslationRequest,
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
    runOperation: (request: { id: string; operation: string; model?: string; ollamaUrl?: string; projectId?: string }): Promise<string> =>
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
    searchHuggingFaceLlm: (options: { query: string; limit?: number }): Promise<HuggingFaceLlmModelCard[]> =>
      ipcRenderer.invoke("runtime:search-huggingface-llm", options),
    testGenerate: (request: TestGenerationRequest): Promise<TestGenerationResult> =>
      ipcRenderer.invoke("runtime:test-generate", request)
  },
  ollama: {
    status: (ollamaUrl?: string): Promise<OllamaStatus> => ipcRenderer.invoke("ollama:status", ollamaUrl),
    start: (): Promise<string> => ipcRenderer.invoke("ollama:start"),
    stop: (): Promise<string> => ipcRenderer.invoke("ollama:stop"),
    deleteModel: (request: { ollamaUrl: string; model: string }): Promise<string> =>
      ipcRenderer.invoke("ollama:delete-model", request),
    enhancePrompt: (request: PromptEnhanceRequest): Promise<PromptEnhanceResult> =>
      ipcRenderer.invoke("ollama:enhance-prompt", request),
    chatHistory: (): Promise<LlmChatMessage[]> => ipcRenderer.invoke("ollama:chat-history"),
    clearChatHistory: (): Promise<LlmChatMessage[]> => ipcRenderer.invoke("ollama:clear-chat-history"),
    chat: (request: LlmChatRequest): Promise<LlmChatMessage[]> => ipcRenderer.invoke("ollama:chat", request)
  },
  tags: {
    list: (): Promise<IllustrationTag[]> => ipcRenderer.invoke("tags:list"),
    generateJapanese: (request: TagDescriptionRequest): Promise<string> => ipcRenderer.invoke("tags:generate-ja", request),
    translateEnglish: (request: TagTranslationRequest): Promise<string> => ipcRenderer.invoke("tags:translate-en", request),
    save: (request: SaveIllustrationTagRequest): Promise<IllustrationTag[]> => ipcRenderer.invoke("tags:save", request),
    delete: (id: string): Promise<IllustrationTag[]> => ipcRenderer.invoke("tags:delete", id)
  },
  loraProjects: {
    list: (): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:list"),
    save: (request: SaveLoraTrainingProjectRequest): Promise<LoraTrainingProject[]> =>
      ipcRenderer.invoke("lora-projects:save", request),
    start: (id: string): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:start", id)
  },
  promptHistory: {
    list: (): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:list"),
    save: (request: SavePromptHistoryRequest): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:save", request),
    clear: (): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:clear")
  },
  app: {
    openPath: (path: string): Promise<string> => ipcRenderer.invoke("app:open-path", path)
  }
};

contextBridge.exposeInMainWorld("illustMK", api);

export type IllustMKApi = typeof api;
