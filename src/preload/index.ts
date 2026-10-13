import { contextBridge, ipcRenderer } from "electron";
import type {
  AppSettings,
  CivitaiModelCard,
  CivitaiSearchResult,
  CivitaiModelType,
  DataPathEntry,
  DeleteModelRequest,
  DownloadCivitaiLoraProjectRequest,
  DownloadRequest,
  DownloadResult,
  HuggingFaceLlmModelCard,
  LlmChatMessage,
  LlmChatRequest,
  ModelAssetType,
  ModelLibraryEntry,
  LoraCaptionWorkspace,
  LoraTrainingRunParams,
  LoraTrainingProject,
  IllustrationTag,
  OllamaStatus,
  OpenPoseAsset,
  OpenPoseCatalogItem,
  PoseConditioning,
  PoseImageImport,
  PoseExtractionRequest,
  PoseExtractionResult,
  PosePreprocessorStatus,
  PoseVisionRequest,
  PoseVisionResult,
  PromptEnhanceRequest,
  PromptEnhanceResult,
  PromptHistoryEntry,
  RuntimePaths,
  RuntimeStatus,
  SavedPose,
  SaveIllustrationTagRequest,
  SaveLoraCaptionWorkspaceRequest,
  SaveLoraTrainingProjectRequest,
  SavePromptHistoryRequest,
  TagDescriptionRequest,
  TagTranslationRequest,
  TestGenerationRequest,
  TestGenerationResult,
  UpdatePromptHistoryMemoRequest,
  UpdateLoraThumbnailRequest
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
    dataPaths: (): Promise<DataPathEntry[]> => ipcRenderer.invoke("runtime:data-paths"),
    addDataPath: (): Promise<DataPathEntry[]> => ipcRenderer.invoke("runtime:add-data-path"),
    importDataPath: (): Promise<DataPathEntry[]> => ipcRenderer.invoke("runtime:import-data-path"),
    switchDataPath: (id: string): Promise<DataPathEntry[]> => ipcRenderer.invoke("runtime:switch-data-path", id),
    deleteDataPath: (id: string): Promise<DataPathEntry[]> => ipcRenderer.invoke("runtime:delete-data-path", id),
    updateDataPathMemo: (request: { id: string; memo: string }): Promise<DataPathEntry[]> =>
      ipcRenderer.invoke("runtime:update-data-path-memo", request),
    ensure: (): Promise<RuntimePaths> => ipcRenderer.invoke("runtime:ensure"),
    status: (comfyUrl?: string): Promise<RuntimeStatus> => ipcRenderer.invoke("runtime:status", comfyUrl),
    startDockerEngine: (): Promise<string> => ipcRenderer.invoke("runtime:start-docker-engine"),
    stopDockerEngine: (): Promise<string> => ipcRenderer.invoke("runtime:stop-docker-engine"),
    build: (): Promise<string> => ipcRenderer.invoke("runtime:build"),
    start: (): Promise<string> => ipcRenderer.invoke("runtime:start"),
    stop: (): Promise<string> => ipcRenderer.invoke("runtime:stop"),
    runOperation: (request: { id: string; operation: string; model?: string; ollamaUrl?: string; projectId?: string; sourceLora?: string; runParams?: Partial<LoraTrainingRunParams> }): Promise<string> =>
      ipcRenderer.invoke("runtime:run-operation", request),
    onOperationLog: (callback: (event: OperationLog) => void): (() => void) => {
      const listener = (_event: Electron.IpcRendererEvent, payload: OperationLog) => callback(payload);
      ipcRenderer.on("runtime:operation-log", listener);
      return () => ipcRenderer.removeListener("runtime:operation-log", listener);
    },
    downloadModel: (request: DownloadRequest): Promise<DownloadResult> =>
      ipcRenderer.invoke("runtime:download-model", request),
    importModelAsset: (request: { targetType: "vae" | "clip" | "controlnet" }): Promise<DownloadResult | null> =>
      ipcRenderer.invoke("runtime:import-model-asset", request),
    deleteModel: (request: DeleteModelRequest): Promise<string> => ipcRenderer.invoke("runtime:delete-model", request),
    searchCivitai: (options: {
      query?: string;
      sort?: "Most Downloaded" | "Highest Rated" | "Newest";
      period?: "AllTime" | "Year" | "Month" | "Week" | "Day";
      limit?: number;
      page?: number;
      cursor?: string;
      nsfw?: boolean;
      baseModels?: string[];
      type?: CivitaiModelType;
    }): Promise<CivitaiModelCard[]> => ipcRenderer.invoke("runtime:search-civitai", options),
    searchCivitaiPage: (options: {
      query?: string;
      sort?: "Most Downloaded" | "Highest Rated" | "Newest";
      period?: "AllTime" | "Year" | "Month" | "Week" | "Day";
      limit?: number;
      page?: number;
      cursor?: string;
      nsfw?: boolean;
      baseModels?: string[];
      type?: CivitaiModelType;
    }): Promise<CivitaiSearchResult> => ipcRenderer.invoke("runtime:search-civitai-page", options),
    searchHuggingFaceLlm: (options: { query: string; limit?: number }): Promise<HuggingFaceLlmModelCard[]> =>
      ipcRenderer.invoke("runtime:search-huggingface-llm", options),
    testGenerate: (request: TestGenerationRequest): Promise<TestGenerationResult> =>
      ipcRenderer.invoke("runtime:test-generate", request)
  },
  models: {
    list: (): Promise<ModelLibraryEntry[]> => ipcRenderer.invoke("models:list"),
    setActive: (id: string): Promise<ModelLibraryEntry[]> => ipcRenderer.invoke("models:set-active", id),
    save: (request: { id: string; name?: string; baseModel?: string; usage?: string; selectedVae?: string; selectedClip?: string }): Promise<ModelLibraryEntry[]> =>
      ipcRenderer.invoke("models:save", request),
    delete: (id: string): Promise<ModelLibraryEntry[]> => ipcRenderer.invoke("models:delete", id),
    importFile: (type: ModelAssetType): Promise<ModelLibraryEntry[]> => ipcRenderer.invoke("models:import", type),
    downloadCivitai: (request: { model: CivitaiModelCard; targetType: ModelAssetType; operationId?: string }): Promise<ModelLibraryEntry[]> =>
      ipcRenderer.invoke("models:download-civitai", request),
    downloadDirect: (request: { url: string; fileName?: string; targetType: ModelAssetType; baseModel?: string; usage?: string; operationId?: string }): Promise<ModelLibraryEntry[]> =>
      ipcRenderer.invoke("models:download-direct", request)
  },
  poses: {
    list: (): Promise<SavedPose[]> => ipcRenderer.invoke("poses:list"),
    save: (request: { id?: string; title: string; pose: PoseConditioning }): Promise<SavedPose[]> => ipcRenderer.invoke("poses:save", request),
    delete: (id: string): Promise<SavedPose[]> => ipcRenderer.invoke("poses:delete", id),
    searchOpenPoses: (query: string): Promise<OpenPoseCatalogItem[]> => ipcRenderer.invoke("poses:search-openposes", query),
    getOpenPose: (id: string): Promise<OpenPoseAsset> => ipcRenderer.invoke("poses:get-openpose", id),
    importImage: (): Promise<PoseImageImport | null> => ipcRenderer.invoke("poses:import-image"),
    importRemoteImage: (request: { url: string; sourcePageUrl?: string }): Promise<PoseImageImport> => ipcRenderer.invoke("poses:import-remote-image", request),
    extractImage: (request: PoseExtractionRequest): Promise<PoseExtractionResult> => ipcRenderer.invoke("poses:extract-image", request),
    preprocessorStatus: (comfyUrl?: string): Promise<PosePreprocessorStatus> => ipcRenderer.invoke("poses:preprocessor-status", comfyUrl),
    analyze: (request: PoseVisionRequest): Promise<PoseVisionResult> => ipcRenderer.invoke("poses:analyze", request)
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
    updateThumbnail: (request: UpdateLoraThumbnailRequest): Promise<LoraTrainingProject[]> =>
      ipcRenderer.invoke("lora-projects:update-thumbnail", request),
    start: (id: string): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:start", id),
    pause: (id: string): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:pause", id),
    stop: (id: string): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:stop", id),
    delete: (request: string | { id: string; deleteOutputFile?: boolean }): Promise<LoraTrainingProject[]> =>
      ipcRenderer.invoke("lora-projects:delete", request),
    importFile: (): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:import"),
    downloadCivitai: (request: DownloadCivitaiLoraProjectRequest): Promise<LoraTrainingProject[]> =>
      ipcRenderer.invoke("lora-projects:download-civitai", request),
    clear: (): Promise<LoraTrainingProject[]> => ipcRenderer.invoke("lora-projects:clear")
  },
  loraCaptions: {
    get: (projectId: string): Promise<LoraCaptionWorkspace> => ipcRenderer.invoke("lora-captions:get", projectId),
    save: (request: SaveLoraCaptionWorkspaceRequest): Promise<LoraCaptionWorkspace> => ipcRenderer.invoke("lora-captions:save", request)
  },
  promptHistory: {
    list: (): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:list"),
    save: (request: SavePromptHistoryRequest): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:save", request),
    updateMemo: (request: UpdatePromptHistoryMemoRequest): Promise<PromptHistoryEntry[]> =>
      ipcRenderer.invoke("prompt-history:update-memo", request),
    delete: (request: { id: string }): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:delete", request),
    clear: (): Promise<PromptHistoryEntry[]> => ipcRenderer.invoke("prompt-history:clear")
  },
  app: {
    openPath: (path: string): Promise<string> => ipcRenderer.invoke("app:open-path", path),
    openExternal: (url: string): Promise<void> => ipcRenderer.invoke("app:open-external", url)
  }
};

contextBridge.exposeInMainWorld("illustMK", api);

export type IllustMKApi = typeof api;
