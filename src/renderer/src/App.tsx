import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Boxes,
  Bot,
  CheckCircle2,
  CircleAlert,
  Download,
  FolderOpen,
  ImageIcon,
  Pause,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Save,
  Server,
  Square,
  TerminalSquare,
  Trash2,
  WandSparkles,
  X
} from "lucide-react";
import type {
  CivitaiModelCard,
  DownloadRequest,
  GenerationLora,
  HuggingFaceLlmModelCard,
  IllustrationTag,
  LlmChatMessage,
  LoraCaptionCustomTag,
  LoraCaptionTag,
  LoraCaptionWorkspace,
  LoraTrainingRunParams,
  LoraTrainingProject,
  OllamaStatus,
  PromptEnhanceMode,
  PromptHistoryEntry,
  RuntimePaths,
  RuntimeStatus,
  TestGenerationRequest,
  TestGenerationResult
} from "../../main/runtime";
import "./styles/app.css";

type ActionState = {
  busy: boolean;
  message: string;
  error: string | null;
};

type LogLine = {
  id: string;
  level: "info" | "stdout" | "stderr";
  line: string;
  time: string;
};

const defaultComfyUrl = "http://127.0.0.1:8188";
const defaultBaseFilters = ["SDXL", "Pony", "Illustrious", "Animagine", "Anima"];
const allBaseFilters = [
  { name: "SDXL", status: "対応" },
  { name: "Pony", status: "対応" },
  { name: "Illustrious", status: "対応" },
  { name: "Animagine", status: "対応" },
  { name: "Anima", status: "対応候補" },
  { name: "Krea", status: "現在は非対応" },
  { name: "Flux", status: "現在は非対応" },
  { name: "Qwen Image", status: "現在は非対応" }
];
const recommendedLlmModels = [
  { name: "qwen2.5:3b-instruct", source: "Ollama", size: "軽量", note: "まず試す用。日本語もそこそこ。" },
  { name: "qwen2.5:7b-instruct", source: "Ollama", size: "標準", note: "日本語prompt整形の本命寄り。" },
  { name: "gemma3:4b", source: "Ollama", size: "軽量", note: "4B級の候補。応答が軽い。" },
  {
    name: "hf.co/soyaakinohara/Spark-X2.5-4B-Heretic-JP-GGUF",
    source: "Hugging Face GGUF",
    size: "軽量",
    note: "日本語性能狙い。Ollamaのhf.co指定で取得を試す候補。"
  }
];

type LoraTrainingPreset = "fast" | "normal" | "quality" | "custom";

const loraTrainingPresets: Record<Exclude<LoraTrainingPreset, "custom">, LoraTrainingRunParams> = {
  fast: { resolution: 512, repeats: 1, epochs: 1, networkDim: 4, learningRate: "1e-4" },
  normal: { resolution: 1024, repeats: 5, epochs: 5, networkDim: 16, learningRate: "1e-4" },
  quality: { resolution: 1024, repeats: 10, epochs: 10, networkDim: 32, learningRate: "1e-4" }
};

const loraPresetLabels: Record<LoraTrainingPreset, string> = {
  fast: "最速",
  normal: "通常",
  quality: "最高精度",
  custom: "カスタム"
};

function App() {
  const [paths, setPaths] = useState<RuntimePaths | null>(null);
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [comfyUrl, setComfyUrl] = useState(defaultComfyUrl);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadFileName, setDownloadFileName] = useState("");
  const [downloadSha, setDownloadSha] = useState("");
  const [vaeUrl, setVaeUrl] = useState("https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors");
  const [clipUrl, setClipUrl] = useState("");
  const [selectedVae, setSelectedVae] = useState("");
  const [selectedClip, setSelectedClip] = useState("");
  const [modelAddMode, setModelAddMode] = useState<"recommend" | "search" | "url">("recommend");
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogModels, setCatalogModels] = useState<CivitaiModelCard[]>([]);
  const [catalogPeriod, setCatalogPeriod] = useState<"Month" | "AllTime">("Month");
  const [baseFilters, setBaseFilters] = useState<string[]>(defaultBaseFilters);
  const [civitaiApiKey, setCivitaiApiKey] = useState("");
  const [huggingFaceApiKey, setHuggingFaceApiKey] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("http://127.0.0.1:11434");
  const [ollamaModel, setOllamaModel] = useState("qwen2.5:3b-instruct");
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);
  const [llmCatalogQuery, setLlmCatalogQuery] = useState("");
  const [hfLlmModels, setHfLlmModels] = useState<HuggingFaceLlmModelCard[]>([]);
  const [llmPrompt, setLlmPrompt] = useState("");
  const [llmChatHistory, setLlmChatHistory] = useState<LlmChatMessage[]>([]);
  const [illustrationTags, setIllustrationTags] = useState<IllustrationTag[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [tagEditingId, setTagEditingId] = useState<string | null>(null);
  const [tagName, setTagName] = useState("");
  const [tagEnglishName, setTagEnglishName] = useState("");
  const [tagPrompt, setTagPrompt] = useState("");
  const [tagJapaneseDescription, setTagJapaneseDescription] = useState("");
  const [tagEnglishDescription, setTagEnglishDescription] = useState("");
  const [japanesePrompt, setJapanesePrompt] = useState("");
  const [promptEnhanceMode, setPromptEnhanceMode] = useState<PromptEnhanceMode>("balanced-detail");
  const [promptNotes, setPromptNotes] = useState("");
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [selectedLoras, setSelectedLoras] = useState<GenerationLora[]>([]);
  const [prompt, setPrompt] = useState("masterpiece, best quality, 1girl, anime style, soft light");
  const [negativePrompt, setNegativePrompt] = useState("low quality, worst quality, blurry, bad anatomy");
  const [lastImage, setLastImage] = useState<string | null>(null);
  const [generationResult, setGenerationResult] = useState<TestGenerationResult | null>(null);
  const [loraCompareResult, setLoraCompareResult] = useState<{ seed: number; withoutLora: TestGenerationResult; withLora: TestGenerationResult } | null>(null);
  const [imageSizePreset, setImageSizePreset] = useState("768x1024");
  const [generationWidth, setGenerationWidth] = useState(768);
  const [generationHeight, setGenerationHeight] = useState(1024);
  const [seedMode, setSeedMode] = useState<"random" | "fixed">("random");
  const [seedValue, setSeedValue] = useState(-1);
  const [promptHistory, setPromptHistory] = useState<PromptHistoryEntry[]>([]);
  const [promptHistoryOpen, setPromptHistoryOpen] = useState(false);
  const [imagePreviewOpen, setImagePreviewOpen] = useState(false);
  const [loraProjects, setLoraProjects] = useState<LoraTrainingProject[]>([]);
  const [selectedLoraProjectId, setSelectedLoraProjectId] = useState<string | null>(null);
  const [loraCreateDialogOpen, setLoraCreateDialogOpen] = useState(false);
  const [loraProjectDialogMode, setLoraProjectDialogMode] = useState<"create" | "edit">("create");
  const [editingLoraProjectId, setEditingLoraProjectId] = useState<string | null>(null);
  const [loraName, setLoraName] = useState("");
  const [loraTriggerWord, setLoraTriggerWord] = useState("");
  const [loraAdditionalTraining, setLoraAdditionalTraining] = useState(false);
  const [loraCatalogOpen, setLoraCatalogOpen] = useState(false);
  const [loraCatalogQuery, setLoraCatalogQuery] = useState("");
  const [loraCatalogModels, setLoraCatalogModels] = useState<CivitaiModelCard[]>([]);
  const [loraCatalogPeriod, setLoraCatalogPeriod] = useState<"Month" | "AllTime">("Month");
  const [loraTrainingPreset, setLoraTrainingPreset] = useState<LoraTrainingPreset>("fast");
  const [loraResolution, setLoraResolution] = useState(1024);
  const [loraRepeats, setLoraRepeats] = useState(10);
  const [loraEpochs, setLoraEpochs] = useState(10);
  const [loraNetworkDim, setLoraNetworkDim] = useState(16);
  const [loraLearningRate, setLoraLearningRate] = useState("1e-4");
  const [captionWorkspace, setCaptionWorkspace] = useState<LoraCaptionWorkspace | null>(null);
  const [captionTags, setCaptionTags] = useState<LoraCaptionTag[]>([]);
  const [captionImages, setCaptionImages] = useState<LoraCaptionWorkspace["images"]>([]);
  const [selectedCaptionTagId, setSelectedCaptionTagId] = useState<string | null>(null);
  const [captionTagEditingId, setCaptionTagEditingId] = useState<string | null>(null);
  const [captionTagJapanese, setCaptionTagJapanese] = useState("");
  const [captionTagEnglish, setCaptionTagEnglish] = useState("");
  const [captionImageEditFileName, setCaptionImageEditFileName] = useState<string | null>(null);
  const [captionImageTagEditingId, setCaptionImageTagEditingId] = useState<string | null>(null);
  const [captionImageTagJapanese, setCaptionImageTagJapanese] = useState("");
  const [captionImageTagEnglish, setCaptionImageTagEnglish] = useState("");
  const [captionPreviewImage, setCaptionPreviewImage] = useState<LoraCaptionWorkspace["images"][number] | null>(null);
  const [action, setAction] = useState<ActionState>({ busy: false, message: "待機中", error: null });
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [autoFollowLogs, setAutoFollowLogs] = useState(true);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [mainTab, setMainTab] = useState<"environment" | "llm" | "lora" | "generate">("environment");
  const [runtimeTab, setRuntimeTab] = useState<"macos" | "docker">("macos");
  const [generateTab, setGenerateTab] = useState<"prompt" | "lora">("prompt");
  const logViewRef = useRef<HTMLDivElement | null>(null);
  const currentOperationId = useRef<string | null>(null);

  const selectedCheckpointInfo = useMemo(
    () => status?.checkpoints.find((checkpoint) => checkpoint.name === selectedCheckpoint) ?? null,
    [selectedCheckpoint, status?.checkpoints]
  );
  const selectedTags = useMemo(
    () => illustrationTags.filter((tag) => selectedTagIds.includes(tag.id)),
    [illustrationTags, selectedTagIds]
  );
  const selectedLoraProject = useMemo(
    () => loraProjects.find((project) => project.id === selectedLoraProjectId) ?? loraProjects[0] ?? null,
    [loraProjects, selectedLoraProjectId]
  );
  const activeLoraRunParams: LoraTrainingRunParams = useMemo(
    () =>
      loraTrainingPreset === "custom"
        ? {
            resolution: loraResolution,
            repeats: loraRepeats,
            epochs: loraEpochs,
            networkDim: loraNetworkDim,
            learningRate: loraLearningRate
          }
        : loraTrainingPresets[loraTrainingPreset],
    [loraEpochs, loraLearningRate, loraNetworkDim, loraRepeats, loraResolution, loraTrainingPreset]
  );
  const selectedProjectOutputExists = Boolean(
    selectedLoraProject && status?.loras.some((file) => file.name === selectedLoraProject.outputName)
  );
  const effectivePrompt = prompt;
  const selectedCheckpointGenerationReady = Boolean(
    selectedCheckpointInfo && (selectedCheckpointInfo.hasClip || selectedClip) && (selectedCheckpointInfo.hasVae || selectedVae)
  );
  const selectedCheckpointCompatibilityLabel = selectedCheckpointInfo
    ? [
        selectedCheckpointInfo.hasClip ? "CLIP内蔵" : selectedClip ? `外部CLIP: ${selectedClip}` : "外部CLIPが必要",
        selectedCheckpointInfo.hasVae ? "VAE内蔵" : selectedVae ? `外部VAE: ${selectedVae}` : "外部VAEが必要"
      ].join(" / ")
    : "未選択";
  const canGenerate = Boolean(status?.comfyReachable && selectedCheckpointGenerationReady && !action.busy);
  const canDownloadLlm = Boolean(ollamaStatus?.reachable && ollamaStatus.dataLinked && ollamaModel.trim() && !action.busy);
  const canSendLlmPrompt = Boolean(ollamaStatus?.reachable && ollamaStatus.dataLinked && ollamaModel.trim() && llmPrompt.trim() && !action.busy);
  const llmPromptDisabledReason = useMemo(() => {
    if (action.busy) return "処理中です。";
    if (!ollamaStatus?.reachable) return "Ollama APIが未接続です。";
    if (!ollamaStatus.dataLinked) return "Ollama保存先がDataPathに切替わっていません。";
    if (!ollamaModel.trim()) return "LLMモデルを選択してください。";
    if (!llmPrompt.trim()) return "LLMプロンプトを入力してください。";
    return "送信できます。";
  }, [action.busy, llmPrompt, ollamaModel, ollamaStatus]);
  const canEnhancePrompt = Boolean(ollamaStatus?.reachable && ollamaStatus.dataLinked && ollamaModel.trim() && japanesePrompt.trim() && !action.busy);
  const enhancePromptDisabledReason = useMemo(() => {
    if (action.busy) return "処理中です。";
    if (!ollamaStatus?.reachable) return "Ollama APIが未接続です。";
    if (!ollamaStatus.dataLinked) return "Ollama保存先がDataPathに切替わっていません。";
    if (!ollamaModel.trim()) return "LLMモデルを選択してください。";
    if (!japanesePrompt.trim()) return "日本語プロンプトを入力してください。";
    return "変換できます。";
  }, [action.busy, japanesePrompt, ollamaModel, ollamaStatus]);
  const filteredLlmModels = useMemo(() => {
    const query = llmCatalogQuery.trim().toLowerCase();
    if (!query) return recommendedLlmModels;
    return recommendedLlmModels.filter((model) =>
      [model.name, model.source, model.size, model.note].some((value) => value.toLowerCase().includes(query))
    );
  }, [llmCatalogQuery]);
  const captionTagById = useMemo(() => new Map(captionTags.map((tag) => [tag.id, tag])), [captionTags]);
  const captionImageEditing = useMemo(
    () => captionImages.find((image) => image.fileName === captionImageEditFileName) ?? null,
    [captionImageEditFileName, captionImages]
  );

  const stateLabel = useMemo(() => {
    if (!status) return "未確認";
    if (!status.nativeInstalled) return "macOS未インストール";
    if (!status.comfyReachable) return "ComfyUI未接続";
    if (!status.checkpoints.length) return "モデル未追加";
    return "生成準備OK";
  }, [status]);

  useEffect(() => {
    void refreshStatus();
    void loadSettings();
    void loadLlmChatHistory();
    void loadIllustrationTags();
    void loadPromptHistory();
    void loadLoraProjects();
  }, []);

  useEffect(() => {
    return window.illustMK.runtime.onOperationLog((event) => {
      const progressMatch = event.line.match(/ダウンロード中: .+\((\d+(?:\.\d+)?)%\)/);
      if (progressMatch) {
        setDownloadProgress(Number(progressMatch[1]));
      }
      appendLog(event.level, event.line, event.id);
    });
  }, []);

  useEffect(() => {
    if (!autoFollowLogs) return;
    const logView = logViewRef.current;
    if (!logView) return;
    logView.scrollTop = logView.scrollHeight;
  }, [autoFollowLogs, logs]);

  useEffect(() => {
    if (status?.checkpoints.length && !selectedCheckpoint) {
      setSelectedCheckpoint(status.checkpoints.find((checkpoint) => checkpoint.simpleWorkflowReady)?.name ?? status.checkpoints[0].name);
    }
  }, [selectedCheckpoint, status?.checkpoints]);

  useEffect(() => {
    if (status?.vae.length && !selectedVae) {
      setSelectedVae(status.vae.find((file) => file.hasVae)?.name ?? status.vae[0].name);
    }
  }, [selectedVae, status?.vae]);

  useEffect(() => {
    if (status?.clip.length && !selectedClip) {
      setSelectedClip(status.clip.find((file) => file.hasClip)?.name ?? status.clip[0].name);
    }
  }, [selectedClip, status?.clip]);

  useEffect(() => {
    if (!loraProjects.length) {
      setSelectedLoraProjectId(null);
      return;
    }
    if (!selectedLoraProjectId || !loraProjects.some((project) => project.id === selectedLoraProjectId)) {
      setSelectedLoraProjectId(loraProjects[0].id);
    }
  }, [loraProjects, selectedLoraProjectId]);

  async function runAction<T>(message: string, task: () => Promise<T>, after?: (value: T) => void): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", message, id);
    setAction({ busy: true, message, error: null });
    setDownloadProgress(null);
    try {
      const value = await task();
      after?.(value);
      appendLog("info", "完了しました。", id);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
    } catch (error) {
      appendLog("stderr", error instanceof Error ? error.message : String(error), id);
      setAction({
        busy: false,
        message: "失敗しました",
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  async function runStreamOperation(
    operation: string,
    message: string,
    options: { projectId?: string; sourceLora?: string; runParams?: LoraTrainingRunParams } = {}
  ): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", message, id);
    setAction({ busy: true, message, error: null });
    setDownloadProgress(null);
    try {
      const result = await window.illustMK.runtime.runOperation({ id, operation, ...options });
      appendLog("info", result, id);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
      await refreshOllamaStatus();
      await loadLoraProjects();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      const stopped = operation === "lora-train" && errorText.includes("停止されました");
      setAction({ busy: false, message: stopped ? "停止しました" : "失敗しました", error: stopped ? null : errorText });
      await refreshStatus();
      await refreshOllamaStatus();
      await loadLoraProjects();
    }
  }

  async function runOllamaPull(modelName = ollamaModel): Promise<void> {
    if (!ollamaStatus?.reachable || !ollamaStatus.dataLinked) {
      const errorText = !ollamaStatus?.dataLinked
        ? "Ollama保存先がDataPathに向いていません。先に「起動」を実行してください。"
        : "Ollama APIに未接続です。先に「起動」を実行してください。";
      appendLog("stderr", errorText);
      setAction({ busy: false, message: "Ollama未接続", error: errorText });
      return;
    }
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", `${modelName} をダウンロード中`, id);
    setAction({ busy: true, message: "Ollamaモデルをpull中", error: null });
    setDownloadProgress(null);
    try {
      setOllamaModel(modelName);
      const result = await window.illustMK.runtime.runOperation({ id, operation: "ollama-pull", model: modelName, ollamaUrl });
      appendLog("info", result, id);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshOllamaStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
      await refreshOllamaStatus();
    }
  }

  async function loadCivitaiModels(mode: "recommend" | "search"): Promise<void> {
    await runAction(
      mode === "search" ? "Civitaiを検索中" : "Civitai人気Top10を取得中",
      () =>
        window.illustMK.runtime.searchCivitai({
          query: mode === "search" ? catalogQuery : undefined,
          sort: "Most Downloaded",
          period: catalogPeriod,
          limit: 10,
          nsfw: false,
          baseModels: baseFilters
        }),
      (models) => {
        setCatalogModels(models);
        appendLog("info", `${models.length}件のモデルを取得しました。`);
      }
    );
  }

  async function downloadCivitaiModel(model: CivitaiModelCard): Promise<void> {
    await runDownloadAction(`${model.name} をダウンロード中`, {
      url: model.downloadUrl,
      targetType: "checkpoints",
      fileName: model.fileName,
      sha256: model.sha256,
      source: "civitai"
    }, () => {
      setSelectedCheckpoint(model.fileName);
    });
  }

  async function searchLoraCatalog(): Promise<void> {
    await runAction(
      "Civitai LoRAを検索中",
      () =>
        window.illustMK.runtime.searchCivitai({
          query: loraCatalogQuery.trim() || undefined,
          sort: "Most Downloaded",
          period: loraCatalogPeriod,
          limit: 20,
          nsfw: false,
          type: "LORA"
        }),
      (models) => {
        setLoraCatalogModels(models);
        appendLog("info", `${models.length}件のLoRAを取得しました。`);
      }
    );
  }

  async function downloadCivitaiLora(model: CivitaiModelCard): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", `${model.name} をLoRA Projectへ追加中`, id);
    setAction({ busy: true, message: "Civitai LoRAを追加中", error: null });
    setDownloadProgress(0);
    try {
      const projects = await window.illustMK.loraProjects.downloadCivitai({ model, operationId: id });
      setLoraProjects(projects);
      setSelectedLoraProjectId(projects[0]?.id ?? null);
      setDownloadProgress(100);
      appendLog("info", "LoRA Projectへ追加しました。", id);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      setDownloadProgress(null);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
      await refreshStatus();
    } finally {
      currentOperationId.current = null;
    }
  }

  async function runDownloadAction(message: string, request: DownloadRequest, after?: () => void): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", message, id);
    setAction({ busy: true, message, error: null });
    setDownloadProgress(0);
    try {
      await window.illustMK.runtime.downloadModel({ ...request, operationId: id });
      after?.();
      setDownloadProgress(100);
      appendLog("info", "ダウンロードが完了しました。", id);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      setDownloadProgress(null);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
      await refreshStatus();
    }
  }

  async function deleteCheckpoint(fileName: string): Promise<void> {
    if (!window.confirm(`${fileName} を削除します。元に戻せません。`)) return;
    await runAction(`${fileName} を削除中`, () =>
      window.illustMK.runtime.deleteModel({ targetType: "checkpoints", fileName }), () => {
        if (selectedCheckpoint === fileName) {
          setSelectedCheckpoint("");
        }
      }
    );
  }

  async function installVae(): Promise<void> {
    const fileName = vaeUrl.split("/").pop()?.split("?")[0] || "external.vae.safetensors";
    await runDownloadAction("VAEをインストール中", {
      url: vaeUrl.trim(),
      targetType: "vae",
      fileName,
      source: vaeUrl.includes("civitai.com") ? "civitai" : "direct"
    }, () => {
      setSelectedVae(fileName);
    });
  }

  async function deleteSelectedVae(): Promise<void> {
    if (!selectedVae) return;
    if (!window.confirm(`${selectedVae} を削除します。元に戻せません。`)) return;
    await runAction(`${selectedVae} を削除中`, () =>
      window.illustMK.runtime.deleteModel({ targetType: "vae", fileName: selectedVae }), () => {
        setSelectedVae("");
      }
    );
    await refreshStatus();
  }

  async function installClip(): Promise<void> {
    const fileName = clipUrl.split("/").pop()?.split("?")[0] || "external_clip.safetensors";
    await runDownloadAction("CLIP/text encoderをインストール中", {
      url: clipUrl.trim(),
      targetType: "clip",
      fileName,
      source: clipUrl.includes("civitai.com") ? "civitai" : "direct"
    }, () => {
      setSelectedClip(fileName);
    });
  }

  async function deleteSelectedClip(): Promise<void> {
    if (!selectedClip) return;
    if (!window.confirm(`${selectedClip} を削除します。元に戻せません。`)) return;
    await runAction(`${selectedClip} を削除中`, () =>
      window.illustMK.runtime.deleteModel({ targetType: "clip", fileName: selectedClip }), () => {
        setSelectedClip("");
      }
    );
    await refreshStatus();
  }

  function appendLog(level: LogLine["level"], line: string, id = currentOperationId.current ?? "manual"): void {
    const cleanLine = stripAnsi(line);
    setLogs((current) => [
      ...current.slice(-300),
      {
        id,
        level,
        line: cleanLine,
        time: new Date().toLocaleTimeString("ja-JP", { hour12: false })
      }
    ]);
  }

  function stripAnsi(value: string): string {
    return value
      .replace(/\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g, "")
      .replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "");
  }

  function handleLogScroll(): void {
    const logView = logViewRef.current;
    if (!logView) return;
    const distanceFromBottom = logView.scrollHeight - logView.scrollTop - logView.clientHeight;
    setAutoFollowLogs(distanceFromBottom < 18);
  }

  function jumpToLatestLog(): void {
    const logView = logViewRef.current;
    if (!logView) return;
    setAutoFollowLogs(true);
    logView.scrollTop = logView.scrollHeight;
  }

  async function refreshStatus(): Promise<void> {
    const [nextPaths, nextStatus] = await Promise.all([
      window.illustMK.runtime.paths(),
      window.illustMK.runtime.status(comfyUrl)
    ]);
    setPaths(nextPaths);
    setStatus(nextStatus);
  }

  async function refreshOllamaStatus(): Promise<void> {
    setOllamaStatus(await window.illustMK.ollama.status(ollamaUrl));
  }

  async function loadSettings(): Promise<void> {
    const settings = await window.illustMK.settings.get();
    setCivitaiApiKey(settings.civitaiApiKey);
    setHuggingFaceApiKey(settings.huggingFaceApiKey);
    setOllamaUrl(settings.ollamaUrl);
    setOllamaModel(settings.ollamaModel);
    setOllamaStatus(await window.illustMK.ollama.status(settings.ollamaUrl));
  }

  async function saveSettings(): Promise<void> {
    await runAction("設定を保存中", () =>
      window.illustMK.settings.save({ civitaiApiKey, huggingFaceApiKey, ollamaUrl, ollamaModel })
    );
  }

  async function loadLlmChatHistory(): Promise<void> {
    setLlmChatHistory(await window.illustMK.ollama.chatHistory());
  }

  async function sendLlmPrompt(): Promise<void> {
    const promptText = llmPrompt;
    await runAction(
      "LLMへプロンプトを送信中",
      () => window.illustMK.ollama.chat({ ollamaUrl, model: ollamaModel, prompt: promptText }),
      (history) => {
        setLlmChatHistory(history);
        setLlmPrompt("");
      }
    );
  }

  async function clearLlmChat(): Promise<void> {
    if (!window.confirm("LLMチャット履歴を削除します。元に戻せません。")) return;
    await runAction("LLMチャット履歴を削除中", () => window.illustMK.ollama.clearChatHistory(), (history) => {
      setLlmChatHistory(history);
    });
  }

  async function searchHuggingFaceLlmModels(): Promise<void> {
    await runAction(
      "Hugging Face LLMを検索中",
      () => window.illustMK.runtime.searchHuggingFaceLlm({ query: llmCatalogQuery, limit: 20 }),
      (models) => {
        setHfLlmModels(models);
        appendLog("info", `${models.length}件のHF GGUFモデルを取得しました。`);
      }
    );
  }

  async function loadIllustrationTags(): Promise<void> {
    setIllustrationTags(await window.illustMK.tags.list());
  }

  async function loadPromptHistory(): Promise<void> {
    setPromptHistory(await window.illustMK.promptHistory.list());
  }

  async function loadLoraProjects(): Promise<void> {
    setLoraProjects(await window.illustMK.loraProjects.list());
  }

  async function clearPromptHistory(): Promise<void> {
    if (!window.confirm("プロンプト履歴を削除します。元に戻せません。")) return;
    await runAction("プロンプト履歴を削除中", () => window.illustMK.promptHistory.clear(), (history) => {
      setPromptHistory(history);
    });
  }

  function applyPromptHistory(entry: PromptHistoryEntry): void {
    setJapanesePrompt(entry.japanesePrompt ?? "");
    setPrompt(entry.prompt);
    setNegativePrompt(entry.negativePrompt);
    setSelectedCheckpoint(entry.checkpoint);
    setGenerationWidth(entry.width);
    setGenerationHeight(entry.height);
    setImageSizePreset(`${entry.width}x${entry.height}`);
    setSeedValue(entry.seed);
    setSeedMode(entry.seed >= 0 ? "fixed" : "random");
    setPromptHistoryOpen(false);
  }

  function updateImageSizePreset(value: string): void {
    setImageSizePreset(value);
    if (value === "custom") return;
    const [width, height] = value.split("x").map(Number);
    if (Number.isFinite(width) && Number.isFinite(height)) {
      setGenerationWidth(width);
      setGenerationHeight(height);
    }
  }

  function openNewTagDialog(): void {
    setTagEditingId(null);
    setTagName("");
    setTagEnglishName("");
    setTagPrompt("");
    setTagJapaneseDescription("");
    setTagEnglishDescription("");
    setTagDialogOpen(true);
  }

  function openEditTagDialog(tag: IllustrationTag): void {
    setTagEditingId(tag.id);
    setTagName(tag.name);
    setTagEnglishName(tag.englishName || tag.name);
    setTagPrompt(tag.sourcePrompt);
    setTagJapaneseDescription(tag.japaneseDescription);
    setTagEnglishDescription(tag.englishDescription);
    setTagDialogOpen(true);
  }

  async function runTagJapaneseGeneration(): Promise<void> {
    await runAction(
      "タグ説明文を生成中",
      () => window.illustMK.tags.generateJapanese({ ollamaUrl, model: ollamaModel, tagName, prompt: tagPrompt }),
      (description) => {
        setTagJapaneseDescription(description);
      }
    );
  }

  async function translateTagEnglishDescription(): Promise<void> {
    await runAction(
      "タグ説明文を英訳中",
      () =>
        window.illustMK.tags.translateEnglish({
          ollamaUrl,
          model: ollamaModel,
          japaneseDescription: tagJapaneseDescription
        }),
      (englishDescription) => {
        setTagEnglishDescription(englishDescription);
      }
    );
  }

  async function saveTagFromDialog(): Promise<void> {
    await runAction(
      "タグを保存中",
      async () => {
        return window.illustMK.tags.save({
          id: tagEditingId ?? undefined,
          name: tagName,
          englishName: tagEnglishName,
          sourcePrompt: tagPrompt,
          japaneseDescription: tagJapaneseDescription,
          englishDescription: tagEnglishDescription
        });
      },
      (tags) => {
        setIllustrationTags(tags);
        setTagDialogOpen(false);
      }
    );
  }

  async function deleteIllustrationTag(id: string): Promise<void> {
    if (!window.confirm("タグを削除します。元に戻せません。")) return;
    await runAction("タグを削除中", () => window.illustMK.tags.delete(id), (tags) => {
      setIllustrationTags(tags);
      setSelectedTagIds((current) => current.filter((tagId) => tagId !== id));
    });
  }

  function toggleIllustrationTag(id: string): void {
    setSelectedTagIds((current) => (current.includes(id) ? current.filter((tagId) => tagId !== id) : [...current, id]));
  }

  function toggleGenerationLora(name: string): void {
    setSelectedLoras((current) =>
      current.some((lora) => lora.name === name)
        ? current.filter((lora) => lora.name !== name)
        : [...current, { name, modelStrength: 0.8, clipStrength: 0.8 }]
    );
  }

  function updateGenerationLora(name: string, field: "modelStrength" | "clipStrength", value: number): void {
    setSelectedLoras((current) => current.map((lora) => (lora.name === name ? { ...lora, [field]: value } : lora)));
  }

  function openNewLoraDialog(): void {
    setLoraProjectDialogMode("create");
    setEditingLoraProjectId(null);
    setLoraName("");
    setLoraTriggerWord("");
    setLoraCreateDialogOpen(true);
  }

  function openEditLoraDialog(): void {
    if (!selectedLoraProject) return;
    setLoraProjectDialogMode("edit");
    setEditingLoraProjectId(selectedLoraProject.id);
    setLoraName(selectedLoraProject.name);
    setLoraTriggerWord(selectedLoraProject.triggerWord);
    setSelectedCheckpoint(selectedLoraProject.baseCheckpoint);
    setLoraCreateDialogOpen(true);
  }

  async function saveLoraProject(): Promise<void> {
    const targetProjectId = loraProjectDialogMode === "edit" ? editingLoraProjectId : null;
    await runAction(
      "LoRAプロジェクトを保存中",
      () =>
        window.illustMK.loraProjects.save({
          id: targetProjectId ?? undefined,
          name: loraName,
          triggerWord: loraTriggerWord,
          baseCheckpoint: selectedCheckpoint
        }),
      (projects) => {
        setLoraProjects(projects);
        setSelectedLoraProjectId(targetProjectId ?? projects[0]?.id ?? null);
        setEditingLoraProjectId(null);
        setLoraCreateDialogOpen(false);
      }
    );
  }

  async function deleteSelectedLoraProject(): Promise<void> {
    if (!selectedLoraProject) return;
    if (!window.confirm(`${selectedLoraProject.name} を削除します。Datasetを含むプロジェクトディレクトリを削除します。出力済みLoRAファイルは削除しません。`)) return;
    await runAction("LoRAプロジェクトを削除中", () => window.illustMK.loraProjects.delete(selectedLoraProject.id), (projects) => {
      setLoraProjects(projects);
      setSelectedLoraProjectId(projects[0]?.id ?? null);
    });
  }

  async function startLoraProject(id: string): Promise<void> {
    await runAction("LoRA学習前チェック中", () => window.illustMK.loraProjects.start(id), (projects) => {
      setLoraProjects(projects);
    });
  }

  async function trainLoraProject(id: string): Promise<void> {
    const sourceLora = loraAdditionalTraining ? selectedLoraProject?.outputName ?? "" : "";
    setLoraProjects((projects) =>
      projects.map((project) =>
        project.id === id
          ? { ...project, status: "running", message: "LoRA学習を実行中です。ログは下段に表示します。" }
          : project
      )
    );
    await runStreamOperation("lora-train", "LoRA学習を開始中", { projectId: id, sourceLora, runParams: activeLoraRunParams });
  }

  async function resumeLoraProject(id: string): Promise<void> {
    const sourceLora = loraAdditionalTraining ? selectedLoraProject?.outputName ?? "" : "";
    setLoraProjects((projects) =>
      projects.map((project) =>
        project.id === id
          ? { ...project, status: "running", message: "保存済みstateからLoRA学習を再開中です。" }
          : project
      )
    );
    await runStreamOperation("lora-resume", "LoRA学習を再開中", { projectId: id, sourceLora, runParams: activeLoraRunParams });
  }

  async function importLoraProjectFile(): Promise<void> {
    await runAction("LoRA safetensorsをインポート中", () => window.illustMK.loraProjects.importFile(), (projects) => {
      setLoraProjects(projects);
    });
    await refreshStatus();
  }

  async function pauseLoraProject(id: string): Promise<void> {
    appendLog("info", "LoRA学習を一時停止中");
    try {
      const projects = await window.illustMK.loraProjects.pause(id);
      setLoraProjects(projects);
      appendLog("info", "LoRA学習を一時停止しました。");
    } catch (error) {
      appendLog("stderr", error instanceof Error ? error.message : String(error));
    }
  }

  async function stopLoraProject(id: string): Promise<void> {
    appendLog("info", "LoRA学習を停止中");
    try {
      const projects = await window.illustMK.loraProjects.stop(id);
      setLoraProjects(projects);
      appendLog("info", "LoRA学習を停止しました。");
    } catch (error) {
      appendLog("stderr", error instanceof Error ? error.message : String(error));
    }
  }

  async function openCaptionEditor(projectId: string): Promise<void> {
    await runAction(
      "キャプション編集を開いています",
      () => window.illustMK.loraCaptions.get(projectId),
      (workspace) => {
        setCaptionWorkspace(workspace);
        setCaptionTags(workspace.tags);
        setCaptionImages(workspace.images);
        setSelectedCaptionTagId(workspace.tags[0]?.id ?? null);
        resetCaptionTagEditor();
      }
    );
  }

  function closeCaptionEditor(): void {
    setCaptionWorkspace(null);
    setCaptionTags([]);
    setCaptionImages([]);
    setSelectedCaptionTagId(null);
    resetCaptionTagEditor();
    closeCaptionImageTagEditor();
    setCaptionPreviewImage(null);
  }

  function resetCaptionTagEditor(): void {
    setCaptionTagEditingId(null);
    setCaptionTagJapanese("");
    setCaptionTagEnglish("");
  }

  function editCaptionTag(tag: LoraCaptionTag): void {
    setCaptionTagEditingId(tag.id);
    setCaptionTagJapanese(tag.japaneseName);
    setCaptionTagEnglish(tag.englishName);
  }

  function upsertCaptionTag(): void {
    const japaneseName = captionTagJapanese.trim();
    if (!japaneseName) return;
    const now = new Date().toISOString();
    if (captionTagEditingId) {
      setCaptionTags((current) =>
        current.map((tag) =>
          tag.id === captionTagEditingId
            ? { ...tag, japaneseName, englishName: captionTagEnglish.trim(), updatedAt: now }
            : tag
        )
      );
      setSelectedCaptionTagId(captionTagEditingId);
    } else {
      const tag: LoraCaptionTag = {
        id: crypto.randomUUID(),
        japaneseName,
        englishName: captionTagEnglish.trim(),
        createdAt: now,
        updatedAt: now
      };
      setCaptionTags((current) => [...current, tag]);
      setSelectedCaptionTagId(tag.id);
    }
    resetCaptionTagEditor();
  }

  function deleteCaptionTag(id: string): void {
    setCaptionTags((current) => current.filter((tag) => tag.id !== id));
    setCaptionImages((current) => current.map((image) => ({ ...image, tagIds: image.tagIds.filter((tagId) => tagId !== id) })));
    if (selectedCaptionTagId === id) setSelectedCaptionTagId(null);
    if (captionTagEditingId === id) resetCaptionTagEditor();
  }

  function addSelectedTagToCaptionImage(fileName: string): void {
    if (!selectedCaptionTagId) return;
    setCaptionImages((current) =>
      current.map((image) =>
        image.fileName === fileName && !image.tagIds.includes(selectedCaptionTagId)
          ? { ...image, tagIds: [...image.tagIds, selectedCaptionTagId] }
          : image
      )
    );
  }

  function removeTagFromCaptionImage(fileName: string, tagId: string): void {
    setCaptionImages((current) =>
      current.map((image) =>
        image.fileName === fileName ? { ...image, tagIds: image.tagIds.filter((currentTagId) => currentTagId !== tagId) } : image
      )
    );
  }

  function addSelectedTagToAllCaptionImages(): void {
    if (!selectedCaptionTagId) return;
    setCaptionImages((current) =>
      current.map((image) =>
        image.tagIds.includes(selectedCaptionTagId) ? image : { ...image, tagIds: [...image.tagIds, selectedCaptionTagId] }
      )
    );
  }

  function removeSelectedTagFromAllCaptionImages(): void {
    if (!selectedCaptionTagId) return;
    setCaptionImages((current) =>
      current.map((image) => ({ ...image, tagIds: image.tagIds.filter((tagId) => tagId !== selectedCaptionTagId) }))
    );
  }

  function openCaptionImageTagEditor(fileName: string): void {
    setCaptionImageEditFileName(fileName);
    resetCaptionImageTagFields();
  }

  function closeCaptionImageTagEditor(): void {
    setCaptionImageEditFileName(null);
    resetCaptionImageTagFields();
  }

  function resetCaptionImageTagFields(): void {
    setCaptionImageTagEditingId(null);
    setCaptionImageTagJapanese("");
    setCaptionImageTagEnglish("");
  }

  function editCaptionImageCustomTag(tag: LoraCaptionCustomTag): void {
    setCaptionImageTagEditingId(tag.id);
    setCaptionImageTagJapanese(tag.japaneseName);
    setCaptionImageTagEnglish(tag.englishName);
  }

  function upsertCaptionImageCustomTag(): void {
    if (!captionImageEditFileName) return;
    const japaneseName = captionImageTagJapanese.trim();
    if (!japaneseName) return;
    const now = new Date().toISOString();
    setCaptionImages((current) =>
      current.map((image) => {
        if (image.fileName !== captionImageEditFileName) return image;
        if (captionImageTagEditingId) {
          return {
            ...image,
            customTags: image.customTags.map((tag) =>
              tag.id === captionImageTagEditingId
                ? { ...tag, japaneseName, englishName: captionImageTagEnglish.trim(), updatedAt: now }
                : tag
            )
          };
        }
        return {
          ...image,
          customTags: [
            ...image.customTags,
            {
              id: crypto.randomUUID(),
              japaneseName,
              englishName: captionImageTagEnglish.trim(),
              createdAt: now,
              updatedAt: now
            }
          ]
        };
      })
    );
    resetCaptionImageTagFields();
  }

  function deleteCaptionImageCustomTag(fileName: string, tagId: string): void {
    setCaptionImages((current) =>
      current.map((image) =>
        image.fileName === fileName ? { ...image, customTags: image.customTags.filter((tag) => tag.id !== tagId) } : image
      )
    );
    if (captionImageTagEditingId === tagId) resetCaptionImageTagFields();
  }

  async function saveCaptionEditor(): Promise<void> {
    if (!captionWorkspace) return;
    await runAction(
      "キャプションを保存中",
      () =>
        window.illustMK.loraCaptions.save({
          projectId: captionWorkspace.projectId,
          ollamaUrl,
          model: ollamaModel,
          tags: captionTags,
          images: captionImages.map((image) => ({ fileName: image.fileName, tagIds: image.tagIds, customTags: image.customTags }))
        }),
      (workspace) => {
        setCaptionWorkspace(workspace);
        setCaptionTags(workspace.tags);
        setCaptionImages(workspace.images);
        closeCaptionEditor();
        void loadLoraProjects();
      }
    );
  }

  async function clearLoraProjectHistory(): Promise<void> {
    if (!window.confirm("LoRA学習ジョブ履歴を消去します。DatasetやLoRAファイルは削除しません。")) return;
    await runAction("LoRA学習ジョブ履歴を消去中", () => window.illustMK.loraProjects.clear(), (projects) => {
      setLoraProjects(projects);
    });
  }

  async function enhanceJapanesePrompt(): Promise<void> {
    await runAction(
      "日本語プロンプトを生成向けに変換中",
      () =>
        window.illustMK.ollama.enhancePrompt({
          ollamaUrl,
          model: ollamaModel,
          japanesePrompt,
          currentPrompt: prompt,
          currentNegativePrompt: negativePrompt,
          selectedTags: selectedTags.map((tag) => ({
            name: tag.name,
            englishName: tag.englishName || tag.name,
            englishDescription: tag.englishDescription
          })),
          mode: promptEnhanceMode
        }),
      (result) => {
        setPrompt(result.positivePrompt);
        setNegativePrompt(result.negativePrompt);
        setPromptNotes(result.notes);
      }
    );
  }

  function imageUrl(path: string): string {
    if (path.startsWith("data:")) return path;
    return `file://${path}`;
  }

  function toggleBaseFilter(filter: string): void {
    setBaseFilters((current) => (current.includes(filter) ? current.filter((item) => item !== filter) : [...current, filter]));
  }

  const downloadRequest: DownloadRequest = {
    url: downloadUrl.trim(),
    targetType: "checkpoints",
    fileName: downloadFileName.trim() || undefined,
    sha256: downloadSha.trim() || undefined
  };

  const testRequest: TestGenerationRequest = {
    prompt: effectivePrompt,
    negativePrompt,
    checkpoint: selectedCheckpoint,
    loras: selectedLoras,
    width: generationWidth,
    height: generationHeight,
    steps: 20,
    cfg: 7,
    seed: seedMode === "random" ? -1 : seedValue,
    comfyUrl,
    vae: selectedVae || undefined,
    clip: selectedClip || undefined
  };

  async function runLoraOnOffCompare(): Promise<void> {
    const compareSeed = seedMode === "fixed" && seedValue >= 0 ? seedValue : Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    setSeedMode("fixed");
    setSeedValue(compareSeed);
    await runAction(
      "同SeedでLoRA ON/OFF比較中",
      async () => {
        appendLog("info", `LoRA比較 Seed: ${compareSeed}`);
        appendLog("info", `LoRA比較 Positive Prompt: ${effectivePrompt}`);
        const baseRequest = { ...testRequest, seed: compareSeed };
        const withoutLora = await window.illustMK.runtime.testGenerate({ ...baseRequest, loras: [] });
        const withLora = await window.illustMK.runtime.testGenerate({ ...baseRequest, loras: selectedLoras });
        return { seed: compareSeed, withoutLora, withLora };
      },
      (result) => {
        setLoraCompareResult(result);
        setGenerationResult(result.withLora);
        if (result.withLora.imageDataUrl || result.withLora.imagePath) {
          setLastImage(result.withLora.imageDataUrl ?? result.withLora.imagePath ?? null);
        }
      }
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-mark">IM</div>
        <div>
          <h1>illustMK</h1>
          <p>AI Illustration Studio</p>
        </div>
        <div className={action.error ? "header-action error" : "header-action"}>
          {action.error ? <CircleAlert size={16} /> : <CheckCircle2 size={16} />}
          <div>
            <strong>{action.message}</strong>
            <span>{action.error ?? "ログは下段に集約"}</span>
          </div>
        </div>
        <div className="connection-pill">
          <span className={status?.comfyReachable ? "dot ready" : "dot"} />
          {stateLabel}
        </div>
      </header>

      <section className="panel global-status-panel">
        <StatusGrid status={status} ollamaStatus={ollamaStatus} runtimeTab={runtimeTab} />
      </section>

      <nav className="main-tabs">
        <button className={mainTab === "environment" ? "tab active" : "tab"} onClick={() => setMainTab("environment")}>
          環境
        </button>
        <button className={mainTab === "llm" ? "tab active" : "tab"} onClick={() => setMainTab("llm")}>
          チャット
        </button>
        <button className={mainTab === "lora" ? "tab active" : "tab"} onClick={() => setMainTab("lora")}>
          LoRA
        </button>
        <button className={mainTab === "generate" ? "tab active" : "tab"} onClick={() => setMainTab("generate")}>
          生成
        </button>
      </nav>

      {mainTab === "environment" ? (
        <section className="environment-layout">
          <div className="environment-stack setup-stack">
            <section className="panel setup-panel">
              <PanelTitle icon={<Server size={18} />} title="ComfyUI" />
              <div className="runtime-tabs">
                <button className={runtimeTab === "macos" ? "runtime-tab active" : "runtime-tab"} onClick={() => setRuntimeTab("macos")}>
                  macOS <span>{status?.activeRuntime === "macos" ? "ACTIVE" : ""}</span>
                </button>
                <button className={runtimeTab === "docker" ? "runtime-tab active" : "runtime-tab"} onClick={() => setRuntimeTab("docker")}>
                  Docker <span>LEGACY</span>
                </button>
              </div>
              <div className="field">
                <label>ComfyUI API</label>
                <div className="inline">
                  <input value={comfyUrl} onChange={(event) => setComfyUrl(event.target.value)} />
                  <button className="icon-button" onClick={() => void refreshStatus()} title="再確認">
                    <RefreshCw size={17} />
                  </button>
                </div>
              </div>

              {runtimeTab === "macos" ? (
                <div className="button-grid">
                  <button onClick={() => void runStreamOperation("native-init", "macOSランタイムを初期化中")}>
                    <FolderOpen size={17} />
                    初期化
                  </button>
                  <button onClick={() => void runStreamOperation("native-install", "macOS ComfyUIをインストール中")}>
                    <Download size={17} />
                    インストール
                  </button>
                  <button onClick={() => void runStreamOperation("native-start", "macOS ComfyUIを開始中")}>
                    <Play size={17} />
                    開始
                  </button>
                  <button onClick={() => void runStreamOperation("native-stop", "macOS ComfyUIを停止中")}>
                    <Square size={17} />
                    停止
                  </button>
                  <button onClick={() => void runStreamOperation("native-logs", "macOS ComfyUIログを取得中")}>
                    <TerminalSquare size={17} />
                    ログ
                  </button>
                </div>
              ) : (
                <div className="button-grid">
                  <button onClick={() => void runStreamOperation("logs", "Docker ComfyUIログを取得中")}>
                    <TerminalSquare size={17} />
                    Dockerログ
                  </button>
                  <button onClick={() => void runStreamOperation("docker-cleanup", "Docker資産を後始末中")}>
                    <Trash2 size={17} />
                    後始末
                  </button>
                </div>
              )}

              <div className="paths">
                <p>アプリ: /Users/kds/Projects/illustMK</p>
                <p>ランタイム: {paths?.root ?? "/Volumes/ARC-8050T2/illustMK"}</p>
                <p>macOS ComfyUI: {paths?.nativeComfy ?? "未初期化"}</p>
              </div>
            </section>

            <section className="panel lora-env-panel">
              <PanelTitle icon={<Boxes size={18} />} title="LoRA" />
              <div className="button-grid">
                <button onClick={() => void runStreamOperation("lora-install", "LoRA学習環境をインストール中")}>
                  <Download size={17} />
                  インストール
                </button>
                <button onClick={() => void loadLoraProjects()}>
                  <RefreshCw size={17} />
                  再読込
                </button>
              </div>
              <div className="paths">
                <p>sd-scripts: {paths?.loraScripts ?? "/Volumes/ARC-8050T2/illustMK/lora-runtime/sd-scripts"}</p>
                <p>venv: {paths?.loraVenv ?? "/Volumes/ARC-8050T2/illustMK/lora-runtime/venv"}</p>
                <p>projects: {paths?.loraProjects ?? "/Volumes/ARC-8050T2/illustMK/lora-projects"}</p>
                <p>outputs: {paths?.loras ?? "/Volumes/ARC-8050T2/illustMK/models/loras"}</p>
              </div>
            </section>

            <section className="panel model-runtime-panel">
              <PanelTitle icon={<Download size={18} />} title="外部CLIP / VAE" />
              <div className="field">
                <label>VAE URL</label>
                <div className="inline">
                  <input value={vaeUrl} onChange={(event) => setVaeUrl(event.target.value)} />
                  <button className="icon-button" disabled={action.busy || !vaeUrl.trim()} onClick={() => void installVae()} title="VAEインストール">
                    <Download size={17} />
                  </button>
                </div>
              </div>
              <div className="field">
                <label>CLIP / text encoder URL</label>
                <div className="inline">
                  <input placeholder="必要なCLIP/text encoderのURL" value={clipUrl} onChange={(event) => setClipUrl(event.target.value)} />
                  <button className="icon-button" disabled={action.busy || !clipUrl.trim()} onClick={() => void installClip()} title="CLIPインストール">
                    <Download size={17} />
                  </button>
                </div>
              </div>
              <div className="two-columns">
                <div className="field">
                  <label>選択VAE</label>
                  <select value={selectedVae} onChange={(event) => setSelectedVae(event.target.value)}>
                    <option value="">未選択</option>
                    {status?.vae.map((file) => (
                      <option key={file.name} value={file.name}>
                        {file.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label>選択CLIP</label>
                  <select value={selectedClip} onChange={(event) => setSelectedClip(event.target.value)}>
                    <option value="">未選択</option>
                    {status?.clip.map((file) => (
                      <option key={file.name} value={file.name}>
                        {file.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="button-row">
                <button className="danger" disabled={action.busy || !selectedVae} onClick={() => void deleteSelectedVae()}>
                  <Trash2 size={15} />
                  VAE削除
                </button>
                <button className="danger" disabled={action.busy || !selectedClip} onClick={() => void deleteSelectedClip()}>
                  <Trash2 size={15} />
                  CLIP削除
                </button>
              </div>
            </section>
          </div>

          <section className="panel llm-panel">
            <PanelTitle icon={<Bot size={18} />} title="LLM / Ollama" />
            <div className="field">
              <label>Ollama API</label>
              <div className="inline">
                <input value={ollamaUrl} onChange={(event) => setOllamaUrl(event.target.value)} />
                <button className="icon-button" onClick={() => void refreshOllamaStatus()} title="再確認">
                  <RefreshCw size={17} />
                </button>
              </div>
            </div>
            <div className="field">
              <label>Hugging Face API Key</label>
              <div className="inline">
                <input
                  type="password"
                  placeholder="任意。HF検索/取得で必要な時に使用"
                  value={huggingFaceApiKey}
                  onChange={(event) => setHuggingFaceApiKey(event.target.value)}
                />
                <button className="icon-button" onClick={() => void saveSettings()} title="保存">
                  <CheckCircle2 size={17} />
                </button>
              </div>
            </div>
            <div className="field">
              <label>LLM Model</label>
              <input list="llm-models" value={ollamaModel} onChange={(event) => setOllamaModel(event.target.value)} />
              <datalist id="llm-models">
                {recommendedLlmModels.map((model) => (
                  <option key={model.name} value={model.name} />
                ))}
              </datalist>
            </div>
            <div className="button-grid">
              <button onClick={() => void runStreamOperation("ollama-install", "Ollama環境をインストール中")}>
                <Download size={17} />
                インストール
              </button>
              <button onClick={() => void runAction("Ollamaを起動中", () => window.illustMK.ollama.start(), () => void refreshOllamaStatus())}>
                <Play size={17} />
                起動
              </button>
              <button onClick={() => void runAction("Ollamaを停止中", () => window.illustMK.ollama.stop(), () => void refreshOllamaStatus())}>
                <Square size={17} />
                停止
              </button>
              <button onClick={() => void runOllamaPull()} disabled={!canDownloadLlm} title={canDownloadLlm ? "Download" : "先にOllamaを起動してください"}>
                <Download size={17} />
                Download
              </button>
              <button onClick={() => void saveSettings()}>
                <CheckCircle2 size={17} />
                保存
              </button>
              <button
                className="danger"
                onClick={() => {
                  if (window.confirm("Ollama.appをアンインストールします。モデル保存先は残します。")) {
                    void runStreamOperation("ollama-uninstall", "Ollama.appをアンインストール中");
                  }
                }}
              >
                <Trash2 size={17} />
                App削除
              </button>
              <button
                className="danger"
                onClick={() => {
                  if (window.confirm("OllamaのDL済みモデル保存先を削除します。元に戻せません。")) {
                    void runStreamOperation("ollama-delete-storage", "Ollamaモデル保存先を削除中");
                  }
                }}
              >
                <Trash2 size={17} />
                モデル全削除
              </button>
            </div>
            <div className="field catalog-search">
              <label>LLM Search</label>
              <div className="inline">
                <input placeholder="DanTagGen, qwen gguf, gemma gguf..." value={llmCatalogQuery} onChange={(event) => setLlmCatalogQuery(event.target.value)} />
                <button className="icon-button" disabled={action.busy || !llmCatalogQuery.trim()} onClick={() => void searchHuggingFaceLlmModels()} title="HF検索">
                  <RefreshCw size={17} />
                </button>
              </div>
            </div>
            {hfLlmModels.length ? (
              <div className="llm-catalog-list">
                {hfLlmModels.map((model) => (
                  <article className="llm-search-card" key={model.id}>
                    <button className="llm-search-select" onClick={() => setOllamaModel(model.pullName)}>
                      <strong>{model.id}</strong>
                      <span>
                        HF GGUF / DL {model.downloads.toLocaleString()} / {model.totalSizeLabel}
                      </span>
                    </button>
                    <p>{model.ggufFiles.slice(0, 4).map((file) => `${file.name} (${file.sizeLabel})`).join(", ") || "GGUF repo"}</p>
                    <button className="llm-download-button" disabled={!ollamaStatus?.reachable || !ollamaStatus.dataLinked || action.busy} onClick={() => void runOllamaPull(model.pullName)}>
                      <Download size={15} />
                      D O W N L O A D
                    </button>
                  </article>
                ))}
              </div>
            ) : null}
            <div className="model-list">
              <h2>おすすめ</h2>
            </div>
            <div className="llm-catalog-list">
              {filteredLlmModels.map((model) => (
                <article className="llm-catalog-card" key={model.name}>
                  <button className="llm-select-button" onClick={() => setOllamaModel(model.name)}>
                    <span>{model.name}</span>
                    <small>{model.source} / {model.size}</small>
                  </button>
                  <p>{model.note}</p>
                  <button disabled={!ollamaStatus?.reachable || !ollamaStatus.dataLinked || action.busy} onClick={() => void runOllamaPull(model.name)}>
                    <Download size={15} />
                    Download
                  </button>
                </article>
              ))}
            </div>
            <div className="paths">
              <p>保存先: {ollamaStatus?.modelPath ?? paths?.ollamaModels ?? "/Volumes/ARC-8050T2/illustMK/llm/ollama/models"}</p>
            </div>
            <div className="model-list">
              <h2>DL済みLLM</h2>
              {ollamaStatus?.models.length ? (
                ollamaStatus.models.map((model) => (
                  <div className="model-row" key={model.name}>
                    <button className={ollamaModel === model.name ? "model selected" : "model"} onClick={() => setOllamaModel(model.name)}>
                      <span>{model.name}</span>
                      <small>{model.sizeLabel}</small>
                    </button>
                    <button
                      className="icon-button danger"
                      disabled={action.busy}
                      onClick={() =>
                        void runAction(`${model.name} を削除中`, () =>
                          window.illustMK.ollama.deleteModel({ ollamaUrl, model: model.name }), () => void refreshOllamaStatus()
                        )
                      }
                      title="削除"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))
              ) : (
                <p className="empty">{ollamaStatus?.message ?? "Ollamaの状態を確認してください。"}</p>
              )}
            </div>
          </section>

          <section className="panel model-panel">
            <PanelTitle icon={<Download size={18} />} title="モデル追加 / 管理" />
            <div className="field">
              <label>Civitai API Key</label>
              <div className="inline">
                <input
                  type="password"
                  placeholder="任意。必要なモデル取得時に使用"
                  value={civitaiApiKey}
                  onChange={(event) => setCivitaiApiKey(event.target.value)}
                />
                <button className="icon-button" onClick={() => void saveSettings()} title="保存">
                  <CheckCircle2 size={17} />
                </button>
              </div>
            </div>
            <div className="model-mode-tabs">
              <button className={modelAddMode === "recommend" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelAddMode("recommend")}>
                おすすめTop10
              </button>
              <button className={modelAddMode === "search" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelAddMode("search")}>
                Civitai検索
              </button>
              <button className={modelAddMode === "url" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelAddMode("url")}>
                URL直接
              </button>
            </div>

            {modelAddMode !== "url" ? (
              <>
                <div className="two-columns">
                  <div className="field">
                    <label>期間</label>
                    <select value={catalogPeriod} onChange={(event) => setCatalogPeriod(event.target.value as "Month" | "AllTime")}>
                      <option value="Month">直近1ヶ月</option>
                      <option value="AllTime">累計</option>
                    </select>
                  </div>
                  {modelAddMode === "search" ? (
                    <div className="field">
                      <label>検索語</label>
                      <input placeholder="anime, illustrious..." value={catalogQuery} onChange={(event) => setCatalogQuery(event.target.value)} />
                    </div>
                  ) : (
                    <div className="field">
                      <label>条件</label>
                      <input value="Checkpoint / safetensors / NSFW除外" readOnly />
                    </div>
                  )}
                </div>
                <div className="filter-row">
                  <label>Base Model Filter</label>
                  <div className="filter-buttons">
                    {allBaseFilters.map((filter) => (
                      <button
                        key={filter.name}
                        className={baseFilters.includes(filter.name) ? "filter-chip active" : "filter-chip"}
                        onClick={() => toggleBaseFilter(filter.name)}
                      >
                        {filter.name}
                        <span>{filter.status}</span>
                      </button>
                    ))}
                    <button className="filter-chip" onClick={() => setBaseFilters(defaultBaseFilters)}>
                      リセット
                    </button>
                  </div>
                </div>
                <button className="primary" disabled={action.busy} onClick={() => void loadCivitaiModels(modelAddMode)}>
                  <Download size={17} />
                  {modelAddMode === "search" ? "検索" : "人気Top10を表示"}
                </button>
                <div className="catalog-list">
                  {catalogModels.map((model) => (
                    <article className="catalog-card" key={`${model.id}-${model.versionId}`}>
                      {model.imageUrl ? <img src={model.imageUrl} alt="" /> : <div className="catalog-no-image">No Image</div>}
                      <div>
                        <h3>{model.name}</h3>
                        <p>{model.creator} / {model.baseModel || "base不明"}</p>
                        <p>{Math.round(model.sizeKB / 1024)} MB / DL {model.downloadCount.toLocaleString()}</p>
                        <button disabled={action.busy} onClick={() => void downloadCivitaiModel(model)}>
                          <Download size={15} />
                          Download
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <>
                <div className="field">
                  <label>Checkpoint URL</label>
                  <input
                    placeholder="https://.../model.safetensors"
                    value={downloadUrl}
                    onChange={(event) => setDownloadUrl(event.target.value)}
                  />
                </div>
                <div className="two-columns">
                  <div className="field">
                    <label>保存名</label>
                    <input
                      placeholder="未指定ならURLから決定"
                      value={downloadFileName}
                      onChange={(event) => setDownloadFileName(event.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label>SHA256</label>
                    <input placeholder="任意" value={downloadSha} onChange={(event) => setDownloadSha(event.target.value)} />
                  </div>
                </div>
                <button
                  className="primary"
                  disabled={!downloadUrl.trim() || action.busy}
                  onClick={() => void runDownloadAction("モデルをダウンロード中", downloadRequest)}
                >
                  <Download size={17} />
                  Checkpointを保存
                </button>
              </>
            )}

            <div className="model-list">
              <h2>検出済みCheckpoint</h2>
              {status?.checkpoints.length ? (
                status.checkpoints.map((file) => (
                  <div className="model-row" key={file.name}>
                    <button
                      className={selectedCheckpoint === file.name ? "model selected" : "model"}
                      onClick={() => setSelectedCheckpoint(file.name)}
                    >
                      <span>{file.name}</span>
                      <small>{file.sizeLabel} / {file.compatibilityLabel}</small>
                    </button>
                    <button
                      className="icon-button danger"
                      disabled={action.busy}
                      onClick={() => void deleteCheckpoint(file.name)}
                      title="削除"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))
              ) : (
                <p className="empty">まだ checkpoint がありません。</p>
              )}
            </div>
          </section>
        </section>
      ) : mainTab === "llm" ? (
        <section className="llm-chat-layout">
          <section className="panel llm-chat-panel">
            <div className="chat-header">
              <PanelTitle icon={<Bot size={18} />} title="LLMチャット / ログ" />
              <div className="chat-meta">
                <span>{ollamaStatus?.reachable && ollamaStatus.dataLinked ? "接続OK" : "未接続"}</span>
                <span>{ollamaModel || "モデル未選択"}</span>
              </div>
            </div>
            <div className="chat-history">
              {llmChatHistory.length ? (
                llmChatHistory.map((message) => (
                  <article className={`chat-message ${message.role}`} key={message.id}>
                    <header>
                      <strong>{message.role === "user" ? "USER" : "LLM"}</strong>
                      <span>{message.model}</span>
                      <time>{new Date(message.createdAt).toLocaleString("ja-JP", { hour12: false })}</time>
                    </header>
                    <p>{message.content}</p>
                  </article>
                ))
              ) : (
                <div className="placeholder compact">
                  <Bot size={36} />
                  <p>LLMへの入力と結果がここに残ります。</p>
                </div>
              )}
            </div>
            <div className="chat-composer">
              <textarea
                placeholder="LLMへそのまま送るプロンプトを入力"
                value={llmPrompt}
                onChange={(event) => setLlmPrompt(event.target.value)}
              />
              <div className="chat-actions">
                <button className="primary" disabled={!canSendLlmPrompt} onClick={() => void sendLlmPrompt()}>
                  <Bot size={17} />
                  送信
                </button>
                <button onClick={() => void loadLlmChatHistory()}>
                  <RefreshCw size={17} />
                  再読込
                </button>
                <button className="danger" disabled={!llmChatHistory.length || action.busy} onClick={() => void clearLlmChat()}>
                  <Trash2 size={17} />
                  履歴削除
                </button>
              </div>
              <p className={canSendLlmPrompt ? "result-message ready" : "result-message"}>
                {llmPromptDisabledReason}
              </p>
            </div>
          </section>
        </section>
      ) : mainTab === "lora" ? (
        <section className="lora-layout">
          <section className="panel lora-project-panel lora-sidebar">
            <div className="lora-create-header">
              <PanelTitle icon={<Boxes size={18} />} title="LoRAプロジェクト" />
            </div>
            <div className="lora-project-list">
              {loraProjects.length ? (
                loraProjects.map((project) => (
                  <article
                    className={project.id === selectedLoraProject?.id ? "lora-project-card selected" : "lora-project-card"}
                    key={project.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedLoraProjectId(project.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") setSelectedLoraProjectId(project.id);
                    }}
                  >
                    <header>
                      <strong>{project.name}</strong>
                      <span>{project.status}</span>
                    </header>
                    <div className="lora-project-meta">
                      <span>Trigger: {project.triggerWord || "なし"}</span>
                      <span>Base: {project.baseCheckpoint}</span>
                      <span>Output: {project.outputName}</span>
                    </div>
                    <p>{project.message}</p>
                  </article>
                ))
              ) : (
                <p className="empty">まだLoRAプロジェクトがありません。</p>
              )}
            </div>
            <div className="lora-sidebar-actions">
              <button className="primary" disabled={action.busy} onClick={openNewLoraDialog}>
                <Plus size={17} />
                新規
              </button>
              <button disabled={action.busy || !selectedLoraProject || selectedLoraProject.status === "running"} onClick={openEditLoraDialog}>
                <Pencil size={17} />
                編集
              </button>
              <button disabled={action.busy} onClick={() => void importLoraProjectFile()}>
                <Plus size={17} />
                インポート
              </button>
              <button disabled={action.busy} onClick={() => setLoraCatalogOpen(true)}>
                <Download size={17} />
                カタログ
              </button>
              <button disabled={action.busy} onClick={() => void loadLoraProjects()}>
                <RefreshCw size={17} />
                更新
              </button>
              <button className="danger" disabled={action.busy || !selectedLoraProject || selectedLoraProject.status === "running"} onClick={() => void deleteSelectedLoraProject()}>
                <Trash2 size={17} />
                削除
              </button>
            </div>
          </section>
          <section className="panel lora-project-panel">
            <div className="lora-create-header">
              <PanelTitle icon={<TerminalSquare size={18} />} title="LoRA学習ジョブ" />
            </div>
            <div className="lora-control-panel">
              <div className="lora-settings-block">
                <PanelTitle icon={<TerminalSquare size={18} />} title="学習パラメータ" />
              </div>
              <button
                type="button"
                className={loraAdditionalTraining ? "toggle-button active" : "toggle-button"}
                disabled={!selectedLoraProject}
                onClick={() => setLoraAdditionalTraining((value) => !value)}
              >
                追加学習
              </button>
              {loraAdditionalTraining && !selectedProjectOutputExists ? (
                <p className="result-message">追加学習には選択中プロジェクトの出力LoRAが必要です。先に新規学習を完了してください。</p>
              ) : null}
              <div className="segmented lora-preset-tabs">
                {(["fast", "normal", "quality", "custom"] as LoraTrainingPreset[]).map((preset) => (
                  <button
                    key={preset}
                    className={loraTrainingPreset === preset ? "active" : ""}
                    onClick={() => setLoraTrainingPreset(preset)}
                  >
                    {loraPresetLabels[preset]}
                  </button>
                ))}
              </div>
              <div className="lora-control-summary">
                <span>
                  {activeLoraRunParams.resolution}px / repeats {activeLoraRunParams.repeats} / epochs {activeLoraRunParams.epochs} / dim {activeLoraRunParams.networkDim} / lr {activeLoraRunParams.learningRate}
                </span>
              </div>
              {loraTrainingPreset === "custom" ? (
                <div className="lora-train-grid">
                  <div className="field">
                    <label>Resolution</label>
                    <input type="number" min="512" step="64" value={loraResolution} onChange={(event) => setLoraResolution(Number(event.target.value))} />
                  </div>
                  <div className="field">
                    <label>Repeats</label>
                    <input type="number" min="1" value={loraRepeats} onChange={(event) => setLoraRepeats(Number(event.target.value))} />
                  </div>
                  <div className="field">
                    <label>Epochs</label>
                    <input type="number" min="1" value={loraEpochs} onChange={(event) => setLoraEpochs(Number(event.target.value))} />
                  </div>
                  <div className="field">
                    <label>Network Dim</label>
                    <input type="number" min="1" value={loraNetworkDim} onChange={(event) => setLoraNetworkDim(Number(event.target.value))} />
                  </div>
                  <div className="field">
                    <label>Learning Rate</label>
                    <input value={loraLearningRate} onChange={(event) => setLoraLearningRate(event.target.value)} />
                  </div>
                </div>
              ) : null}
              <div className="button-row">
                <button disabled={!selectedLoraProject} onClick={() => selectedLoraProject && void window.illustMK.app.openPath(selectedLoraProject.datasetPath)}>
                  <FolderOpen size={15} />
                  Datasetを開く
                </button>
                <button disabled={!selectedLoraProject} onClick={() => selectedLoraProject && void openCaptionEditor(selectedLoraProject.id)}>
                  <ImageIcon size={15} />
                  キャプション編集
                </button>
                <button disabled={action.busy || !selectedLoraProject} onClick={() => selectedLoraProject && void startLoraProject(selectedLoraProject.id)}>
                  <Play size={15} />
                  学習前チェック
                </button>
                <button className="primary" disabled={action.busy || !selectedLoraProject || selectedLoraProject.status === "running" || (loraAdditionalTraining && !selectedProjectOutputExists)} onClick={() => selectedLoraProject && void trainLoraProject(selectedLoraProject.id)}>
                  <WandSparkles size={15} />
                  学習開始
                </button>
                <button disabled={action.busy || !selectedLoraProject || selectedLoraProject.status === "running" || (loraAdditionalTraining && !selectedProjectOutputExists)} onClick={() => selectedLoraProject && void resumeLoraProject(selectedLoraProject.id)}>
                  <RefreshCw size={15} />
                  再開
                </button>
                <button disabled={!selectedLoraProject || selectedLoraProject.status !== "running"} onClick={() => selectedLoraProject && void pauseLoraProject(selectedLoraProject.id)}>
                  <Pause size={15} />
                  一時停止
                </button>
                <button className="danger" disabled={!selectedLoraProject || selectedLoraProject.status !== "running"} onClick={() => selectedLoraProject && void stopLoraProject(selectedLoraProject.id)}>
                  <Square size={15} />
                  停止
                </button>
              </div>
            </div>
          </section>
        </section>
      ) : (
        <section className="generate-layout">
          <section className="panel preview-panel">
            <PanelTitle icon={<ImageIcon size={18} />} title="テスト生成" />
            <div className="preview-stage" style={{ aspectRatio: `${generationWidth} / ${generationHeight}` }}>
              {lastImage ? (
                <button className="preview-image-button" type="button" onClick={() => setImagePreviewOpen(true)} title="拡大表示">
                  <img src={imageUrl(lastImage)} alt="Generated preview" />
                </button>
              ) : (
                <div className="placeholder">
                  <ImageIcon size={42} />
                  <p>生成結果がここに表示されます。</p>
                </div>
              )}
            </div>
          </section>

          <section className="panel generate-panel">
            <div className="runtime-tabs">
              <button className={generateTab === "prompt" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("prompt")}>
                プロンプト
              </button>
              <button className={generateTab === "lora" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("lora")}>
                LoRA
              </button>
            </div>
            {generateTab === "prompt" ? (
              <PromptTransformPanel
                japanesePrompt={japanesePrompt}
                setJapanesePrompt={setJapanesePrompt}
                prompt={prompt}
                setPrompt={setPrompt}
                negativePrompt={negativePrompt}
                setNegativePrompt={setNegativePrompt}
                promptNotes={promptNotes}
                promptEnhanceMode={promptEnhanceMode}
                setPromptEnhanceMode={setPromptEnhanceMode}
                canEnhancePrompt={canEnhancePrompt}
                disabledReason={enhancePromptDisabledReason}
                onEnhance={() => void enhanceJapanesePrompt()}
                illustrationTags={illustrationTags}
                selectedTagIds={selectedTagIds}
                onToggleTag={toggleIllustrationTag}
                onAddTag={openNewTagDialog}
                onEditTag={openEditTagDialog}
                onDeleteTag={(id) => void deleteIllustrationTag(id)}
              />
            ) : (
              <GenerationLoraPanel
                loras={status?.loras ?? []}
                selectedLoras={selectedLoras}
                onToggleLora={toggleGenerationLora}
                onUpdateLora={updateGenerationLora}
              />
            )}
            <div className="field">
              <label>Checkpoint</label>
              <select value={selectedCheckpoint} onChange={(event) => setSelectedCheckpoint(event.target.value)}>
                {status?.checkpoints.length ? (
                  status.checkpoints.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name} ({file.sizeLabel} / {file.compatibilityLabel})
                    </option>
                  ))
                ) : (
                  <option value="">未追加</option>
                )}
              </select>
            </div>
            <div className="two-columns">
              <div className="field">
                <label>外部VAE</label>
                <select value={selectedVae} onChange={(event) => setSelectedVae(event.target.value)}>
                  <option value="">未選択</option>
                  {status?.vae.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>外部CLIP</label>
                <select value={selectedClip} onChange={(event) => setSelectedClip(event.target.value)}>
                  <option value="">未選択</option>
                  {status?.clip.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="generation-controls">
              <div className="field">
                <label>画像サイズ</label>
                <select value={imageSizePreset} onChange={(event) => updateImageSizePreset(event.target.value)}>
                  <option value="768x1024">768 x 1024</option>
                  <option value="1024x1024">1024 x 1024</option>
                  <option value="1024x1536">1024 x 1536</option>
                  <option value="1536x1024">1536 x 1024</option>
                  <option value="custom">Custom</option>
                </select>
              </div>
              <div className="two-columns">
                <div className="field">
                  <label>Width</label>
                  <input type="number" min="256" step="64" value={generationWidth} onChange={(event) => setGenerationWidth(Number(event.target.value))} />
                </div>
                <div className="field">
                  <label>Height</label>
                  <input type="number" min="256" step="64" value={generationHeight} onChange={(event) => setGenerationHeight(Number(event.target.value))} />
                </div>
              </div>
              <div className="field">
                <label>Seed</label>
                <select value={seedMode} onChange={(event) => setSeedMode(event.target.value as "random" | "fixed")}>
                  <option value="random">Random (-1)</option>
                  <option value="fixed">Fixed</option>
                </select>
              </div>
              <div className="field">
                <label>Seed Value</label>
                <input type="number" disabled={seedMode === "random"} value={seedValue} onChange={(event) => setSeedValue(Number(event.target.value))} />
              </div>
            </div>
            <div className="prompt-inspector">
              <div className="tag-builder-header">
                <div>
                  <h2>ComfyUIへ渡す最終Positive Prompt</h2>
                </div>
                <button onClick={() => setPromptHistoryOpen(true)}>
                  <RefreshCw size={17} />
                  履歴
                </button>
              </div>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </div>

            <div className="button-row">
              <button
                className="generate"
                disabled={!canGenerate}
                onClick={() =>
                  void runAction(
                    "ComfyUIでテスト生成中",
                    async () => {
                      appendLog("info", `ComfyUI Positive Prompt: ${testRequest.prompt}`);
                      appendLog("info", `ComfyUI Negative Prompt: ${testRequest.negativePrompt}`);
                      appendLog("info", `ComfyUI LoRA: ${selectedLoras.length ? selectedLoras.map((lora) => `${lora.name} model=${lora.modelStrength} clip=${lora.clipStrength}`).join(", ") : "なし"}`);
                      appendLog("info", `ComfyUI Size/Seed: ${testRequest.width}x${testRequest.height} / ${testRequest.seed}`);
                      const result = await window.illustMK.runtime.testGenerate(testRequest);
                      const history = await window.illustMK.promptHistory.save({
                        ...testRequest,
                        japanesePrompt,
                        imagePath: result.imagePath,
                        imageDataUrl: result.imageDataUrl
                      });
                      return { result, history };
                    },
                    ({ result, history }) => {
                      setGenerationResult(result);
                      setPromptHistory(history);
                      if (result.imageDataUrl || result.imagePath) {
                        setLastImage(result.imageDataUrl ?? result.imagePath ?? null);
                      }
                    }
                  )
                }
              >
                <WandSparkles size={18} />
                GENERATE TEST
              </button>
              <button className="generate secondary-generate" disabled={!canGenerate || !selectedLoras.length} onClick={() => void runLoraOnOffCompare()}>
                <ImageIcon size={18} />
                LoRA ON/OFF比較
              </button>
            </div>
            {loraCompareResult ? (
              <div className="lora-compare-grid">
                <article>
                  <h3>LoRA OFF / seed {loraCompareResult.seed}</h3>
                  {loraCompareResult.withoutLora.imageDataUrl || loraCompareResult.withoutLora.imagePath ? (
                    <img src={imageUrl(loraCompareResult.withoutLora.imageDataUrl ?? loraCompareResult.withoutLora.imagePath ?? "")} alt="LoRA OFF" />
                  ) : null}
                </article>
                <article>
                  <h3>LoRA ON / seed {loraCompareResult.seed}</h3>
                  {loraCompareResult.withLora.imageDataUrl || loraCompareResult.withLora.imagePath ? (
                    <img src={imageUrl(loraCompareResult.withLora.imageDataUrl ?? loraCompareResult.withLora.imagePath ?? "")} alt="LoRA ON" />
                  ) : null}
                </article>
              </div>
            ) : null}
            {selectedCheckpointInfo && !selectedCheckpointGenerationReady ? (
              <p className="result-message">
                このCheckpointは追加ファイルが必要です。{selectedCheckpointCompatibilityLabel}
              </p>
            ) : null}
            {generationResult?.message && <p className="result-message">{generationResult.message}</p>}
          </section>
        </section>
      )}
      {tagDialogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel tag-modal">
            <div className="modal-header">
              <PanelTitle icon={<WandSparkles size={18} />} title={tagEditingId ? "タグ編集" : "タグ追加"} />
              <button className="icon-button" onClick={() => setTagDialogOpen(false)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="two-columns">
              <div className="field">
                <label>タグ名</label>
                <input placeholder="制服" value={tagName} onChange={(event) => setTagName(event.target.value)} />
              </div>
              <div className="field">
                <label>英名タグ名</label>
                <input placeholder="school_uniform" value={tagEnglishName} onChange={(event) => setTagEnglishName(event.target.value)} />
              </div>
            </div>
            <div className="two-columns">
              <div className="field">
                <label>LLM Model</label>
                <input value={ollamaModel} onChange={(event) => setOllamaModel(event.target.value)} />
              </div>
            </div>
            <div className="field">
              <label>プロンプト</label>
              <textarea
                placeholder="例: セーラー服をイラストとして描く時の重要なディテールをまとめて。"
                value={tagPrompt}
                onChange={(event) => setTagPrompt(event.target.value)}
              />
            </div>
            <button className="primary" disabled={action.busy || !tagName.trim() || !tagPrompt.trim()} onClick={() => void runTagJapaneseGeneration()}>
              <Bot size={17} />
              実行
            </button>
            <div className="field">
              <label>日本語説明文</label>
              <textarea
                className="tag-description"
                placeholder="LLMの結果がここに表示されます。手編集できます。"
                value={tagJapaneseDescription}
                onChange={(event) => setTagJapaneseDescription(event.target.value)}
              />
            </div>
            <button className="primary" disabled={action.busy || !tagJapaneseDescription.trim()} onClick={() => void translateTagEnglishDescription()}>
              <Bot size={17} />
              英訳
            </button>
            <div className="field">
              <label>英訳説明文</label>
              <textarea
                className="tag-description"
                placeholder="英訳ボタンの結果がここに表示されます。手編集できます。"
                value={tagEnglishDescription}
                onChange={(event) => setTagEnglishDescription(event.target.value)}
              />
            </div>
            <div className="modal-actions">
              <button onClick={() => setTagDialogOpen(false)}>キャンセル</button>
              <button className="primary" disabled={action.busy || !tagName.trim() || !tagEnglishName.trim() || !tagPrompt.trim() || !tagJapaneseDescription.trim() || !tagEnglishDescription.trim()} onClick={() => void saveTagFromDialog()}>
                <CheckCircle2 size={17} />
                保存
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {promptHistoryOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel prompt-history-modal">
            <div className="modal-header">
              <PanelTitle icon={<RefreshCw size={18} />} title="プロンプト履歴" />
              <button className="icon-button" onClick={() => setPromptHistoryOpen(false)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="prompt-history-list">
              {promptHistory.length ? (
                promptHistory.map((entry) => (
                  <article className="prompt-history-card" key={entry.id}>
                    <div className="prompt-history-thumb">
                      {entry.imageDataUrl || entry.imagePath ? (
                        <img src={imageUrl(entry.imageDataUrl ?? entry.imagePath ?? "")} alt="生成履歴サムネイル" />
                      ) : (
                        <ImageIcon size={24} />
                      )}
                    </div>
                    <div className="prompt-history-body">
                      <header>
                        <strong>{entry.checkpoint || "checkpoint未選択"}</strong>
                        <span>{entry.width} x {entry.height} / seed {entry.seed}</span>
                        <time>{new Date(entry.createdAt).toLocaleString("ja-JP", { hour12: false })}</time>
                      </header>
                      {entry.japanesePrompt ? <p className="history-japanese">{entry.japanesePrompt}</p> : null}
                      <p>{entry.prompt}</p>
                      <small>{entry.negativePrompt}</small>
                      <button onClick={() => applyPromptHistory(entry)}>
                        <CheckCircle2 size={17} />
                        流用
                      </button>
                    </div>
                  </article>
                ))
              ) : (
                <p className="empty">まだ履歴がありません。</p>
              )}
            </div>
            <div className="modal-actions">
              <button className="danger" disabled={!promptHistory.length || action.busy} onClick={() => void clearPromptHistory()}>
                <Trash2 size={17} />
                履歴削除
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {imagePreviewOpen && lastImage ? (
        <div className="modal-backdrop" onClick={() => setImagePreviewOpen(false)}>
          <section className="modal-panel image-preview-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <PanelTitle icon={<ImageIcon size={18} />} title="生成画像プレビュー" />
              <button className="icon-button" onClick={() => setImagePreviewOpen(false)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="image-preview-frame" style={{ aspectRatio: `${generationWidth} / ${generationHeight}` }}>
              <img src={imageUrl(lastImage)} alt="Generated preview enlarged" />
            </div>
          </section>
        </div>
      ) : null}
      {captionWorkspace ? (
        <div className="modal-backdrop">
          <section className="modal-panel caption-modal">
            <div className="modal-header">
              <div>
                <PanelTitle icon={<ImageIcon size={18} />} title={`キャプション編集: ${captionWorkspace.projectName}`} />
                <p className="modal-subtitle">{captionWorkspace.imageDir}</p>
              </div>
              <button className="icon-button" onClick={closeCaptionEditor} title="保存せず閉じる">
                <X size={17} />
              </button>
            </div>
            <div className="caption-editor-layout">
              <aside className="caption-tag-panel">
                <div className="caption-tag-editor">
                  <div className="two-columns">
                    <div className="field">
                      <label>日本語タグ</label>
                      <input placeholder="例: 麦わら帽子" value={captionTagJapanese} onChange={(event) => setCaptionTagJapanese(event.target.value)} />
                    </div>
                    <div className="field">
                      <label>英語タグ</label>
                      <input placeholder="空なら保存時に翻訳" value={captionTagEnglish} onChange={(event) => setCaptionTagEnglish(event.target.value)} />
                    </div>
                  </div>
                  <div className="button-row">
                    <button className="primary" disabled={!captionTagJapanese.trim()} onClick={upsertCaptionTag}>
                      <Plus size={16} />
                      {captionTagEditingId ? "タグ更新" : "タグ追加"}
                    </button>
                    <button onClick={resetCaptionTagEditor}>クリア</button>
                  </div>
                </div>
                <div className="caption-tag-list">
                  {captionTags.length ? (
                    captionTags.map((tag) => (
                      <article className={selectedCaptionTagId === tag.id ? "caption-tag selected" : "caption-tag"} key={tag.id}>
                        <button onClick={() => setSelectedCaptionTagId(tag.id)}>
                          <strong>{tag.japaneseName}</strong>
                          <span>{tag.englishName || "保存時に英訳"}</span>
                        </button>
                        <button className="icon-button" onClick={() => editCaptionTag(tag)} title="編集">
                          <Pencil size={15} />
                        </button>
                        <button className="icon-button danger" onClick={() => deleteCaptionTag(tag.id)} title="削除">
                          <Trash2 size={15} />
                        </button>
                      </article>
                    ))
                  ) : (
                    <p className="empty">まず左上でタグを追加してください。</p>
                  )}
                </div>
              </aside>
              <section className="caption-image-panel">
                <div className="caption-summary">
                  <span>画像 {captionImages.length}件</span>
                  <span>選択中: {selectedCaptionTagId ? captionTagById.get(selectedCaptionTagId)?.japaneseName ?? "不明" : "なし"}</span>
                  <button disabled={!selectedCaptionTagId || !captionImages.length} onClick={addSelectedTagToAllCaptionImages}>
                    <Plus size={15} />
                    全部に付与
                  </button>
                  <button disabled={!selectedCaptionTagId || !captionImages.length} onClick={removeSelectedTagFromAllCaptionImages}>
                    <X size={15} />
                    全部から解除
                  </button>
                </div>
                <div className="caption-image-grid">
                  {captionImages.length ? (
                    captionImages.map((image) => (
                      <article className="caption-image-card" key={image.fileName}>
                        <div className="caption-image-frame">
                          <button className="caption-image-preview-button" onClick={() => setCaptionPreviewImage(image)} title="拡大表示">
                            <img src={image.imageDataUrl} alt={image.fileName} />
                          </button>
                          <button className="caption-add-button" disabled={!selectedCaptionTagId} onClick={() => addSelectedTagToCaptionImage(image.fileName)} title="選択中タグを追加">
                            <Plus size={18} />
                          </button>
                          <button className="caption-edit-button" onClick={() => openCaptionImageTagEditor(image.fileName)} title="個別タグを編集">
                            <Pencil size={17} />
                          </button>
                        </div>
                        <strong>{image.fileName}</strong>
                        <div className="caption-chip-list">
                          {image.tagIds.length || image.customTags.length ? (
                            <>
                            {image.tagIds.map((tagId) => {
                              const tag = captionTagById.get(tagId);
                              return tag ? (
                                <button className="caption-chip" key={tagId} onClick={() => removeTagFromCaptionImage(image.fileName, tagId)}>
                                  {tag.japaneseName}
                                  <X size={13} />
                                </button>
                              ) : null;
                            })}
                            {image.customTags.map((tag) => (
                              <button className="caption-chip custom" key={tag.id} onClick={() => openCaptionImageTagEditor(image.fileName)}>
                                {tag.japaneseName}
                                <Pencil size={13} />
                              </button>
                            ))}
                            </>
                          ) : (
                            <span>タグ未設定</span>
                          )}
                        </div>
                      </article>
                    ))
                  ) : (
                    <p className="empty">Datasetに画像がありません。Datasetを開いて画像を入れてください。</p>
                  )}
                </div>
              </section>
            </div>
            <div className="modal-actions caption-actions">
              <button onClick={closeCaptionEditor}>保存せず閉じる</button>
              <button className="primary" disabled={action.busy || !captionImages.length} onClick={() => void saveCaptionEditor()}>
                <Save size={17} />
                キャプション保存
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {captionImageEditing ? (
        <div className="modal-backdrop image-tag-backdrop">
          <section className="modal-panel caption-image-tag-modal">
            <div className="modal-header">
              <PanelTitle icon={<Pencil size={18} />} title={`個別タグ: ${captionImageEditing.fileName}`} />
              <button className="icon-button" onClick={closeCaptionImageTagEditor} title="閉じる">
                <X size={17} />
              </button>
            </div>
            <div className="caption-image-tag-layout">
              <div className="caption-image-tag-preview">
                <img src={captionImageEditing.imageDataUrl} alt={captionImageEditing.fileName} />
              </div>
              <div className="caption-image-tag-form">
                <div className="field">
                  <label>日本語の個別描写</label>
                  <textarea
                    placeholder="例: 夕方の柔らかい逆光、机に突っ伏して眠っている"
                    value={captionImageTagJapanese}
                    onChange={(event) => setCaptionImageTagJapanese(event.target.value)}
                  />
                </div>
                <div className="field">
                  <label>英語タグ</label>
                  <textarea
                    placeholder="空なら保存時に翻訳"
                    value={captionImageTagEnglish}
                    onChange={(event) => setCaptionImageTagEnglish(event.target.value)}
                  />
                </div>
                <div className="button-row">
                  <button className="primary" disabled={!captionImageTagJapanese.trim()} onClick={upsertCaptionImageCustomTag}>
                    <Plus size={16} />
                    {captionImageTagEditingId ? "個別タグ更新" : "個別タグ追加"}
                  </button>
                  <button onClick={resetCaptionImageTagFields}>クリア</button>
                </div>
                <div className="caption-custom-list">
                  {captionImageEditing.customTags.length ? (
                    captionImageEditing.customTags.map((tag) => (
                      <article className="caption-custom-row" key={tag.id}>
                        <button onClick={() => editCaptionImageCustomTag(tag)}>
                          <strong>{tag.japaneseName}</strong>
                          <span>{tag.englishName || "保存時に英訳"}</span>
                        </button>
                        <button className="icon-button danger" onClick={() => deleteCaptionImageCustomTag(captionImageEditing.fileName, tag.id)} title="削除">
                          <Trash2 size={15} />
                        </button>
                      </article>
                    ))
                  ) : (
                    <p className="empty">この画像専用の描写はまだありません。</p>
                  )}
                </div>
              </div>
            </div>
            <div className="modal-actions">
              <button onClick={closeCaptionImageTagEditor}>閉じる</button>
            </div>
          </section>
        </div>
      ) : null}
      {captionPreviewImage ? (
        <div className="modal-backdrop image-tag-backdrop" onClick={() => setCaptionPreviewImage(null)}>
          <section className="modal-panel image-preview-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <PanelTitle icon={<ImageIcon size={18} />} title={captionPreviewImage.fileName} />
              <button className="icon-button" onClick={() => setCaptionPreviewImage(null)} title="閉じる">
                <X size={17} />
              </button>
            </div>
            <div className="image-preview-frame">
              <img src={captionPreviewImage.imageDataUrl} alt={captionPreviewImage.fileName} />
            </div>
          </section>
        </div>
      ) : null}
      {loraCreateDialogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel lora-create-modal">
            <div className="modal-header">
              <PanelTitle icon={<Boxes size={18} />} title={loraProjectDialogMode === "edit" ? "LoRAプロジェクト編集" : "LoRAプロジェクト新規作成"} />
              <button className="icon-button" onClick={() => setLoraCreateDialogOpen(false)} title="閉じる">
                <X size={17} />
              </button>
            </div>
            <div className="field">
              <label>名前</label>
              <input placeholder="my_character_lora" value={loraName} onChange={(event) => setLoraName(event.target.value)} />
            </div>
            <div className="field">
              <label>ターゲット / Trigger Word</label>
              <input placeholder="mkchar" value={loraTriggerWord} onChange={(event) => setLoraTriggerWord(event.target.value)} />
            </div>
            <div className="field">
              <label>Base Checkpoint</label>
              <select value={selectedCheckpoint} onChange={(event) => setSelectedCheckpoint(event.target.value)}>
                {status?.checkpoints.length ? (
                  status.checkpoints.map((file) => (
                    <option key={file.name} value={file.name}>
                      {file.name}
                    </option>
                  ))
                ) : (
                  <option value="">未追加</option>
                )}
              </select>
            </div>
            <div className="modal-actions">
              <button onClick={() => setLoraCreateDialogOpen(false)}>キャンセル</button>
              <button className="primary" disabled={action.busy || !loraName.trim() || !selectedCheckpoint} onClick={() => void saveLoraProject()}>
                <CheckCircle2 size={17} />
                {loraProjectDialogMode === "edit" ? "保存" : "作成"}
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {loraCatalogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel lora-catalog-modal">
            <div className="modal-header">
              <PanelTitle icon={<Download size={18} />} title="Civitai LoRAカタログ" />
              <button className="icon-button" onClick={() => setLoraCatalogOpen(false)} title="閉じる">
                <X size={17} />
              </button>
            </div>
            <div className="two-columns">
              <div className="field">
                <label>検索語</label>
                <input placeholder="character, style, outfit..." value={loraCatalogQuery} onChange={(event) => setLoraCatalogQuery(event.target.value)} />
              </div>
              <div className="field">
                <label>期間</label>
                <select value={loraCatalogPeriod} onChange={(event) => setLoraCatalogPeriod(event.target.value as "Month" | "AllTime")}>
                  <option value="Month">直近1ヶ月</option>
                  <option value="AllTime">累計</option>
                </select>
              </div>
            </div>
            <button className="primary" disabled={action.busy} onClick={() => void searchLoraCatalog()}>
              <RefreshCw size={17} />
              検索
            </button>
            <div className="catalog-list lora-catalog-list">
              {loraCatalogModels.length ? (
                loraCatalogModels.map((model) => (
                  <article className="catalog-card lora-catalog-card" key={`${model.id}-${model.versionId}`}>
                    {model.imageUrl ? <img src={model.imageUrl} alt="" /> : <div className="catalog-no-image">No Image</div>}
                    <div>
                      <h3>{model.name}</h3>
                      <p>{model.creator} / {model.baseModel || "base不明"} / {model.versionName || "version不明"}</p>
                      <p>{Math.round(model.sizeKB / 1024)} MB / DL {model.downloadCount.toLocaleString()}</p>
                      <button disabled={action.busy} onClick={() => void downloadCivitaiLora(model)}>
                        <Download size={15} />
                        Projectへ追加
                      </button>
                    </div>
                  </article>
                ))
              ) : (
                <p className="empty">検索語を入れて検索してください。</p>
              )}
            </div>
          </section>
        </div>
      ) : null}
      <section className="log-dock global-log">
        <LogPanel
          action={action}
          logs={logs}
          logViewRef={logViewRef}
          onScroll={handleLogScroll}
          onLatest={jumpToLatestLog}
          onClear={() => setLogs([])}
          downloadProgress={downloadProgress}
        />
      </section>
    </main>
  );

}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="panel-title">
      {icon}
      <h2>{title}</h2>
    </div>
  );
}

function PromptTransformPanel({
  japanesePrompt,
  setJapanesePrompt,
  prompt,
  setPrompt,
  negativePrompt,
  setNegativePrompt,
  promptNotes,
  promptEnhanceMode,
  setPromptEnhanceMode,
  canEnhancePrompt,
  disabledReason,
  onEnhance,
  illustrationTags,
  selectedTagIds,
  onToggleTag,
  onAddTag,
  onEditTag,
  onDeleteTag
}: {
  japanesePrompt: string;
  setJapanesePrompt: (value: string) => void;
  prompt: string;
  setPrompt: (value: string) => void;
  negativePrompt: string;
  setNegativePrompt: (value: string) => void;
  promptNotes: string;
  promptEnhanceMode: PromptEnhanceMode;
  setPromptEnhanceMode: (value: PromptEnhanceMode) => void;
  canEnhancePrompt: boolean;
  disabledReason: string;
  onEnhance: () => void;
  illustrationTags: IllustrationTag[];
  selectedTagIds: string[];
  onToggleTag: (id: string) => void;
  onAddTag: () => void;
  onEditTag: (tag: IllustrationTag) => void;
  onDeleteTag: (id: string) => void;
}) {
  return (
    <div className="prompt-transform">
      <div className="field">
        <label>Japanese Prompt</label>
        <textarea
          placeholder="例: 夕方の教室で、青緑のツインテールの女子高生。少し照れている感じ。"
          value={japanesePrompt}
          onChange={(event) => setJapanesePrompt(event.target.value)}
        />
      </div>
      <div className="field">
        <label>変換モード</label>
        <select value={promptEnhanceMode} onChange={(event) => setPromptEnhanceMode(event.target.value as PromptEnhanceMode)}>
          <option value="balanced-detail">おすすめ: 本文 + 脚注/定義を反映</option>
          <option value="footnote-strict">脚注重視: 定義文を強制採用</option>
          <option value="main-subject">本文重視: 主文を優先</option>
          <option value="tag-dense">タグ密度重視: 細かいタグへ分解</option>
        </select>
      </div>
      <TagSelector
        illustrationTags={illustrationTags}
        selectedTagIds={selectedTagIds}
        onToggleTag={onToggleTag}
        onAddTag={onAddTag}
        onEditTag={onEditTag}
        onDeleteTag={onDeleteTag}
      />
      <button className="primary" disabled={!canEnhancePrompt} title={disabledReason} onClick={onEnhance}>
        <Bot size={17} />
        日本語から生成promptへ変換
      </button>
      <p className={canEnhancePrompt ? "result-message ready" : "result-message"}>{disabledReason}</p>
      {promptNotes ? <p className="result-message">{promptNotes}</p> : null}
      <div className="field">
        <label>Positive Prompt</label>
        <textarea
          value={prompt}
          onChange={(event) => {
            setPrompt(event.target.value);
          }}
        />
      </div>
      <div className="field">
        <label>Negative Prompt</label>
        <textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} />
      </div>
    </div>
  );
}

function TagSelector({
  illustrationTags,
  selectedTagIds,
  onToggleTag,
  onAddTag,
  onEditTag,
  onDeleteTag
}: {
  illustrationTags: IllustrationTag[];
  selectedTagIds: string[];
  onToggleTag: (id: string) => void;
  onAddTag: () => void;
  onEditTag: (tag: IllustrationTag) => void;
  onDeleteTag: (id: string) => void;
}) {
  return (
    <div className="tag-builder">
      <div className="tag-builder-header">
        <h2>タグ</h2>
        <button onClick={onAddTag}>
          <WandSparkles size={17} />
          タグ追加
        </button>
      </div>
      <div className="tag-chip-list">
        {illustrationTags.length ? (
          illustrationTags.map((tag) => (
            <div className={selectedTagIds.includes(tag.id) ? "tag-chip selected" : "tag-chip"} key={tag.id}>
              <button onClick={() => onToggleTag(tag.id)}>
                <span>{tag.name}</span>
                <small>{tag.englishName || tag.name}</small>
              </button>
              <button className="icon-button" onClick={() => onEditTag(tag)} title="編集">
                <Bot size={15} />
              </button>
              <button className="icon-button danger" onClick={() => onDeleteTag(tag.id)} title="削除">
                <Trash2 size={15} />
              </button>
            </div>
          ))
        ) : (
          <p className="empty">まだタグがありません。</p>
        )}
      </div>
    </div>
  );
}

function GenerationLoraPanel({
  loras,
  selectedLoras,
  onToggleLora,
  onUpdateLora
}: {
  loras: RuntimeStatus["loras"];
  selectedLoras: GenerationLora[];
  onToggleLora: (name: string) => void;
  onUpdateLora: (name: string, field: "modelStrength" | "clipStrength", value: number) => void;
}) {
  return (
    <div className="lora-apply-panel">
      <div className="tag-builder-header">
        <h2>LoRA適用</h2>
        <span>{selectedLoras.length}件選択</span>
      </div>
      <div className="lora-apply-list">
        {loras.length ? (
          loras.map((lora) => {
            const selected = selectedLoras.find((item) => item.name === lora.name);
            return (
              <article className={selected ? "lora-apply-card selected" : "lora-apply-card"} key={lora.name}>
                <button onClick={() => onToggleLora(lora.name)}>
                  <strong>{lora.name}</strong>
                  <small>{lora.sizeLabel}</small>
                </button>
                {selected ? (
                  <div className="lora-strength-grid">
                    <label>
                      Model
                      <input
                        type="number"
                        min="-2"
                        max="2"
                        step="0.05"
                        value={selected.modelStrength}
                        onChange={(event) => onUpdateLora(lora.name, "modelStrength", Number(event.target.value))}
                      />
                    </label>
                    <label>
                      CLIP
                      <input
                        type="number"
                        min="-2"
                        max="2"
                        step="0.05"
                        value={selected.clipStrength}
                        onChange={(event) => onUpdateLora(lora.name, "clipStrength", Number(event.target.value))}
                      />
                    </label>
                  </div>
                ) : null}
              </article>
            );
          })
        ) : (
          <p className="empty">まだLoRAがありません。環境タブのモデル追加でLoRAを保存してください。</p>
        )}
      </div>
    </div>
  );
}

function LogPanel({
  action,
  logs,
  logViewRef,
  onScroll,
  onLatest,
  onClear,
  downloadProgress
}: {
  action: ActionState;
  logs: LogLine[];
  logViewRef: React.RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  onLatest: () => void;
  onClear: () => void;
  downloadProgress: number | null;
}) {
  return (
    <>
      <div className="progress-header">
        <span>{action.busy ? `処理中: ${action.message}` : `ログ: ${action.message}`}</span>
        <div className="log-actions">
          <button className="tiny-button" onClick={onLatest}>
            Latest
          </button>
          <button className="tiny-button" onClick={onClear}>
            Clear
          </button>
        </div>
      </div>
      <div className={downloadProgress !== null ? "progress-bar determinate" : action.busy ? "progress-bar active" : "progress-bar"}>
        <span style={downloadProgress !== null ? { width: `${downloadProgress}%` } : undefined} />
      </div>
      <div className="log-view" ref={logViewRef} onScroll={onScroll}>
        {logs.length ? (
          logs.map((log, index) => (
            <p key={`${log.id}-${index}`} className={`log-line ${log.level}`}>
              <span>{log.time}</span>
              {log.line}
            </p>
          ))
        ) : (
          <p className="log-line info">
            <span>--:--:--</span>
            ログはここに表示されます。
          </p>
        )}
      </div>
    </>
  );
}

type StatusTone = "ok" | "warn" | "error";

function StatusGrid({
  status,
  ollamaStatus,
  runtimeTab
}: {
  status: RuntimeStatus | null;
  ollamaStatus?: OllamaStatus | null;
  runtimeTab: "macos" | "docker";
}) {
  const runtimeItems: Array<{ label: string; value: string; tone: StatusTone }> =
    runtimeTab === "docker"
      ? [
          { label: "Docker", value: status?.dockerDesktopInstalled ? "検出" : "未検出", tone: status?.dockerDesktopInstalled ? "ok" : "error" },
          { label: "Docker API", value: status?.dockerDaemonReady ? "接続OK" : "未起動", tone: status?.dockerDaemonReady ? "ok" : "warn" },
          { label: "Compose", value: status?.composeCommand ? "OK" : "未検出", tone: status?.composeCommand ? "ok" : "error" }
        ]
      : [
          { label: "macOS Runtime", value: status?.nativeInstalled ? "OK" : "未インストール", tone: status?.nativeInstalled ? "ok" : "error" },
          { label: "ComfyUI", value: status?.comfyReachable ? "接続中" : status?.nativeRunning ? "起動中" : "未接続", tone: status?.comfyReachable ? "ok" : status?.nativeRunning ? "warn" : "error" }
        ];

  const ollamaValue = ollamaStatus?.reachable && ollamaStatus.dataLinked ? "接続OK" : ollamaStatus?.reachable ? "保存先NG" : ollamaStatus?.installed ? "未起動" : "未検出";
  const ollamaTone: StatusTone = ollamaStatus?.reachable && ollamaStatus.dataLinked ? "ok" : ollamaStatus?.installed ? "warn" : "error";
  const dataTone: StatusTone = ollamaStatus?.dataLinked ? "ok" : ollamaStatus?.installed ? "warn" : "error";
  const items = [
    ...runtimeItems,
    { label: "Runtime Path", value: status?.pathsReady ? "OK" : "未初期化", tone: status?.pathsReady ? "ok" : "error" },
    { label: "Ollama", value: ollamaValue, tone: ollamaTone },
    { label: "DataPath", value: ollamaStatus?.dataLinked ? "OK" : "未切替", tone: dataTone }
  ];

  return (
    <div className="status-grid">
      {items.map(({ label, value, tone }) => (
        <div className={`status-card ${tone}`} key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
