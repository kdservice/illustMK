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
  Play,
  RefreshCw,
  Server,
  Square,
  TerminalSquare,
  Trash2,
  WandSparkles
} from "lucide-react";
import type {
  CivitaiModelCard,
  DownloadRequest,
  HuggingFaceLlmModelCard,
  IllustrationTag,
  LlmChatMessage,
  OllamaStatus,
  PromptEnhanceMode,
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
const defaultBaseFilters = ["SDXL", "Pony", "Illustrious", "Animagine"];
const allBaseFilters = [
  { name: "SDXL", status: "対応" },
  { name: "Pony", status: "対応" },
  { name: "Illustrious", status: "対応" },
  { name: "Animagine", status: "対応" },
  { name: "Anima", status: "現在は非対応" },
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

function App() {
  const [paths, setPaths] = useState<RuntimePaths | null>(null);
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [comfyUrl, setComfyUrl] = useState(defaultComfyUrl);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadFileName, setDownloadFileName] = useState("");
  const [downloadSha, setDownloadSha] = useState("");
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
  const [tagPrompt, setTagPrompt] = useState("");
  const [tagJapaneseDescription, setTagJapaneseDescription] = useState("");
  const [tagEnglishDescription, setTagEnglishDescription] = useState("");
  const [japanesePrompt, setJapanesePrompt] = useState("");
  const [promptEnhanceMode, setPromptEnhanceMode] = useState<PromptEnhanceMode>("balanced-detail");
  const [promptNotes, setPromptNotes] = useState("");
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [prompt, setPrompt] = useState("masterpiece, best quality, 1girl, anime style, soft light");
  const [negativePrompt, setNegativePrompt] = useState("low quality, worst quality, blurry, bad anatomy");
  const [lastImage, setLastImage] = useState<string | null>(null);
  const [generationResult, setGenerationResult] = useState<TestGenerationResult | null>(null);
  const [action, setAction] = useState<ActionState>({ busy: false, message: "待機中", error: null });
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [autoFollowLogs, setAutoFollowLogs] = useState(true);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [mainTab, setMainTab] = useState<"environment" | "llm" | "generate">("environment");
  const [runtimeTab, setRuntimeTab] = useState<"macos" | "docker">("macos");
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
  const effectivePrompt = useMemo(() => {
    const tagPromptText = selectedTags.map((tag) => tag.englishDescription).filter(Boolean).join(", ");
    return [prompt, tagPromptText].filter((part) => part.trim()).join(", ");
  }, [prompt, selectedTags]);
  const canGenerate = Boolean(status?.comfyReachable && selectedCheckpointInfo?.simpleWorkflowReady && !action.busy);
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

  async function runStreamOperation(operation: string, message: string): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", message, id);
    setAction({ busy: true, message, error: null });
    setDownloadProgress(null);
    try {
      const result = await window.illustMK.runtime.runOperation({ id, operation });
      appendLog("info", result, id);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
      await refreshOllamaStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
      await refreshStatus();
      await refreshOllamaStatus();
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

  function appendLog(level: LogLine["level"], line: string, id = currentOperationId.current ?? "manual"): void {
    setLogs((current) => [
      ...current.slice(-300),
      {
        id,
        level,
        line,
        time: new Date().toLocaleTimeString("ja-JP", { hour12: false })
      }
    ]);
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

  function openNewTagDialog(): void {
    setTagEditingId(null);
    setTagName("");
    setTagPrompt("");
    setTagJapaneseDescription("");
    setTagEnglishDescription("");
    setTagDialogOpen(true);
  }

  function openEditTagDialog(tag: IllustrationTag): void {
    setTagEditingId(tag.id);
    setTagName(tag.name);
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

  async function saveTagFromDialog(): Promise<void> {
    await runAction(
      "タグを保存中",
      async () => {
        const englishDescription = await window.illustMK.tags.translateEnglish({
          ollamaUrl,
          model: ollamaModel,
          japaneseDescription: tagJapaneseDescription
        });
        setTagEnglishDescription(englishDescription);
        return window.illustMK.tags.save({
          id: tagEditingId ?? undefined,
          name: tagName,
          sourcePrompt: tagPrompt,
          japaneseDescription: tagJapaneseDescription,
          englishDescription
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
    width: 768,
    height: 1024,
    steps: 20,
    cfg: 7,
    seed: -1,
    comfyUrl
  };

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
        <StatusGrid status={status} ollamaStatus={ollamaStatus} />
      </section>

      <nav className="main-tabs">
        <button className={mainTab === "environment" ? "tab active" : "tab"} onClick={() => setMainTab("environment")}>
          環境
        </button>
        <button className={mainTab === "llm" ? "tab active" : "tab"} onClick={() => setMainTab("llm")}>
          LLM
        </button>
        <button className={mainTab === "generate" ? "tab active" : "tab"} onClick={() => setMainTab("generate")}>
          生成
        </button>
      </nav>

      {mainTab === "environment" ? (
        <section className="environment-layout">
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

          <section className="panel llm-panel">
            <PanelTitle icon={<Bot size={18} />} title="LLM / Ollama" />
            <div className="status-grid compact">
              <div>
                <span>Ollama.app</span>
                <strong>{ollamaStatus?.installed ? "検出" : "未検出"}</strong>
              </div>
              <div>
                <span>API</span>
                <strong>{ollamaStatus?.reachable ? "接続OK" : "未接続"}</strong>
              </div>
              <div>
                <span>DataPath</span>
                <strong>{ollamaStatus?.dataLinked ? "OK" : "未切替"}</strong>
              </div>
            </div>
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
      ) : (
        <section className="generate-layout">
          <section className="panel preview-panel">
            <PanelTitle icon={<ImageIcon size={18} />} title="テスト生成" />
            <div className="preview-stage">
              {lastImage ? (
                <img src={imageUrl(lastImage)} alt="Generated preview" />
              ) : (
                <div className="placeholder">
                  <ImageIcon size={42} />
                  <p>生成結果がここに表示されます。</p>
                </div>
              )}
            </div>
          </section>

          <section className="panel generate-panel">
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
            />
            <div className="tag-builder">
              <div className="tag-builder-header">
                <div>
                  <h2>タグ</h2>
                  <p>{selectedTags.length ? `${selectedTags.length}件を生成promptへ追加` : "選択したタグの英訳説明文を生成promptへ追加します。"}</p>
                </div>
                <button onClick={openNewTagDialog}>
                  <WandSparkles size={17} />
                  タグ追加
                </button>
              </div>
              <div className="tag-chip-list">
                {illustrationTags.length ? (
                  illustrationTags.map((tag) => (
                    <div className={selectedTagIds.includes(tag.id) ? "tag-chip selected" : "tag-chip"} key={tag.id}>
                      <button onClick={() => toggleIllustrationTag(tag.id)}>
                        <span>{tag.name}</span>
                      </button>
                      <button className="icon-button" onClick={() => openEditTagDialog(tag)} title="編集">
                        <Bot size={15} />
                      </button>
                      <button className="icon-button danger" onClick={() => void deleteIllustrationTag(tag.id)} title="削除">
                        <Trash2 size={15} />
                      </button>
                    </div>
                  ))
                ) : (
                  <p className="empty">まだタグがありません。</p>
                )}
              </div>
            </div>
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

            <button
              className="generate"
              disabled={!canGenerate}
              onClick={() =>
                void runAction("ComfyUIでテスト生成中", () => window.illustMK.runtime.testGenerate(testRequest), (result) => {
                  setGenerationResult(result);
                  if (result.imageDataUrl || result.imagePath) {
                    setLastImage(result.imageDataUrl ?? result.imagePath ?? null);
                  }
                })
              }
            >
              <WandSparkles size={18} />
              GENERATE TEST
            </button>
            {selectedCheckpointInfo && !selectedCheckpointInfo.simpleWorkflowReady ? (
              <p className="result-message">
                このCheckpointは外部CLIP/VAEが必要です。今のテスト生成では、簡易生成OKのCheckpointを選んでください。
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
            {tagEnglishDescription ? (
              <div className="field">
                <label>英訳説明文</label>
                <textarea className="tag-description" value={tagEnglishDescription} onChange={(event) => setTagEnglishDescription(event.target.value)} />
              </div>
            ) : null}
            <div className="modal-actions">
              <button onClick={() => setTagDialogOpen(false)}>キャンセル</button>
              <button className="primary" disabled={action.busy || !tagName.trim() || !tagPrompt.trim() || !tagJapaneseDescription.trim()} onClick={() => void saveTagFromDialog()}>
                <CheckCircle2 size={17} />
                保存
              </button>
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
  onEnhance
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
      <button className="primary" disabled={!canEnhancePrompt} title={disabledReason} onClick={onEnhance}>
        <Bot size={17} />
        日本語から生成promptへ変換
      </button>
      <p className={canEnhancePrompt ? "result-message ready" : "result-message"}>{disabledReason}</p>
      {promptNotes ? <p className="result-message">{promptNotes}</p> : null}
      <div className="field">
        <label>Positive Prompt</label>
        <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      </div>
      <div className="field">
        <label>Negative Prompt</label>
        <textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} />
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

function StatusGrid({ status, ollamaStatus }: { status: RuntimeStatus | null; ollamaStatus?: OllamaStatus | null }) {
  const items = [
    ["Desktop", status?.dockerDesktopInstalled ? "OK" : "未検出"],
    ["Docker", status?.dockerAvailable ? "OK" : "未検出"],
    ["Daemon", status?.dockerDaemonReady ? "接続OK" : "未起動"],
    ["Compose", status?.composeCommand ?? "未検出"],
    ["Runtime", status?.pathsReady ? "OK" : "未初期化"],
    ["ComfyUI", status?.comfyReachable ? "接続中" : "未接続"],
    ["Image Models", `${status?.checkpoints.length ?? 0}`],
    ["Ollama", ollamaStatus?.reachable && ollamaStatus.dataLinked ? "接続OK" : ollamaStatus?.reachable ? "保存先NG" : ollamaStatus?.installed ? "未起動" : "未検出"],
    ["LLM Models", `${ollamaStatus?.models.length ?? 0}`]
  ];

  return (
    <div className="status-grid">
      {items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
