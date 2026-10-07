import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Boxes,
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
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [prompt, setPrompt] = useState("masterpiece, best quality, 1girl, anime style, soft light");
  const [negativePrompt, setNegativePrompt] = useState("low quality, worst quality, blurry, bad anatomy");
  const [lastImage, setLastImage] = useState<string | null>(null);
  const [generationResult, setGenerationResult] = useState<TestGenerationResult | null>(null);
  const [action, setAction] = useState<ActionState>({ busy: false, message: "待機中", error: null });
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [autoFollowLogs, setAutoFollowLogs] = useState(true);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [mainTab, setMainTab] = useState<"environment" | "generate">("environment");
  const [runtimeTab, setRuntimeTab] = useState<"macos" | "docker">("macos");
  const logViewRef = useRef<HTMLDivElement | null>(null);
  const currentOperationId = useRef<string | null>(null);

  const selectedCheckpointInfo = useMemo(
    () => status?.checkpoints.find((checkpoint) => checkpoint.name === selectedCheckpoint) ?? null,
    [selectedCheckpoint, status?.checkpoints]
  );
  const canGenerate = Boolean(status?.comfyReachable && selectedCheckpointInfo?.simpleWorkflowReady && !action.busy);

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
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
      await refreshStatus();
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

  async function loadSettings(): Promise<void> {
    const settings = await window.illustMK.settings.get();
    setCivitaiApiKey(settings.civitaiApiKey);
  }

  async function saveSettings(): Promise<void> {
    await runAction("Civitai APIキーを保存中", () => window.illustMK.settings.save({ civitaiApiKey }));
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
    prompt,
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
        <div className="connection-pill">
          <span className={status?.comfyReachable ? "dot ready" : "dot"} />
          {stateLabel}
        </div>
      </header>

      <nav className="main-tabs">
        <button className={mainTab === "environment" ? "tab active" : "tab"} onClick={() => setMainTab("environment")}>
          環境
        </button>
        <button className={mainTab === "generate" ? "tab active" : "tab"} onClick={() => setMainTab("generate")}>
          生成
        </button>
      </nav>

      {mainTab === "environment" ? (
        <section className="environment-layout">
          <section className="panel setup-panel">
            <PanelTitle icon={<Server size={18} />} title="環境構築" />
            <div className="runtime-tabs">
              <button className={runtimeTab === "macos" ? "runtime-tab active" : "runtime-tab"} onClick={() => setRuntimeTab("macos")}>
                macOS <span>{status?.activeRuntime === "macos" ? "ACTIVE" : ""}</span>
              </button>
              <button className={runtimeTab === "docker" ? "runtime-tab active" : "runtime-tab"} onClick={() => setRuntimeTab("docker")}>
                Docker <span>LEGACY</span>
              </button>
            </div>
            <StatusGrid status={status} />

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

            <div className={action.error ? "action-log error" : "action-log"}>
              {action.error ? <CircleAlert size={18} /> : <CheckCircle2 size={18} />}
              <div>
                <strong>{action.message}</strong>
                <p>{action.error ?? "環境操作の結果は下段ログに集約します。"}</p>
              </div>
            </div>
          </section>

          <section className="panel">
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

          <section className="log-dock environment-log">
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
            <div className="field">
              <label>Positive Prompt</label>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </div>
            <div className="field">
              <label>Negative Prompt</label>
              <textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} />
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

function StatusGrid({ status }: { status: RuntimeStatus | null }) {
  const items = [
    ["Desktop", status?.dockerDesktopInstalled ? "OK" : "未検出"],
    ["Docker", status?.dockerAvailable ? "OK" : "未検出"],
    ["Daemon", status?.dockerDaemonReady ? "接続OK" : "未起動"],
    ["Compose", status?.composeCommand ?? "未検出"],
    ["Runtime", status?.pathsReady ? "OK" : "未初期化"],
    ["ComfyUI", status?.comfyReachable ? "接続中" : "未接続"],
    ["Models", `${status?.checkpoints.length ?? 0}`]
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
