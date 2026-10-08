import { createHash, randomUUID } from "node:crypto";
import { closeSync, createReadStream, createWriteStream, openSync } from "node:fs";
import { access, cp, lstat, mkdir, readFile, readlink, readdir, rename, rm, stat, symlink, unlink, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { homedir } from "node:os";
import { Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { execFile, spawn, type ChildProcess, type ChildProcessWithoutNullStreams, type ExecFileOptions } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export type RuntimePaths = {
  root: string;
  llm: string;
  ollama: string;
  ollamaModels: string;
  ollamaLog: string;
  ollamaPid: string;
  models: string;
  checkpoints: string;
  loras: string;
  vae: string;
  outputs: string;
  workflows: string;
  loraProjects: string;
  loraRuntime: string;
  loraScripts: string;
  loraVenv: string;
  loraPython: string;
  loraLog: string;
  native: string;
  nativeComfy: string;
  nativeVenv: string;
  nativePython: string;
  nativeLog: string;
  nativePid: string;
  docker: string;
  logs: string;
  downloads: string;
  composeFile: string;
};

export type RuntimeStatus = {
  runtimeRoot: string;
  activeRuntime: "macos" | "docker";
  nativeReady: boolean;
  nativeInstalled: boolean;
  nativeRunning: boolean;
  nativePid: number | null;
  dockerDesktopInstalled: boolean;
  dockerDesktopPath: string | null;
  dockerAvailable: boolean;
  dockerVersion: string | null;
  dockerDaemonReady: boolean;
  dockerDaemonMessage: string | null;
  composeAvailable: boolean;
  composeCommand: string | null;
  pathsReady: boolean;
  composeReady: boolean;
  comfyReachable: boolean;
  comfyUrl: string;
  containerState: string;
  checkpoints: ModelFileInfo[];
  loras: ModelFileInfo[];
  outputs: string[];
};

export type ModelFileInfo = {
  name: string;
  sizeBytes: number;
  sizeLabel: string;
  hasClip: boolean;
  hasVae: boolean;
  simpleWorkflowReady: boolean;
  compatibilityLabel: string;
};

export type DownloadRequest = {
  url: string;
  targetType: "checkpoints" | "loras" | "vae";
  fileName?: string;
  sha256?: string;
  source?: "civitai" | "direct";
  operationId?: string;
};

export type DeleteModelRequest = {
  targetType: "checkpoints" | "loras" | "vae";
  fileName: string;
};

export type DownloadResult = {
  ok: boolean;
  path: string;
  sha256: string;
  message: string;
};

export type CivitaiModelCard = {
  id: number;
  name: string;
  type: string;
  nsfw: boolean;
  creator: string;
  versionId: number;
  versionName: string;
  baseModel: string;
  fileName: string;
  sizeKB: number;
  downloadUrl: string;
  pageUrl: string;
  imageUrl: string | null;
  downloadCount: number;
  rating: number;
  sha256?: string;
};

export type HuggingFaceLlmModelCard = {
  id: string;
  name: string;
  author: string;
  pullName: string;
  downloads: number;
  likes: number;
  updatedAt: string | null;
  tags: string[];
  ggufFiles: Array<{
    name: string;
    sizeBytes: number | null;
    sizeLabel: string;
  }>;
  totalSizeBytes: number | null;
  totalSizeLabel: string;
};

export type AppSettings = {
  civitaiApiKey: string;
  huggingFaceApiKey: string;
  ollamaUrl: string;
  ollamaModel: string;
};

export type OllamaModelInfo = {
  name: string;
  sizeBytes: number;
  sizeLabel: string;
  modifiedAt: string | null;
};

export type OllamaStatus = {
  url: string;
  reachable: boolean;
  installed: boolean;
  appPath: string | null;
  dataLinked: boolean;
  modelPath: string;
  models: OllamaModelInfo[];
  message: string | null;
};

export type PromptEnhanceMode = "balanced-detail" | "footnote-strict" | "main-subject" | "tag-dense";

export type PromptEnhanceRequest = {
  ollamaUrl: string;
  model: string;
  japanesePrompt: string;
  currentPrompt: string;
  currentNegativePrompt: string;
  selectedTags?: Pick<IllustrationTag, "name" | "englishName" | "englishDescription">[];
  mode?: PromptEnhanceMode;
};

export type PromptEnhanceResult = {
  positivePrompt: string;
  negativePrompt: string;
  notes: string;
};

export type LlmChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  model: string;
  createdAt: string;
};

export type LlmChatRequest = {
  ollamaUrl: string;
  model: string;
  prompt: string;
};

export type IllustrationTag = {
  id: string;
  name: string;
  englishName: string;
  sourcePrompt: string;
  japaneseDescription: string;
  englishDescription: string;
  createdAt: string;
  updatedAt: string;
};

export type TagDescriptionRequest = {
  ollamaUrl: string;
  model: string;
  tagName: string;
  prompt: string;
};

export type TagTranslationRequest = {
  ollamaUrl: string;
  model: string;
  japaneseDescription: string;
};

export type SaveIllustrationTagRequest = {
  id?: string;
  name: string;
  englishName: string;
  sourcePrompt: string;
  japaneseDescription: string;
  englishDescription: string;
};

export type TestGenerationRequest = {
  prompt: string;
  negativePrompt: string;
  checkpoint: string;
  loras?: GenerationLora[];
  width: number;
  height: number;
  steps: number;
  cfg: number;
  seed: number;
  comfyUrl: string;
};

export type GenerationLora = {
  name: string;
  modelStrength: number;
  clipStrength: number;
};

export type LoraTrainingProject = {
  id: string;
  name: string;
  triggerWord: string;
  sourceLora: string;
  baseCheckpoint: string;
  datasetPath: string;
  outputName: string;
  resolution: number;
  repeats: number;
  epochs: number;
  networkDim: number;
  learningRate: string;
  createdAt: string;
  updatedAt: string;
  status: "draft" | "ready" | "running" | "complete" | "blocked" | "failed";
  message: string;
};

export type SaveLoraTrainingProjectRequest = {
  id?: string;
  name: string;
  triggerWord: string;
  sourceLora?: string;
  baseCheckpoint: string;
  datasetPath?: string;
  outputName?: string;
  resolution: number;
  repeats: number;
  epochs: number;
  networkDim: number;
  learningRate: string;
};

export type PromptHistoryEntry = {
  id: string;
  japanesePrompt: string;
  prompt: string;
  negativePrompt: string;
  checkpoint: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  seed: number;
  imagePath?: string;
  imageDataUrl?: string;
  createdAt: string;
};

export type SavePromptHistoryRequest = TestGenerationRequest & {
  japanesePrompt: string;
  imagePath?: string;
  imageDataUrl?: string;
};

export type TestGenerationResult = {
  ok: boolean;
  message: string;
  imagePath?: string;
  imageDataUrl?: string;
  promptId?: string;
};

export type RuntimeLogEvent = {
  level: "info" | "stdout" | "stderr";
  line: string;
};

export type RuntimeLogSink = (event: RuntimeLogEvent) => void;

const runtimeRoot = process.env.ILLUSTMK_RUNTIME_ROOT ?? "/Volumes/ARC-8050T2/illustMK";
const appDataRoot = process.env.ILLUSTMK_APP_DATA_ROOT ?? join(process.cwd(), ".illustmk");
const settingsPath = join(appDataRoot, "settings.json");
const llmChatHistoryPath = join(appDataRoot, "llm-chat-history.json");
const illustrationTagsPath = join(appDataRoot, "illustration-tags.json");
const promptHistoryPath = join(appDataRoot, "prompt-history.json");
const loraTrainingProjectsPath = join(appDataRoot, "lora-training-projects.json");
const comfyPort = Number(process.env.ILLUSTMK_COMFY_PORT ?? "8188");
let nativeProcess: ChildProcess | null = null;
let ollamaProcess: ChildProcess | null = null;
const dockerDesktopPaths = ["/Applications/Docker.app", "/Applications/Utilities/Docker.app"];
const ollamaAppPaths = ["/Applications/Ollama.app", "/Applications/Utilities/Ollama.app", join(homedir(), "Applications", "Ollama.app")];
const userOllamaHome = join(homedir(), ".ollama");
const dockerDesktopBinPaths = dockerDesktopPaths.map((path) => join(path, "Contents", "Resources", "bin"));
const dockerCommandPath = [
  ...dockerDesktopBinPaths,
  "/opt/homebrew/bin",
  "/usr/local/bin",
  "/usr/bin",
  "/bin",
  "/usr/sbin",
  "/sbin"
].join(":");

export function getRuntimePaths(): RuntimePaths {
  const root = runtimeRoot;
  const native = join(root, "native");
  const docker = join(root, "docker", "comfyui");
  const ollama = join(root, "llm", "ollama");
  const loraRuntime = join(root, "lora-runtime");
  return {
    root,
    llm: join(root, "llm"),
    ollama,
    ollamaModels: join(ollama, "models"),
    ollamaLog: join(ollama, "logs", "ollama.log"),
    ollamaPid: join(ollama, "ollama.pid"),
    models: join(root, "models"),
    checkpoints: join(root, "models", "checkpoints"),
    loras: join(root, "models", "loras"),
    vae: join(root, "models", "vae"),
    outputs: join(root, "outputs"),
    workflows: join(root, "workflows"),
    loraProjects: join(root, "lora-projects"),
    loraRuntime,
    loraScripts: join(loraRuntime, "sd-scripts"),
    loraVenv: join(loraRuntime, "venv"),
    loraPython: join(loraRuntime, "venv", "bin", "python"),
    loraLog: join(loraRuntime, "logs", "lora-training.log"),
    native,
    nativeComfy: join(native, "ComfyUI"),
    nativeVenv: join(native, "venv"),
    nativePython: join(native, "venv", "bin", "python"),
    nativeLog: join(native, "logs", "comfyui.log"),
    nativePid: join(native, "comfyui.pid"),
    docker,
    logs: join(root, "logs"),
    downloads: join(root, "downloads"),
    composeFile: join(docker, "docker-compose.yml")
  };
}

export async function ensureRuntimeFiles(): Promise<RuntimePaths> {
  const paths = getRuntimePaths();
  await Promise.all([
    mkdir(paths.checkpoints, { recursive: true }),
    mkdir(paths.loras, { recursive: true }),
    mkdir(paths.vae, { recursive: true }),
    mkdir(paths.outputs, { recursive: true }),
    mkdir(paths.workflows, { recursive: true }),
    mkdir(paths.loraProjects, { recursive: true }),
    mkdir(paths.loraRuntime, { recursive: true }),
    mkdir(join(paths.loraRuntime, "logs"), { recursive: true }),
    mkdir(paths.native, { recursive: true }),
    mkdir(join(paths.native, "logs"), { recursive: true }),
    mkdir(paths.ollamaModels, { recursive: true }),
    mkdir(join(paths.ollama, "logs"), { recursive: true }),
    mkdir(paths.docker, { recursive: true }),
    mkdir(paths.logs, { recursive: true }),
    mkdir(paths.downloads, { recursive: true })
  ]);
  await writeFile(join(paths.docker, "Dockerfile"), dockerfileText(), "utf8");
  await writeFile(paths.composeFile, composeText(paths), "utf8");
  await writeFile(join(paths.workflows, "phase1-test-generation.json"), JSON.stringify(testWorkflowTemplate(), null, 2), "utf8");
  return paths;
}

export async function getAppSettings(): Promise<AppSettings> {
  try {
    const text = await readFile(settingsPath, "utf8");
    const parsed = JSON.parse(text) as Partial<AppSettings>;
    return {
      civitaiApiKey: parsed.civitaiApiKey ?? "",
      huggingFaceApiKey: parsed.huggingFaceApiKey ?? "",
      ollamaUrl: parsed.ollamaUrl ?? "http://127.0.0.1:11434",
      ollamaModel: parsed.ollamaModel ?? "qwen2.5:3b-instruct"
    };
  } catch {
    return { civitaiApiKey: "", huggingFaceApiKey: "", ollamaUrl: "http://127.0.0.1:11434", ollamaModel: "qwen2.5:3b-instruct" };
  }
}

export async function saveAppSettings(settings: AppSettings): Promise<AppSettings> {
  await mkdir(appDataRoot, { recursive: true });
  const safeSettings: AppSettings = {
    civitaiApiKey: settings.civitaiApiKey.trim(),
    huggingFaceApiKey: settings.huggingFaceApiKey.trim(),
    ollamaUrl: settings.ollamaUrl.trim() || "http://127.0.0.1:11434",
    ollamaModel: settings.ollamaModel.trim() || "qwen2.5:3b-instruct"
  };
  await writeFile(settingsPath, JSON.stringify(safeSettings, null, 2), "utf8");
  return safeSettings;
}

export async function getStatus(comfyUrl = `http://127.0.0.1:${comfyPort}`): Promise<RuntimeStatus> {
  const paths = getRuntimePaths();
  const [
    dockerDesktopPath,
    docker,
    daemon,
    compose,
    pathsReady,
    nativeInstalled,
    nativeRunning,
    composeReady,
    comfyReachable,
    checkpoints,
    loras,
    outputs,
    containerState
  ] =
    await Promise.all([
      findDockerDesktop(),
      commandVersion("docker", ["--version"]),
      getDockerDaemonStatus(),
      getComposeCommand(),
      exists(paths.models),
      exists(paths.nativePython),
      isNativeRunning(),
      exists(paths.composeFile),
      checkComfy(comfyUrl),
      listModelFiles(paths.checkpoints),
      listModelFiles(paths.loras),
      listFiles(paths.outputs),
      getContainerState()
    ]);

  return {
    runtimeRoot: paths.root,
    activeRuntime: "macos",
    nativeReady: pathsReady && nativeInstalled,
    nativeInstalled,
    nativeRunning,
    nativePid: await readNativePid(),
    dockerDesktopInstalled: Boolean(dockerDesktopPath),
    dockerDesktopPath,
    dockerAvailable: docker.ok,
    dockerVersion: docker.text,
    dockerDaemonReady: daemon.ok,
    dockerDaemonMessage: daemon.text,
    composeAvailable: Boolean(compose),
    composeCommand: compose?.join(" ") ?? null,
    pathsReady,
    composeReady,
    comfyReachable,
    comfyUrl,
    containerState,
    checkpoints,
    loras,
    outputs
  };
}

export async function initNativeRuntime(onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  onLog({ level: "info", line: `macOS runtime: ${paths.native}` });
  onLog({ level: "info", line: "共有モデルディレクトリと出力先を準備しました。" });
  return "macOSランタイムを初期化しました。";
}

export async function installNativeRuntime(onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  if (!(await exists(paths.nativeComfy))) {
    await spawnNativeCommand("git", ["clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", paths.nativeComfy], onLog);
  } else {
    onLog({ level: "info", line: "ComfyUI は既に存在します。" });
  }

  if (!(await exists(paths.nativePython))) {
    await spawnNativeCommand("python3", ["-m", "venv", paths.nativeVenv], onLog);
  } else {
    onLog({ level: "info", line: "venv は既に存在します。" });
  }

  await writeNativeExtraModelPaths(paths);
  await spawnNativeCommand(paths.nativePython, ["-m", "pip", "install", "--upgrade", "pip"], onLog);
  await spawnNativeCommand(paths.nativePython, ["-m", "pip", "install", "-r", join(paths.nativeComfy, "requirements.txt")], onLog);
  return "macOS ComfyUI環境のインストールが完了しました。";
}

export async function installLoraRuntime(onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  if (!(await exists(paths.loraScripts))) {
    await spawnNativeCommand("git", ["clone", "--depth", "1", "https://github.com/kohya-ss/sd-scripts.git", paths.loraScripts], onLog);
  } else {
    onLog({ level: "info", line: "sd-scripts は既に存在します。" });
  }

  if (!(await exists(paths.loraPython))) {
    await spawnNativeCommand("python3", ["-m", "venv", paths.loraVenv], onLog);
  } else {
    onLog({ level: "info", line: "LoRA venv は既に存在します。" });
  }

  await spawnNativeCommand(paths.loraPython, ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], onLog);
  await spawnNativeCommand(paths.loraPython, ["-m", "pip", "install", "-r", join(paths.loraScripts, "requirements.txt")], onLog, {
    cwd: paths.loraScripts
  });
  return "LoRA学習環境のインストールが完了しました。";
}

export async function startNativeRuntime(onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  if (!(await exists(paths.nativePython)) || !(await exists(join(paths.nativeComfy, "main.py")))) {
    throw new Error("macOS ComfyUI環境が未インストールです。先に「インストール」を実行してください。");
  }
  if (await checkComfy(`http://127.0.0.1:${comfyPort}`)) {
    return "ComfyUI は既に接続可能です。";
  }

  await writeNativeExtraModelPaths(paths);
  await writeFile(paths.nativeLog, "", "utf8");
  onLog({ level: "info", line: "macOS ComfyUI を起動します。" });
  const logFd = openSync(paths.nativeLog, "a");
  const child = spawn(
    paths.nativePython,
    [
      "main.py",
      "--listen",
      "127.0.0.1",
      "--port",
      String(comfyPort),
      "--output-directory",
      paths.outputs
    ],
    {
      cwd: paths.nativeComfy,
      detached: true,
      stdio: ["ignore", logFd, logFd],
      env: {
        ...process.env,
        PYTORCH_ENABLE_MPS_FALLBACK: "1",
        PYTHONUNBUFFERED: "1"
      }
    }
  );
  closeSync(logFd);
  nativeProcess = child;
  await writeFile(paths.nativePid, String(child.pid ?? ""), "utf8");
  onLog({ level: "info", line: `ComfyUIログ: ${paths.nativeLog}` });
  child.unref();

  const ready = await waitForComfy(`http://127.0.0.1:${comfyPort}`, 120000);
  if (!ready) {
    throw new Error("ComfyUI の起動を待ちましたが、API接続を確認できませんでした。ログを確認してください。");
  }
  return "macOS ComfyUI を起動しました。";
}

export async function stopNativeRuntime(onLog: RuntimeLogSink): Promise<string> {
  const pid = await readNativePid();
  if (!pid) {
    return "macOS ComfyUI のPIDはありません。";
  }
  onLog({ level: "info", line: `macOS ComfyUI を停止します: pid=${pid}` });
  try {
    process.kill(pid, "SIGTERM");
  } catch (error) {
    onLog({ level: "stderr", line: normalizeCommandError(error) });
  }
  await delay(1500);
  await unlink(getRuntimePaths().nativePid).catch(() => undefined);
  nativeProcess = null;
  return "macOS ComfyUI を停止しました。";
}

export async function showNativeLogs(onLog: RuntimeLogSink): Promise<string> {
  const paths = getRuntimePaths();
  if (!(await exists(paths.nativeLog))) {
    onLog({ level: "info", line: "macOS ComfyUIログはまだありません。" });
    return "macOS ComfyUIログはまだありません。";
  }
  const text = await readFile(paths.nativeLog, "utf8");
  for (const line of text.split(/\r?\n/).slice(-200)) {
    if (line.trim()) onLog({ level: line.includes("Traceback") || line.includes("Error") ? "stderr" : "stdout", line });
  }
  return "macOS ComfyUIログを取得しました。";
}

export async function cleanupNativeRuntime(onLog: RuntimeLogSink): Promise<string> {
  onLog({ level: "info", line: "macOS ComfyUI環境を削除します。" });
  await stopNativeRuntime(onLog).catch((error) => onLog({ level: "stderr", line: normalizeCommandError(error) }));
  await rm(getRuntimePaths().native, { recursive: true, force: true });
  return "macOS ComfyUI環境を削除しました。モデル/出力は残しています。";
}

export async function cleanupDockerRuntime(onLog: RuntimeLogSink): Promise<string> {
  const paths = getRuntimePaths();
  onLog({ level: "info", line: "Dockerコンテナ/イメージ/生成ファイルを後始末します。" });
  if ((await getDockerDaemonStatus()).ok) {
    await spawnDockerFile("docker", ["rm", "-f", "illustmk-comfyui"], onLog).catch((error) =>
      onLog({ level: "stderr", line: normalizeCommandError(error) })
    );
    await spawnDockerFile("docker", ["rmi", "illustmk-comfyui:phase1"], onLog).catch((error) =>
      onLog({ level: "stderr", line: normalizeCommandError(error) })
    );
  } else {
    onLog({ level: "info", line: "Docker daemon未接続のため、コンテナ/イメージ削除はスキップします。" });
  }
  await rm(join(paths.root, "docker"), { recursive: true, force: true });
  return "Dockerランタイム資産を後始末しました。Docker Desktop本体は削除していません。";
}

export async function startDockerEngine(): Promise<string> {
  const dockerDesktopPath = await findDockerDesktop();
  if (!dockerDesktopPath) {
    throw new Error("Docker Desktop.app が見つかりません。/Applications に Docker Desktop をインストールしてください。");
  }

  const current = await getDockerDaemonStatus();
  if (current.ok) {
    return "Docker daemon はすでに接続可能です。";
  }

  await execFileAsync("open", [dockerDesktopPath], { timeout: 10000 });
  const ready = await waitForDockerDaemon(true, 120000);
  if (!ready.ok) {
    throw new Error(
      [
        "Docker Desktop の起動を要求しましたが、Docker daemon 接続を確認できませんでした。",
        "初回起動の確認画面や権限確認が Docker Desktop 側に出ていないか確認してください。",
        ready.text ? `詳細: ${ready.text}` : null
      ]
        .filter(Boolean)
        .join("\n")
    );
  }

  return "Docker Desktop を起動し、Docker daemon 接続を確認しました。";
}

export async function stopDockerEngine(): Promise<string> {
  const dockerDesktopPath = await findDockerDesktop();
  if (!dockerDesktopPath) {
    throw new Error("Docker Desktop.app が見つかりません。");
  }

  const current = await getDockerDaemonStatus();
  if (!current.ok) {
    return "Docker daemon はすでに未接続です。";
  }

  await execFileAsync("osascript", ["-e", 'tell application "Docker" to quit'], { timeout: 10000 });
  const stopped = await waitForDockerDaemon(false, 60000);
  if (!stopped.ok) {
    throw new Error("Docker Desktop に終了要求を出しましたが、Docker daemon がまだ接続可能です。");
  }

  return "Docker Desktop を終了しました。";
}

export async function buildRuntime(): Promise<string> {
  const paths = await ensureRuntimeFiles();
  await assertDockerDaemonReady();
  const { stdout, stderr } = await execDockerFile("docker", ["build", "-t", "illustmk-comfyui:phase1", paths.docker], {
    maxBuffer: 1024 * 1024 * 20
  });
  return [stdout, stderr].filter(Boolean).join("\n");
}

export async function buildRuntimeWithLogs(onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  onLog({ level: "info", line: `Runtime: ${paths.root}` });
  await assertDockerDaemonReady();
  onLog({ level: "info", line: "docker build を開始します。" });
  await spawnDockerFile("docker", ["build", "--progress=plain", "-t", "illustmk-comfyui:phase1", paths.docker], onLog);
  onLog({ level: "info", line: "docker build が完了しました。" });
  return "Dockerイメージの構築が完了しました。";
}

export async function startRuntime(): Promise<string> {
  const paths = await ensureRuntimeFiles();
  await assertDockerDaemonReady();
  await buildRuntime();
  const state = await getContainerState();
  if (state === "exited" || state === "created") {
    const { stdout, stderr } = await execDockerFile("docker", ["start", "illustmk-comfyui"], {
      maxBuffer: 1024 * 1024 * 20
    });
    return [stdout, stderr].filter(Boolean).join("\n");
  }
  if (state === "running") {
    return "illustmk-comfyui is already running.";
  }
  const { stdout, stderr } = await execDockerFile("docker", dockerRunArgs(paths), { maxBuffer: 1024 * 1024 * 20 });
  return [stdout, stderr].filter(Boolean).join("\n");
}

export async function startRuntimeWithLogs(onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  await assertDockerDaemonReady();
  await buildRuntimeWithLogs(onLog);
  const state = await getContainerState();
  onLog({ level: "info", line: `Container state: ${state}` });
  if (state === "exited" || state === "created") {
    await spawnDockerFile("docker", ["start", "illustmk-comfyui"], onLog);
    return "ComfyUIコンテナを起動しました。";
  }
  if (state === "running") {
    return "ComfyUIコンテナはすでに起動中です。";
  }
  await spawnDockerFile("docker", dockerRunArgs(paths), onLog);
  return "ComfyUIコンテナを作成して起動しました。";
}

export async function stopRuntime(): Promise<string> {
  await assertDockerDaemonReady();
  const { stdout, stderr } = await execDockerFile("docker", ["stop", "illustmk-comfyui"], { maxBuffer: 1024 * 1024 * 20 });
  return [stdout, stderr].filter(Boolean).join("\n");
}

export async function stopRuntimeWithLogs(onLog: RuntimeLogSink): Promise<string> {
  await assertDockerDaemonReady();
  onLog({ level: "info", line: "ComfyUIコンテナを停止します。" });
  await spawnDockerFile("docker", ["stop", "illustmk-comfyui"], onLog);
  return "ComfyUIコンテナを停止しました。";
}

export async function showComfyContainerLogs(onLog: RuntimeLogSink): Promise<string> {
  await assertDockerDaemonReady();
  onLog({ level: "info", line: "ComfyUIコンテナログを取得します。" });
  await spawnDockerFile("docker", ["logs", "--tail", "200", "illustmk-comfyui"], onLog);
  return "ComfyUIコンテナログを取得しました。";
}

export async function downloadModel(request: DownloadRequest, onLog?: RuntimeLogSink): Promise<DownloadResult> {
  const paths = await ensureRuntimeFiles();
  const targetDir = paths[request.targetType];
  const url = new URL(request.url);
  const fileName = sanitizeFileName(request.fileName || basename(url.pathname));
  if (!fileName) {
    throw new Error("保存ファイル名を決定できません。");
  }

  const targetPath = join(targetDir, fileName);
  const headers: Record<string, string> = {
    "User-Agent": "illustMK/0.1",
    ...(request.source === "civitai" || url.hostname.includes("civitai.com") ? await civitaiAuthHeader() : {})
  };
  const response = await fetch(await withCivitaiToken(url), { headers });
  if (!response.ok || !response.body) {
    throw new Error(downloadErrorMessage(response.status, url));
  }

  const totalBytes = Number(response.headers.get("content-length") ?? 0);
  let receivedBytes = 0;
  let lastProgressLog = 0;
  onLog?.({ level: "info", line: `保存先: ${targetPath}` });
  onLog?.({
    level: "info",
    line: totalBytes > 0 ? `ダウンロード開始: 0 B / ${formatBytes(totalBytes)}` : "ダウンロード開始: サイズ不明"
  });

  const progressStream = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      receivedBytes += chunk.length;
      const now = Date.now();
      if (now - lastProgressLog >= 1000) {
        lastProgressLog = now;
        onLog?.({ level: "info", line: downloadProgressLine(receivedBytes, totalBytes) });
      }
      callback(null, chunk);
    }
  });

  await pipeline(response.body as unknown as NodeJS.ReadableStream, progressStream, createWriteStream(targetPath));
  onLog?.({ level: "info", line: downloadProgressLine(receivedBytes, totalBytes) });
  const digest = await sha256File(targetPath);
  if (request.sha256 && digest.toLowerCase() !== request.sha256.toLowerCase()) {
    throw new Error(`SHA256が一致しません: ${digest}`);
  }
  onLog?.({ level: "info", line: `SHA256: ${digest}` });

  return {
    ok: true,
    path: targetPath,
    sha256: digest,
    message: "ダウンロードが完了しました。"
  };
}

export async function deleteModelFile(request: DeleteModelRequest): Promise<string> {
  const paths = getRuntimePaths();
  const targetDir = paths[request.targetType];
  const fileName = sanitizeFileName(request.fileName);
  if (!fileName || fileName !== request.fileName) {
    throw new Error("削除対象のファイル名が不正です。");
  }

  await rm(join(targetDir, fileName), { force: true });
  return `${fileName} を削除しました。`;
}

export async function searchCivitaiModels(options: {
  query?: string;
  sort?: "Most Downloaded" | "Highest Rated" | "Newest";
  period?: "AllTime" | "Year" | "Month" | "Week" | "Day";
  limit?: number;
  nsfw?: boolean;
  baseModels?: string[];
}): Promise<CivitaiModelCard[]> {
  const params = new URLSearchParams({
    limit: String(Math.max(options.limit ?? 10, 50)),
    types: "Checkpoint",
    sort: options.sort ?? "Most Downloaded",
    period: options.period ?? "Month",
    nsfw: String(options.nsfw ?? false),
    primaryFileOnly: "true"
  });
  if (options.query?.trim()) {
    params.set("query", options.query.trim());
  }

  const response = await fetch(`https://civitai.com/api/v1/models?${params.toString()}`, {
    headers: {
      Accept: "application/json",
      "User-Agent": "illustMK/0.1",
      ...(await civitaiAuthHeader())
    }
  });
  if (!response.ok) {
    throw new Error(`Civitai API取得に失敗しました: HTTP ${response.status}`);
  }

  const json = (await response.json()) as { items?: unknown[] };
  return (json.items ?? [])
    .flatMap(normalizeCivitaiModel)
    .filter((model) => isAllowedBaseModel(model, options.baseModels ?? defaultBaseModelFilters))
    .slice(0, options.limit ?? 10);
}

export async function searchHuggingFaceLlmModels(options: { query: string; limit?: number }): Promise<HuggingFaceLlmModelCard[]> {
  const query = options.query.trim();
  if (!query) throw new Error("検索語を入力してください。");
  const params = new URLSearchParams({
    search: query,
    filter: "gguf",
    sort: "downloads",
    direction: "-1",
    limit: String(options.limit ?? 20),
    full: "true"
  });
  const response = await fetch(`https://huggingface.co/api/models?${params.toString()}`, {
    headers: {
      Accept: "application/json",
      ...(await huggingFaceAuthHeader())
    },
    signal: AbortSignal.timeout(20000)
  });
  if (!response.ok) {
    throw new Error(`Hugging Face検索に失敗しました: HTTP ${response.status}`);
  }
  const json = (await response.json()) as Array<Record<string, unknown>>;
  const cards = json.map(normalizeHuggingFaceModelCard).filter((model) => model.ggufFiles.length || model.tags.includes("gguf"));
  return Promise.all(cards.map(withHuggingFaceModelFileSizes));
}

export async function getOllamaStatus(ollamaUrl = "http://127.0.0.1:11434"): Promise<OllamaStatus> {
  const appPath = await findOllamaApp();
  const paths = getRuntimePaths();
  const dataLinked = await isOllamaDataLinked();
  try {
    const response = await fetch(`${ollamaUrl.replace(/\/$/, "")}/api/tags`, { signal: AbortSignal.timeout(3000) });
    if (!response.ok) {
      return {
        url: ollamaUrl,
        reachable: false,
        installed: Boolean(appPath),
        appPath,
        dataLinked,
        modelPath: paths.ollamaModels,
        models: [],
        message: `Ollama API HTTP ${response.status}`
      };
    }
    const json = (await response.json()) as { models?: Array<Record<string, unknown>> };
    const models = (json.models ?? []).map((model) => ({
      name: String(model.name ?? ""),
      sizeBytes: Number(model.size ?? 0),
      sizeLabel: formatBytes(Number(model.size ?? 0)),
      modifiedAt: model.modified_at ? String(model.modified_at) : null
    })).filter((model) => model.name);
    return {
      url: ollamaUrl,
      reachable: true,
      installed: Boolean(appPath),
      appPath,
      dataLinked,
      modelPath: paths.ollamaModels,
      models,
      message: "Ollama API 接続OK"
    };
  } catch (error) {
    return {
      url: ollamaUrl,
      reachable: false,
      installed: Boolean(appPath),
      appPath,
      dataLinked,
      modelPath: paths.ollamaModels,
      models: [],
      message: normalizeCommandError(error)
    };
  }
}

export async function startOllama(): Promise<string> {
  const paths = await ensureRuntimeFiles();
  const linked = await isOllamaDataLinked();
  if ((await getOllamaStatus()).reachable && linked) {
    return "Ollama API は既に接続可能です。";
  }
  if (!linked) {
    await stopOllama().catch(() => undefined);
    await ensureOllamaDataLink();
  }
  const ollamaBin = await findOllamaBinary();
  if (ollamaBin) {
    await writeFile(paths.ollamaLog, "", "utf8");
    const logFd = openSync(paths.ollamaLog, "a");
    const child = spawn(ollamaBin, ["serve"], {
      detached: true,
      stdio: ["ignore", logFd, logFd],
      env: {
        ...process.env,
        OLLAMA_HOST: "127.0.0.1:11434",
        OLLAMA_MODELS: paths.ollamaModels
      }
    });
    closeSync(logFd);
    ollamaProcess = child;
    await writeFile(paths.ollamaPid, String(child.pid ?? ""), "utf8");
    child.unref();
    const ready = await waitForOllama("http://127.0.0.1:11434", 30000);
    if (ready) {
      return `Ollama API を起動しました。モデル保存先: ${paths.ollamaModels}`;
    }
  }

  const appPath = await findOllamaApp();
  if (!appPath) {
    throw new Error("Ollama.app が見つかりません。公式サイトからOllamaをインストールしてください。");
  }
  await execFileAsync("open", [appPath], { timeout: 10000 });
  return "Ollama.app を起動しました。アプリ起動では保存先が標準設定になる場合があります。";
}

export async function installOllama(onLog: RuntimeLogSink): Promise<string> {
  await ensureRuntimeFiles();
  await ensureOllamaDataLink(onLog);

  const current = await findOllamaApp();
  if (current) {
    onLog({ level: "info", line: `Ollama.app は既に存在します: ${current}` });
    return `Ollama.app は既にインストール済みです。Ollamaデータ保存先: ${getRuntimePaths().ollama}`;
  }

  const paths = await ensureRuntimeFiles();
  const dmgPath = join(paths.downloads, "Ollama.dmg");
  onLog({ level: "info", line: "Ollama.dmg を公式サイトからダウンロードします。" });
  await downloadFileWithProgress("https://ollama.com/download/Ollama.dmg", dmgPath, onLog);

  onLog({ level: "info", line: "Ollama.dmg をマウントします。" });
  const { stdout } = await execFileAsync("hdiutil", ["attach", dmgPath, "-nobrowse", "-readonly"], { timeout: 60000 });
  const mountPoint = parseMountPoint(stdout);
  if (!mountPoint) {
    throw new Error("Ollama.dmg のマウント先を特定できませんでした。");
  }

  try {
    const sourceApp = join(mountPoint, "Ollama.app");
    const destination = await copyOllamaApp(sourceApp, onLog);
    return `Ollama.app をインストールしました: ${destination}`;
  } finally {
    await execFileAsync("hdiutil", ["detach", mountPoint], { timeout: 60000 }).catch((error) =>
      onLog({ level: "stderr", line: normalizeCommandError(error) })
    );
  }
}

export async function stopOllama(): Promise<string> {
  const pid = await readPidFile(getRuntimePaths().ollamaPid);
  if (pid) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      // The app fallback may be running without our pid file target.
    }
    await unlink(getRuntimePaths().ollamaPid).catch(() => undefined);
    ollamaProcess = null;
  }
  await execFileAsync("osascript", ["-e", 'tell application "Ollama" to quit'], { timeout: 10000 }).catch(() => undefined);
  return "Ollamaを終了しました。";
}

export async function uninstallOllama(onLog: RuntimeLogSink): Promise<string> {
  await stopOllama().catch((error) => onLog({ level: "stderr", line: normalizeCommandError(error) }));
  let removed = 0;
  for (const appPath of ollamaAppPaths) {
    if (await exists(appPath)) {
      onLog({ level: "info", line: `削除: ${appPath}` });
      await rm(appPath, { recursive: true, force: true });
      removed += 1;
    }
  }
  return removed ? "Ollama.app をアンインストールしました。" : "削除対象のOllama.appは見つかりませんでした。";
}

export async function deleteOllamaModelStorage(onLog: RuntimeLogSink): Promise<string> {
  await stopOllama().catch((error) => onLog({ level: "stderr", line: normalizeCommandError(error) }));
  await ensureOllamaDataLink(onLog);
  const modelsPath = getRuntimePaths().ollamaModels;
  onLog({ level: "info", line: `Ollamaモデル保存先を削除します: ${modelsPath}` });
  await rm(modelsPath, { recursive: true, force: true });
  await mkdir(modelsPath, { recursive: true });
  return "Ollamaのモデル保存先を削除しました。";
}

export async function pullOllamaModel(ollamaUrl: string, model: string, onLog: RuntimeLogSink): Promise<string> {
  const modelName = model.trim();
  if (!modelName) throw new Error("Ollamaモデル名を入力してください。");
  const status = await getOllamaStatus(ollamaUrl);
  if (!status.dataLinked) {
    throw new Error("Ollama保存先がDataPathに向いていません。先にLLM / Ollamaパネルの「起動」を実行して保存先を切替えてください。");
  }
  if (!status.reachable) {
    throw new Error("Ollama APIに接続できません。先にLLM / Ollamaパネルの「起動」を実行してください。");
  }
  onLog({ level: "info", line: `Ollama pull: ${modelName}` });
  const response = await fetch(`${ollamaUrl.replace(/\/$/, "")}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: modelName, stream: true })
  });
  if (!response.ok || !response.body) {
    throw new Error(`Ollama pullに失敗しました: HTTP ${response.status}`);
  }

  let buffer = "";
  const decoder = new TextDecoder();
  for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
    buffer += decoder.decode(chunk, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      emitOllamaPullLine(line, onLog);
    }
  }
  if (buffer.trim()) emitOllamaPullLine(buffer, onLog);
  return `${modelName} のpullが完了しました。`;
}

export async function deleteOllamaModel(ollamaUrl: string, model: string): Promise<string> {
  const modelName = model.trim();
  if (!modelName) throw new Error("削除するOllamaモデル名を指定してください。");
  const response = await fetch(`${ollamaUrl.replace(/\/$/, "")}/api/delete`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: modelName })
  });
  if (!response.ok) {
    throw new Error(`Ollamaモデル削除に失敗しました: HTTP ${response.status}`);
  }
  return `${modelName} を削除しました。`;
}

export async function enhancePromptWithOllama(request: PromptEnhanceRequest): Promise<PromptEnhanceResult> {
  if (!request.japanesePrompt.trim()) {
    throw new Error("日本語プロンプトを入力してください。");
  }
  if (!request.model.trim()) {
    throw new Error("Ollamaモデル名を入力してください。");
  }

  const instruction = [
    "You are an expert prompt engineer for anime image generation models such as SDXL, Illustrious, Pony, and Animagine.",
    "Convert the user's Japanese request into English image-generation prompts.",
    "Return strict JSON only with keys: positivePrompt, negativePrompt, notes.",
    "positivePrompt should be comma-separated English tags and short phrases.",
    "negativePrompt should be comma-separated English tags.",
    "Use only the Japanese request and the selected tag details in this call as source material.",
    "Do not reuse, infer from, or continue any previous prompt, previous answer, chat history, or unrelated context.",
    "Do not invent concrete visual details that are not present in the Japanese request, its definitions, or the selected tag details.",
    "The Japanese request may contain definitions, footnotes, detail sections, or explanatory text after the main sentence.",
    "Treat definitions and footnotes as mandatory visual requirements, not commentary.",
    "If a term is defined, expand the term using its definition instead of only translating the label.",
    "If selected tag details are provided, integrate them into positivePrompt as mandatory visual requirements.",
    "When multiple tags are selected, keep each tag's details conceptually separate and do not assign one tag's details to the wrong subject.",
    "Do not output tag detail labels such as 'school_uniform detail:' in positivePrompt; convert them into ordinary image-generation phrases.",
    "Preserve concrete visual details about clothing, shape, color, material, pose, expression, setting, lighting, and composition.",
    promptEnhanceModeInstruction(request.mode ?? "balanced-detail"),
    "Do not include markdown."
  ].join("\n");
  const selectedTagText = (request.selectedTags ?? [])
    .map((tag) => `${tag.englishName || tag.name} detail: ${tag.englishDescription}`)
    .filter((line) => line.trim())
    .join("\n");
  const userText = [
    `Japanese request: ${request.japanesePrompt}`,
    selectedTagText ? `Selected tag details:\n${selectedTagText}` : "Selected tag details: none"
  ].join("\n");

  const response = await fetch(`${request.ollamaUrl.replace(/\/$/, "")}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      stream: false,
      format: "json",
      prompt: `${instruction}\n\n${userText}`
    })
  });
  if (!response.ok) {
    throw new Error(`Ollama prompt変換に失敗しました: HTTP ${response.status}`);
  }
  const json = (await response.json()) as { response?: string };
  return parsePromptEnhanceResponse(json.response ?? "");
}

function promptEnhanceModeInstruction(mode: PromptEnhanceMode): string {
  if (mode === "footnote-strict") {
    return [
      "Mode: footnote-strict.",
      "Prioritize definitions, footnotes, and detail sections as strongly as the main sentence.",
      "Do not omit any concrete visual detail from explanatory sections.",
      "If the request says things like 'とは', '定義', '詳細', '脚注', 'つまり', or '以下', treat the following text as required prompt material.",
      "The notes field must briefly mention which definition or footnote details were incorporated."
    ].join("\n");
  }
  if (mode === "main-subject") {
    return [
      "Mode: main-subject.",
      "Prioritize the main subject sentence and use definitions only when they clarify visible appearance.",
      "Keep the prompt compact, but do not discard concrete visible details."
    ].join("\n");
  }
  if (mode === "tag-dense") {
    return [
      "Mode: tag-dense.",
      "Produce a dense comma-separated prompt suitable for anime checkpoint models.",
      "Break definitions into many concrete visual tags and short phrases.",
      "Prefer explicit clothing parts, accessories, materials, colors, and silhouette terms over broad labels."
    ].join("\n");
  }
  return [
    "Mode: balanced-detail.",
    "Recommended default: preserve the main subject while also expanding definitions and footnotes into concrete visual prompt details.",
    "Do not over-compress defined terms such as uniforms, hairstyles, outfits, props, or backgrounds into a single generic tag.",
    "If details are omitted, explain why in notes."
  ].join("\n");
}

export async function getLlmChatHistory(): Promise<LlmChatMessage[]> {
  try {
    const text = await readFile(llmChatHistoryPath, "utf8");
    const parsed = JSON.parse(text) as LlmChatMessage[];
    return Array.isArray(parsed) ? parsed.filter(isLlmChatMessage) : [];
  } catch {
    return [];
  }
}

export async function clearLlmChatHistory(): Promise<LlmChatMessage[]> {
  await mkdir(appDataRoot, { recursive: true });
  await writeFile(llmChatHistoryPath, "[]", "utf8");
  return [];
}

export async function sendLlmChatPrompt(request: LlmChatRequest): Promise<LlmChatMessage[]> {
  const prompt = request.prompt.trim();
  if (!prompt) throw new Error("LLMへ送るプロンプトを入力してください。");
  if (!request.model.trim()) throw new Error("Ollamaモデル名を入力してください。");
  const status = await getOllamaStatus(request.ollamaUrl);
  if (!status.dataLinked) {
    throw new Error("Ollama保存先がDataPathに向いていません。先に環境タブのLLM / Ollamaパネルで「起動」を実行してください。");
  }
  if (!status.reachable) {
    throw new Error("Ollama APIに接続できません。先に環境タブのLLM / Ollamaパネルで「起動」を実行してください。");
  }

  const userMessage: LlmChatMessage = {
    id: randomUUID(),
    role: "user",
    content: prompt,
    model: request.model,
    createdAt: new Date().toISOString()
  };
  const response = await fetch(`${request.ollamaUrl.replace(/\/$/, "")}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      stream: false,
      prompt
    })
  });
  if (!response.ok) {
    throw new Error(`Ollama LLMテストに失敗しました: HTTP ${response.status}`);
  }
  const json = (await response.json()) as { response?: string };
  const assistantMessage: LlmChatMessage = {
    id: randomUUID(),
    role: "assistant",
    content: String(json.response ?? ""),
    model: request.model,
    createdAt: new Date().toISOString()
  };
  const history = [...(await getLlmChatHistory()), userMessage, assistantMessage].slice(-200);
  await saveLlmChatHistory(history);
  return history;
}

export async function getIllustrationTags(): Promise<IllustrationTag[]> {
  try {
    const text = await readFile(illustrationTagsPath, "utf8");
    const parsed = JSON.parse(text) as unknown[];
    return Array.isArray(parsed) ? parsed.map(normalizeIllustrationTag).filter((tag): tag is IllustrationTag => Boolean(tag)) : [];
  } catch {
    return [];
  }
}

export async function generateTagJapaneseDescription(request: TagDescriptionRequest): Promise<string> {
  const tagName = request.tagName.trim();
  const prompt = request.prompt.trim();
  if (!tagName) throw new Error("タグ名を入力してください。");
  if (!prompt) throw new Error("タグ作成用プロンプトを入力してください。");
  if (!request.model.trim()) throw new Error("Ollamaモデル名を入力してください。");
  await assertOllamaReady(request.ollamaUrl);

  const instruction = [
    "あなたはAIイラスト生成向けのタグ辞書を作る補助AIです。",
    "ユーザーのタグ名と依頼文を読み、イラスト生成時に役立つ日本語説明文を作ってください。",
    "衣装、形状、色、素材、雰囲気、避けたい曖昧さを具体化してください。",
    "出力は日本語本文だけ。Markdown、見出し、箇条書き記号は不要です。"
  ].join("\n");
  const response = await generateOllamaText(request.ollamaUrl, request.model, `${instruction}\n\nタグ名: ${tagName}\n依頼文: ${prompt}`);
  return response.trim();
}

export async function translateTagDescription(request: TagTranslationRequest): Promise<string> {
  const japaneseDescription = request.japaneseDescription.trim();
  if (!japaneseDescription) throw new Error("日本語説明文を入力してください。");
  if (!request.model.trim()) throw new Error("Ollamaモデル名を入力してください。");
  await assertOllamaReady(request.ollamaUrl);

  const instruction = [
    "Translate the following Japanese image-generation tag description into concise English.",
    "Write it as usable anime image generation prompt fragments.",
    "Prefer concrete comma-separated phrases. Do not add markdown."
  ].join("\n");
  const response = await generateOllamaText(request.ollamaUrl, request.model, `${instruction}\n\n${japaneseDescription}`);
  return response.trim();
}

export async function saveIllustrationTag(request: SaveIllustrationTagRequest): Promise<IllustrationTag[]> {
  const name = request.name.trim();
  const englishName = request.englishName.trim();
  const sourcePrompt = request.sourcePrompt.trim();
  const japaneseDescription = request.japaneseDescription.trim();
  const englishDescription = request.englishDescription.trim();
  if (!name) throw new Error("タグ名を入力してください。");
  if (!englishName) throw new Error("英名タグ名を入力してください。");
  if (!sourcePrompt) throw new Error("タグ作成用プロンプトを入力してください。");
  if (!japaneseDescription) throw new Error("日本語説明文を入力してください。");
  if (!englishDescription) throw new Error("英訳説明文を入力してください。");

  const now = new Date().toISOString();
  const tags = await getIllustrationTags();
  const existing = request.id ? tags.find((tag) => tag.id === request.id) : null;
  const nextTag: IllustrationTag = {
    id: existing?.id ?? randomUUID(),
    name,
    englishName,
    sourcePrompt,
    japaneseDescription,
    englishDescription,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now
  };
  const nextTags = existing
    ? tags.map((tag) => (tag.id === existing.id ? nextTag : tag))
    : [...tags, nextTag];
  await saveIllustrationTags(nextTags);
  return nextTags;
}

export async function deleteIllustrationTag(id: string): Promise<IllustrationTag[]> {
  const tags = (await getIllustrationTags()).filter((tag) => tag.id !== id);
  await saveIllustrationTags(tags);
  return tags;
}

export async function getSavedPromptHistory(): Promise<PromptHistoryEntry[]> {
  try {
    const text = await readFile(promptHistoryPath, "utf8");
    const parsed = JSON.parse(text) as PromptHistoryEntry[];
    return Array.isArray(parsed)
      ? parsed.filter(isPromptHistoryEntry).map((entry) => ({ ...entry, japanesePrompt: entry.japanesePrompt ?? "" }))
      : [];
  } catch {
    return [];
  }
}

export async function clearPromptHistory(): Promise<PromptHistoryEntry[]> {
  await mkdir(appDataRoot, { recursive: true });
  await writeFile(promptHistoryPath, "[]", "utf8");
  return [];
}

export async function getLoraTrainingProjects(): Promise<LoraTrainingProject[]> {
  try {
    const text = await readFile(loraTrainingProjectsPath, "utf8");
    const parsed = JSON.parse(text) as LoraTrainingProject[];
    return Array.isArray(parsed) ? parsed.filter(isLoraTrainingProject).map((project) => ({ ...project, sourceLora: project.sourceLora ?? "" })) : [];
  } catch {
    return [];
  }
}

export async function saveLoraTrainingProject(request: SaveLoraTrainingProjectRequest): Promise<LoraTrainingProject[]> {
  const name = request.name.trim();
  if (!name) throw new Error("LoRA名を入力してください。");
  if (!request.baseCheckpoint.trim()) throw new Error("ベースCheckpointを選択してください。");
  const paths = await ensureRuntimeFiles();
  const now = new Date().toISOString();
  const projects = await getLoraTrainingProjects();
  const existing = request.id ? projects.find((project) => project.id === request.id) : null;
  const id = existing?.id ?? randomUUID();
  const safeName = safeFileName(name);
  const projectRoot = join(paths.loraProjects, safeName);
  const datasetPath = request.datasetPath?.trim() || join(projectRoot, "dataset");
  const outputName = request.outputName?.trim() || `${safeName}.safetensors`;
  const project: LoraTrainingProject = {
    id,
    name,
    triggerWord: request.triggerWord.trim(),
    sourceLora: request.sourceLora?.trim() ?? existing?.sourceLora ?? "",
    baseCheckpoint: request.baseCheckpoint.trim(),
    datasetPath,
    outputName,
    resolution: request.resolution,
    repeats: request.repeats,
    epochs: request.epochs,
    networkDim: request.networkDim,
    learningRate: request.learningRate.trim() || "1e-4",
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
    status: "draft",
    message: "プロジェクトを作成しました。datasetに学習画像とcaptionを配置してください。"
  };
  await mkdir(datasetPath, { recursive: true });
  await mkdir(join(projectRoot, "config"), { recursive: true });
  await writeFile(join(projectRoot, "config", "lora-training.json"), JSON.stringify(project, null, 2), "utf8");
  const nextProjects = existing
    ? projects.map((item) => (item.id === existing.id ? project : item))
    : [project, ...projects];
  await saveLoraTrainingProjects(nextProjects);
  return nextProjects;
}

export async function startLoraTrainingProject(id: string): Promise<LoraTrainingProject[]> {
  const paths = await ensureRuntimeFiles();
  const projects = await getLoraTrainingProjects();
  const project = projects.find((item) => item.id === id);
  if (!project) throw new Error("LoRAプロジェクトが見つかりません。");
  const updated = await validateLoraTrainingProject(project, paths.loras);
  const nextProjects = projects.map((item) => (item.id === id ? updated : item));
  await saveLoraTrainingProjects(nextProjects);
  return nextProjects;
}

export async function trainLoraTrainingProject(id: string, onLog: RuntimeLogSink): Promise<string> {
  const paths = await ensureRuntimeFiles();
  if (!(await exists(paths.loraPython)) || !(await exists(join(paths.loraScripts, "sdxl_train_network.py")))) {
    throw new Error("LoRA学習環境が未インストールです。環境タブのLoRAパネルで「インストール」を実行してください。");
  }
  const projects = await getLoraTrainingProjects();
  const project = projects.find((item) => item.id === id);
  if (!project) throw new Error("LoRAプロジェクトが見つかりません。");
  const readyProject = await validateLoraTrainingProject(project, paths.loras);
  if (readyProject.status === "blocked") {
    await saveLoraTrainingProjects(projects.map((item) => (item.id === id ? readyProject : item)));
    throw new Error(readyProject.message);
  }

  const runningProject: LoraTrainingProject = {
    ...readyProject,
    status: "running",
    updatedAt: new Date().toISOString(),
    message: "LoRA学習を実行中です。ログは下段に表示します。"
  };
  await saveLoraTrainingProjects(projects.map((item) => (item.id === id ? runningProject : item)));

  await writeFile(paths.loraLog, "", "utf8");
  onLog({ level: "info", line: `LoRA学習ログ: ${paths.loraLog}` });
  const outputBaseName = project.outputName.replace(/\.safetensors$/i, "");
  const commandArgs = [
    "sdxl_train_network.py",
    "--pretrained_model_name_or_path",
    join(paths.checkpoints, project.baseCheckpoint),
    "--train_data_dir",
    project.datasetPath,
    "--output_dir",
    paths.loras,
    "--output_name",
    outputBaseName,
    "--save_model_as",
    "safetensors",
    "--resolution",
    `${project.resolution},${project.resolution}`,
    "--train_batch_size",
    "1",
    "--max_train_epochs",
    String(project.epochs),
    "--network_module",
    "networks.lora",
    "--network_dim",
    String(project.networkDim),
    "--learning_rate",
    project.learningRate,
    "--caption_extension",
    ".txt",
    "--enable_bucket",
    "--mixed_precision",
    "no"
  ];
  if (project.sourceLora.trim()) {
    commandArgs.push("--network_weights", join(paths.loras, project.sourceLora));
  }
  if (project.triggerWord.trim()) {
    onLog({ level: "info", line: `Trigger Word: ${project.triggerWord.trim()}` });
  }
  if (project.sourceLora.trim()) {
    onLog({ level: "info", line: `追加学習元LoRA: ${project.sourceLora.trim()}` });
  }

  try {
    await spawnNativeCommand(paths.loraPython, commandArgs, onLog, { cwd: paths.loraScripts });
  } catch (error) {
    const failedProjects = await getLoraTrainingProjects();
    const failedProject: LoraTrainingProject = {
      ...runningProject,
      status: "failed",
      updatedAt: new Date().toISOString(),
      message: error instanceof Error ? error.message : String(error)
    };
    await saveLoraTrainingProjects(failedProjects.map((item) => (item.id === id ? failedProject : item)));
    throw error;
  }

  const completedProjects = await getLoraTrainingProjects();
  const completedProject: LoraTrainingProject = {
    ...runningProject,
    status: "complete",
    updatedAt: new Date().toISOString(),
    message: `LoRA学習が完了しました。出力: ${join(paths.loras, `${outputBaseName}.safetensors`)}`
  };
  await saveLoraTrainingProjects(completedProjects.map((item) => (item.id === id ? completedProject : item)));
  return completedProject.message;
}

export async function savePromptHistoryEntry(request: SavePromptHistoryRequest): Promise<PromptHistoryEntry[]> {
  const entry: PromptHistoryEntry = {
    id: randomUUID(),
    japanesePrompt: request.japanesePrompt,
    prompt: request.prompt,
    negativePrompt: request.negativePrompt,
    checkpoint: request.checkpoint,
    width: request.width,
    height: request.height,
    steps: request.steps,
    cfg: request.cfg,
    seed: request.seed,
    imagePath: request.imagePath,
    imageDataUrl: request.imageDataUrl,
    createdAt: new Date().toISOString()
  };
  const history = [entry, ...(await getSavedPromptHistory()).filter((item) => item.japanesePrompt !== request.japanesePrompt || item.prompt !== request.prompt || item.negativePrompt !== request.negativePrompt)].slice(0, 100);
  await savePromptHistory(history);
  return history;
}

async function civitaiAuthHeader(): Promise<Record<string, string>> {
  const settings = await getAppSettings();
  return settings.civitaiApiKey ? { Authorization: `Bearer ${settings.civitaiApiKey}` } : {};
}

async function savePromptHistory(history: PromptHistoryEntry[]): Promise<void> {
  await mkdir(appDataRoot, { recursive: true });
  await writeFile(promptHistoryPath, JSON.stringify(history, null, 2), "utf8");
}

async function saveLoraTrainingProjects(projects: LoraTrainingProject[]): Promise<void> {
  await mkdir(appDataRoot, { recursive: true });
  await writeFile(loraTrainingProjectsPath, JSON.stringify(projects, null, 2), "utf8");
}

async function validateLoraTrainingProject(project: LoraTrainingProject, loraOutputPath: string): Promise<LoraTrainingProject> {
  const datasetFiles = await listFiles(project.datasetPath);
  const imageCount = datasetFiles.filter((file) => /\.(png|jpe?g|webp)$/i.test(file)).length;
  const captionCount = datasetFiles.filter((file) => /\.txt$/i.test(file)).length;
  const now = new Date().toISOString();
  if (project.sourceLora.trim() && !(await exists(join(loraOutputPath, project.sourceLora)))) {
    return {
      ...project,
      status: "blocked",
      updatedAt: now,
      message: `追加学習元LoRAが見つかりません: ${project.sourceLora}`
    };
  }
  if (imageCount === 0) {
    return {
      ...project,
      status: "blocked",
      updatedAt: now,
      message: "datasetに学習画像がありません。画像と必要なcaptionを追加してください。"
    };
  }
  await mkdir(loraOutputPath, { recursive: true });
  return {
    ...project,
    status: "ready",
    updatedAt: now,
    message: [
      "学習前チェックOKです。",
      `画像 ${imageCount}件 / caption ${captionCount}件`,
      "Phase4の次ステップでsd-scripts/kohya実行環境に接続します。"
    ].join(" ")
  };
}

function isLoraTrainingProject(value: unknown): value is LoraTrainingProject {
  if (!value || typeof value !== "object") return false;
  const project = value as Partial<LoraTrainingProject>;
  return (
    typeof project.id === "string" &&
    typeof project.name === "string" &&
    typeof project.triggerWord === "string" &&
    (typeof project.sourceLora === "string" || project.sourceLora === undefined) &&
    typeof project.baseCheckpoint === "string" &&
    typeof project.datasetPath === "string" &&
    typeof project.outputName === "string" &&
    typeof project.resolution === "number" &&
    typeof project.repeats === "number" &&
    typeof project.epochs === "number" &&
    typeof project.networkDim === "number" &&
    typeof project.learningRate === "string" &&
    typeof project.createdAt === "string" &&
    typeof project.updatedAt === "string" &&
    (project.status === "draft" ||
      project.status === "ready" ||
      project.status === "running" ||
      project.status === "complete" ||
      project.status === "blocked" ||
      project.status === "failed") &&
    typeof project.message === "string"
  );
}

function safeFileName(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || `lora-${Date.now()}`;
}

function isPromptHistoryEntry(value: unknown): value is PromptHistoryEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Partial<PromptHistoryEntry>;
  return (
    typeof entry.id === "string" &&
    (typeof entry.japanesePrompt === "string" || entry.japanesePrompt === undefined) &&
    typeof entry.prompt === "string" &&
    typeof entry.negativePrompt === "string" &&
    typeof entry.checkpoint === "string" &&
    typeof entry.width === "number" &&
    typeof entry.height === "number" &&
    typeof entry.steps === "number" &&
    typeof entry.cfg === "number" &&
    typeof entry.seed === "number" &&
    typeof entry.createdAt === "string"
  );
}

async function assertOllamaReady(ollamaUrl: string): Promise<void> {
  const status = await getOllamaStatus(ollamaUrl);
  if (!status.dataLinked) {
    throw new Error("Ollama保存先がDataPathに向いていません。先に環境タブのLLM / Ollamaパネルで「起動」を実行してください。");
  }
  if (!status.reachable) {
    throw new Error("Ollama APIに接続できません。先に環境タブのLLM / Ollamaパネルで「起動」を実行してください。");
  }
}

async function generateOllamaText(ollamaUrl: string, model: string, prompt: string): Promise<string> {
  const response = await fetch(`${ollamaUrl.replace(/\/$/, "")}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, stream: false, prompt })
  });
  if (!response.ok) {
    throw new Error(`Ollama生成に失敗しました: HTTP ${response.status}`);
  }
  const json = (await response.json()) as { response?: string };
  return String(json.response ?? "");
}

async function saveIllustrationTags(tags: IllustrationTag[]): Promise<void> {
  await mkdir(appDataRoot, { recursive: true });
  await writeFile(illustrationTagsPath, JSON.stringify(tags, null, 2), "utf8");
}

function normalizeIllustrationTag(value: unknown): IllustrationTag | null {
  if (!value || typeof value !== "object") return null;
  const tag = value as Partial<IllustrationTag>;
  const { id, name, englishName, sourcePrompt, japaneseDescription, englishDescription, createdAt, updatedAt } = tag;
  if (
    typeof id !== "string" ||
    typeof name !== "string" ||
    typeof sourcePrompt !== "string" ||
    typeof japaneseDescription !== "string" ||
    typeof englishDescription !== "string" ||
    typeof createdAt !== "string" ||
    typeof updatedAt !== "string"
  ) {
    return null;
  }
  return {
    id,
    name,
    englishName: typeof englishName === "string" && englishName.trim() ? englishName : name,
    sourcePrompt,
    japaneseDescription,
    englishDescription,
    createdAt,
    updatedAt
  };
}

async function saveLlmChatHistory(history: LlmChatMessage[]): Promise<void> {
  await mkdir(appDataRoot, { recursive: true });
  await writeFile(llmChatHistoryPath, JSON.stringify(history, null, 2), "utf8");
}

function isLlmChatMessage(value: unknown): value is LlmChatMessage {
  if (!value || typeof value !== "object") return false;
  const message = value as Partial<LlmChatMessage>;
  return (
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant") &&
    typeof message.content === "string" &&
    typeof message.model === "string" &&
    typeof message.createdAt === "string"
  );
}

async function huggingFaceAuthHeader(): Promise<Record<string, string>> {
  const settings = await getAppSettings();
  return settings.huggingFaceApiKey ? { Authorization: `Bearer ${settings.huggingFaceApiKey}` } : {};
}

async function withCivitaiToken(url: URL): Promise<URL> {
  if (!url.hostname.includes("civitai.com")) return url;
  const settings = await getAppSettings();
  if (settings.civitaiApiKey && !url.searchParams.has("token")) {
    url.searchParams.set("token", settings.civitaiApiKey);
  }
  return url;
}

function downloadErrorMessage(status: number, url: URL): string {
  if (status === 401 && url.hostname.includes("civitai.com")) {
    return [
      "CivitaiのダウンロードがHTTP 401で拒否されました。",
      "Civitai APIキーを保存してから再試行してください。",
      "保存済みでも失敗する場合、そのモデルはログイン、年齢確認、または利用権限が必要な可能性があります。"
    ].join("\n");
  }
  return `ダウンロードに失敗しました: HTTP ${status}`;
}

export async function runTestGeneration(request: TestGenerationRequest): Promise<TestGenerationResult> {
  const paths = getRuntimePaths();
  const modelInfo = await inspectModelFile(join(paths.checkpoints, request.checkpoint));
  if (!modelInfo.simpleWorkflowReady) {
    return {
      ok: false,
      message: [
        "このCheckpointは現在の簡易テスト生成では使えません。",
        `Checkpoint: ${request.checkpoint}`,
        `判定: ${modelInfo.compatibilityLabel}`,
        "理由: promptを読むCLIP/text encoder、または画像復元用VAEがCheckpoint内にありません。",
        "今後のPhaseでは外部CLIP/VAEを選択するworkflowに対応させます。今はCLIP/VAE入りのSDXL/Pony/Illustrious系Checkpointを選んでください。"
      ].join("\n")
    };
  }
  const workflow = testWorkflowTemplate();
  workflow["4"].inputs.ckpt_name = request.checkpoint;
  applyLorasToWorkflow(workflow, request.loras ?? []);
  workflow["6"].inputs.text = request.prompt;
  workflow["7"].inputs.text = request.negativePrompt;
  workflow["5"].inputs.width = request.width;
  workflow["5"].inputs.height = request.height;
  workflow["5"].inputs.batch_size = 1;
  workflow["3"].inputs.seed = normalizeSeed(request.seed);
  workflow["3"].inputs.steps = request.steps;
  workflow["3"].inputs.cfg = request.cfg;

  const promptResponse = await fetch(`${request.comfyUrl.replace(/\/$/, "")}/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: workflow })
  });
  if (!promptResponse.ok) {
    const errorText = await promptResponse.text();
    return { ok: false, message: `ComfyUIへの投入に失敗しました: HTTP ${promptResponse.status}\n${errorText}` };
  }

  const promptJson = (await promptResponse.json()) as { prompt_id?: string };
  const promptId = promptJson.prompt_id;
  if (!promptId) {
    return { ok: false, message: "ComfyUIからprompt_idが返りませんでした。" };
  }

  const before = new Set(await listFiles(paths.outputs));
  const deadline = Date.now() + 1000 * 300;
  while (Date.now() < deadline) {
    await delay(1500);
    const history = await getPromptHistory(request.comfyUrl, promptId);
    const errorMessage = extractPromptError(history);
    if (errorMessage) {
      return {
        ok: false,
        message: errorMessage,
        promptId
      };
    }
    const outputs = await listFiles(paths.outputs);
    const fresh = outputs.find((name) => !before.has(name) && /\.(png|jpe?g|webp)$/i.test(name));
    if (fresh) {
      const imagePath = join(paths.outputs, fresh);
      return {
        ok: true,
        message: "テスト生成が完了しました。",
        imagePath,
        imageDataUrl: await imageDataUrl(imagePath),
        promptId
      };
    }
  }

  return {
    ok: false,
    message: "生成完了を待機しましたが、新しい画像を確認できませんでした。",
    promptId
  };
}

function applyLorasToWorkflow(
  workflow: Record<string, { class_type: string; inputs: Record<string, unknown> }>,
  loras: GenerationLora[]
): void {
  const activeLoras = loras.filter((lora) => lora.name.trim() && (lora.modelStrength !== 0 || lora.clipStrength !== 0));
  let modelSource: [string, number] = ["4", 0];
  let clipSource: [string, number] = ["4", 1];
  activeLoras.forEach((lora, index) => {
    const nodeId = String(20 + index);
    workflow[nodeId] = {
      class_type: "LoraLoader",
      inputs: {
        lora_name: lora.name,
        strength_model: lora.modelStrength,
        strength_clip: lora.clipStrength,
        model: modelSource,
        clip: clipSource
      }
    };
    modelSource = [nodeId, 0];
    clipSource = [nodeId, 1];
  });
  workflow["3"].inputs.model = modelSource;
  workflow["6"].inputs.clip = clipSource;
  workflow["7"].inputs.clip = clipSource;
}

async function imageDataUrl(path: string): Promise<string> {
  const ext = path.toLowerCase().split(".").pop();
  const mime = ext === "jpg" || ext === "jpeg" ? "image/jpeg" : ext === "webp" ? "image/webp" : "image/png";
  const data = await readFile(path);
  return `data:${mime};base64,${data.toString("base64")}`;
}

async function getPromptHistory(comfyUrl: string, promptId: string): Promise<unknown> {
  try {
    const response = await fetch(`${comfyUrl.replace(/\/$/, "")}/history/${promptId}`);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

function extractPromptError(history: unknown): string | null {
  if (!history || typeof history !== "object") return null;
  const root = history as Record<string, unknown>;
  const entry = Object.values(root)[0] as Record<string, unknown> | undefined;
  const status = entry?.status as Record<string, unknown> | undefined;
  if (status?.status_str !== "error") return null;
  const messages = Array.isArray(status.messages) ? status.messages : [];
  const errorTuple = messages.find((message) => Array.isArray(message) && message[0] === "execution_error") as unknown[] | undefined;
  const error = errorTuple?.[1] as Record<string, unknown> | undefined;
  const nodeId = error?.node_id ? `node ${String(error.node_id)}` : "node unknown";
  const nodeType = error?.node_type ? String(error.node_type) : "unknown";
  const exception = error?.exception_message ? String(error.exception_message) : "ComfyUI実行エラー";
  if (String(error?.exception_type ?? "") === "BrokenPipeError" || exception.includes("Broken pipe")) {
    return [
      `ComfyUI実行エラー: ${nodeId} / ${nodeType}`,
      "ComfyUIのログ出力先パイプが切れています。",
      "環境タブでmacOS ComfyUIを停止してから、アプリを再起動し、もう一度開始してください。"
    ].join("\n");
  }
  return [`ComfyUI実行エラー: ${nodeId} / ${nodeType}`, exception].join("\n");
}

async function commandVersion(command: string, args: string[]): Promise<{ ok: boolean; text: string | null }> {
  try {
    const { stdout, stderr } = await execDockerFile(command, args, { timeout: 10000 });
    return { ok: true, text: String(stdout || stderr).trim() };
  } catch {
    return { ok: false, text: null };
  }
}

async function getDockerDaemonStatus(): Promise<{ ok: boolean; text: string | null }> {
  try {
    await execDockerFile("docker", ["info"], { timeout: 10000, maxBuffer: 1024 * 1024 });
    return { ok: true, text: "Docker daemon 接続OK" };
  } catch (error) {
    return { ok: false, text: normalizeCommandError(error) };
  }
}

async function waitForDockerDaemon(expectReady: boolean, timeoutMs: number): Promise<{ ok: boolean; text: string | null }> {
  const deadline = Date.now() + timeoutMs;
  let last: { ok: boolean; text: string | null } = { ok: false, text: null };
  while (Date.now() < deadline) {
    last = await getDockerDaemonStatus();
    if (last.ok === expectReady) {
      return { ok: true, text: last.text };
    }
    await delay(2000);
  }
  return { ok: false, text: last.text };
}

async function assertDockerDaemonReady(): Promise<void> {
  const daemon = await getDockerDaemonStatus();
  if (!daemon.ok) {
    throw new Error(
      [
        "Docker daemon に接続できません。",
        "Docker Desktop または Docker Engine を起動してから、もう一度 Build / Start を押してください。",
        daemon.text ? `詳細: ${daemon.text}` : null
      ]
        .filter(Boolean)
        .join("\n")
    );
  }
}

function normalizeCommandError(error: unknown): string {
  if (error && typeof error === "object") {
    const maybe = error as { stderr?: string; stdout?: string; message?: string };
    return (maybe.stderr || maybe.stdout || maybe.message || String(error)).trim();
  }
  return String(error);
}

async function getComposeCommand(): Promise<string[] | null> {
  const plugin = await commandVersion("docker", ["compose", "version"]);
  if (plugin.ok) {
    return ["docker", "compose"];
  }

  const standalone = await commandVersion("docker-compose", ["version"]);
  if (standalone.ok) {
    return ["docker-compose"];
  }

  return null;
}

async function composeArgs(composeFile: string, tail: string[]): Promise<{ command: string; args: string[] } | null> {
  const compose = await getComposeCommand();
  if (!compose) {
    return null;
  }

  const [command, ...baseArgs] = compose;
  return { command, args: [...baseArgs, "-f", composeFile, ...tail] };
}

function dockerRunArgs(paths: RuntimePaths): string[] {
  return [
    "run",
    "-d",
    "--name",
    "illustmk-comfyui",
    "--restart",
    "unless-stopped",
    "-p",
    `${comfyPort}:8188`,
    "-v",
    `${paths.checkpoints}:/opt/ComfyUI/models/checkpoints:ro`,
    "-v",
    `${paths.loras}:/opt/ComfyUI/models/loras:ro`,
    "-v",
    `${paths.vae}:/opt/ComfyUI/models/vae:ro`,
    "-v",
    `${paths.outputs}:/data/outputs:rw`,
    "--security-opt",
    "no-new-privileges:true",
    "--cap-drop",
    "ALL",
    "illustmk-comfyui:phase1"
  ];
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function findDockerDesktop(): Promise<string | null> {
  for (const path of dockerDesktopPaths) {
    if (await exists(path)) {
      return path;
    }
  }
  return null;
}

async function findOllamaApp(): Promise<string | null> {
  for (const path of ollamaAppPaths) {
    if (await exists(path)) return path;
  }
  return null;
}

async function findOllamaBinary(): Promise<string | null> {
  const candidates = [
    "/opt/homebrew/bin/ollama",
    "/usr/local/bin/ollama",
    "/usr/bin/ollama",
    "/bin/ollama",
    ...ollamaAppPaths.map((path) => join(path, "Contents", "Resources", "ollama"))
  ];
  for (const candidate of candidates) {
    if (await exists(candidate)) return candidate;
  }
  return null;
}

async function isOllamaDataLinked(): Promise<boolean> {
  try {
    const entry = await lstat(userOllamaHome);
    if (!entry.isSymbolicLink()) return false;
    const target = await readlink(userOllamaHome);
    return resolve(homedir(), target) === resolve(getRuntimePaths().ollama);
  } catch {
    return false;
  }
}

async function ensureOllamaDataLink(onLog: RuntimeLogSink = () => undefined): Promise<void> {
  const paths = getRuntimePaths();
  await mkdir(paths.ollama, { recursive: true });
  await mkdir(join(paths.ollama, "logs"), { recursive: true });

  if (await isOllamaDataLinked()) {
    onLog({ level: "info", line: `Ollamaデータ保存先は設定済みです: ${userOllamaHome} -> ${paths.ollama}` });
    return;
  }

  let homeEntry;
  try {
    homeEntry = await lstat(userOllamaHome);
  } catch {
    await symlink(paths.ollama, userOllamaHome, "dir");
    onLog({ level: "info", line: `Ollamaデータ保存先を作成しました: ${userOllamaHome} -> ${paths.ollama}` });
    return;
  }

  const backupRoot = join(paths.ollama, `previous-dot-ollama-${timestampForPath()}`);
  if (homeEntry.isSymbolicLink()) {
    await movePathCrossDevice(userOllamaHome, backupRoot);
    await symlink(paths.ollama, userOllamaHome, "dir");
    onLog({ level: "info", line: `既存の ~/.ollama リンクを退避しました: ${backupRoot}` });
    onLog({ level: "info", line: `Ollamaデータ保存先を切替えました: ${userOllamaHome} -> ${paths.ollama}` });
    return;
  }

  if (homeEntry.isDirectory()) {
    onLog({ level: "info", line: `既存の ~/.ollama を外部DataPathへ移動します: ${paths.ollama}` });
    await mergeDirectoryIntoTarget(userOllamaHome, paths.ollama, backupRoot, onLog);
    await rm(userOllamaHome, { recursive: true, force: true });
    await symlink(paths.ollama, userOllamaHome, "dir");
    onLog({ level: "info", line: `Ollamaデータ保存先を切替えました: ${userOllamaHome} -> ${paths.ollama}` });
    return;
  }

  await movePathCrossDevice(userOllamaHome, backupRoot);
  await symlink(paths.ollama, userOllamaHome, "dir");
  onLog({ level: "info", line: `既存の ~/.ollama を退避しました: ${backupRoot}` });
  onLog({ level: "info", line: `Ollamaデータ保存先を切替えました: ${userOllamaHome} -> ${paths.ollama}` });
}

async function mergeDirectoryIntoTarget(source: string, target: string, backupRoot: string, onLog: RuntimeLogSink): Promise<void> {
  await mkdir(target, { recursive: true });
  const entries = await readdir(source);
  for (const entry of entries) {
    const sourcePath = join(source, entry);
    const targetPath = join(target, entry);
    if (!(await exists(targetPath))) {
      await movePathCrossDevice(sourcePath, targetPath);
      continue;
    }

    const [sourceEntry, targetEntry] = await Promise.all([lstat(sourcePath), lstat(targetPath)]);
    if (sourceEntry.isDirectory() && targetEntry.isDirectory()) {
      await mergeDirectoryIntoTarget(sourcePath, targetPath, join(backupRoot, entry), onLog);
      await rm(sourcePath, { recursive: true, force: true });
      continue;
    }

    const backupPath = join(backupRoot, entry);
    await mkdir(backupRoot, { recursive: true });
    await movePathCrossDevice(sourcePath, backupPath);
    onLog({ level: "info", line: `同名ファイルを退避しました: ${backupPath}` });
  }
}

async function movePathCrossDevice(source: string, target: string): Promise<void> {
  await mkdir(resolve(target, ".."), { recursive: true });
  try {
    await rename(source, target);
  } catch (error) {
    if (!isCrossDeviceError(error)) throw error;
    await cp(source, target, { recursive: true, errorOnExist: true, force: false, preserveTimestamps: true });
    await rm(source, { recursive: true, force: true });
  }
}

function isCrossDeviceError(error: unknown): boolean {
  return Boolean(error && typeof error === "object" && "code" in error && (error as { code?: string }).code === "EXDEV");
}

function timestampForPath(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

async function listFiles(path: string): Promise<string[]> {
  try {
    const entries = await readdir(path);
    const files = await Promise.all(
      entries.map(async (entry) => {
        const entryPath = join(path, entry);
        const entryStat = await stat(entryPath);
        return entryStat.isFile() ? entry : null;
      })
    );
    return files.filter((file): file is string => Boolean(file)).sort();
  } catch {
    return [];
  }
}

async function listModelFiles(path: string): Promise<ModelFileInfo[]> {
  try {
    const entries = await readdir(path);
    const files = await Promise.all(
      entries.map(async (entry) => {
        const entryPath = join(path, entry);
        const entryStat = await stat(entryPath);
        if (!entryStat.isFile()) return null;
        const modelInfo = await inspectModelFile(entryPath);
        return {
          ...modelInfo,
          name: entry,
          sizeBytes: entryStat.size,
          sizeLabel: formatBytes(entryStat.size)
        };
      })
    );
    return files
      .filter((file): file is ModelFileInfo => Boolean(file))
      .sort((left, right) => left.name.localeCompare(right.name));
  } catch {
    return [];
  }
}

async function inspectModelFile(path: string): Promise<Omit<ModelFileInfo, "name" | "sizeBytes" | "sizeLabel">> {
  try {
    const fd = await openForHeader(path);
    try {
      const lengthBuffer = Buffer.alloc(8);
      await fd.read(lengthBuffer, 0, lengthBuffer.length, 0);
      const headerLength = Number(lengthBuffer.readBigUInt64LE(0));
      if (!Number.isFinite(headerLength) || headerLength <= 0 || headerLength > 128 * 1024 * 1024) {
        return modelCompatibility(false, false);
      }
      const headerBuffer = Buffer.alloc(headerLength);
      await fd.read(headerBuffer, 0, headerBuffer.length, 8);
      const header = JSON.parse(headerBuffer.toString("utf8")) as Record<string, unknown>;
      const keys = Object.keys(header).filter((key) => key !== "__metadata__");
      const hasClip = keys.some(isClipTensorKey);
      const hasVae = keys.some(isVaeTensorKey);
      return modelCompatibility(hasClip, hasVae);
    } finally {
      await fd.close();
    }
  } catch {
    return modelCompatibility(false, false);
  }
}

async function openForHeader(path: string) {
  const { open } = await import("node:fs/promises");
  return open(path, "r");
}

function modelCompatibility(hasClip: boolean, hasVae: boolean): Omit<ModelFileInfo, "name" | "sizeBytes" | "sizeLabel"> {
  const simpleWorkflowReady = hasClip && hasVae;
  return {
    hasClip,
    hasVae,
    simpleWorkflowReady,
    compatibilityLabel: simpleWorkflowReady
      ? "簡易生成OK"
      : `外部${[hasClip ? null : "CLIP", hasVae ? null : "VAE"].filter(Boolean).join("+")}が必要`
  };
}

function isClipTensorKey(key: string): boolean {
  return /(^|\.)cond_stage_model\.|(^|\.)conditioner\.|(^|\.)text_model\.|(^|\.)clip_l\.|(^|\.)clip_g\.|text_encoder/i.test(key);
}

function isVaeTensorKey(key: string): boolean {
  return /(^|\.)first_stage_model\.|(^|\.)vae\./i.test(key);
}

async function checkComfy(comfyUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${comfyUrl.replace(/\/$/, "")}/system_stats`, { signal: AbortSignal.timeout(3000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function getContainerState(): Promise<string> {
  try {
    const { stdout } = await execDockerFile("docker", ["inspect", "-f", "{{.State.Status}}", "illustmk-comfyui"], {
      timeout: 10000
    });
    return String(stdout).trim();
  } catch {
    return "not_created";
  }
}

function execDockerFile(command: string, args: string[], options: ExecFileOptions = {}) {
  return execFileAsync(command, args, {
    ...options,
    env: {
      ...process.env,
      ...options.env,
      PATH: `${dockerCommandPath}:${process.env.PATH ?? ""}`
    }
  });
}

function spawnDockerFile(command: string, args: string[], onLog: RuntimeLogSink): Promise<void> {
  return new Promise((resolve, reject) => {
    onLog({ level: "info", line: `$ ${command} ${args.join(" ")}` });
    const child = spawn(command, args, {
      env: {
        ...process.env,
        PATH: `${dockerCommandPath}:${process.env.PATH ?? ""}`
      }
    });

    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      for (const line of chunk.split(/\r?\n/)) {
        if (line.trim()) onLog({ level: "stdout", line });
      }
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      for (const line of chunk.split(/\r?\n/)) {
        if (line.trim()) onLog({ level: "stderr", line });
      }
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(stderr.trim() || `${command} exited with code ${code}`));
      }
    });
  });
}

function spawnNativeCommand(command: string, args: string[], onLog: RuntimeLogSink, options: { cwd?: string } = {}): Promise<void> {
  return new Promise((resolve, reject) => {
    onLog({ level: "info", line: `$ ${command} ${args.join(" ")}` });
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: {
        ...process.env,
        PYTORCH_ENABLE_MPS_FALLBACK: "1",
        PATH: `/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${process.env.PATH ?? ""}`
      }
    });
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => emitChunk("stdout", chunk, onLog));
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
      emitChunk("stderr", chunk, onLog);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(stderr.trim() || `${command} exited with code ${code}`));
    });
  });
}

function pipeProcessToLog(child: ChildProcessWithoutNullStreams, logPath: string, onLog: RuntimeLogSink): void {
  const logStream = createWriteStream(logPath, { flags: "a" });
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    logStream.write(chunk);
    emitChunk("stdout", chunk, onLog);
  });
  child.stderr.on("data", (chunk: string) => {
    logStream.write(chunk);
    emitChunk("stderr", chunk, onLog);
  });
  child.on("close", (code) => {
    logStream.write(`\n[illustMK] ComfyUI exited with code ${code}\n`);
    logStream.end();
  });
}

function emitChunk(level: RuntimeLogEvent["level"], chunk: string, onLog: RuntimeLogSink): void {
  for (const line of chunk.split(/\r?\n/)) {
    if (line.trim()) onLog({ level, line });
  }
}

async function writeNativeExtraModelPaths(paths: RuntimePaths): Promise<void> {
  if (!(await exists(paths.nativeComfy))) return;
  const yaml = `illustmk:
  base_path: ${paths.root}
  checkpoints: models/checkpoints
  loras: models/loras
  vae: models/vae
`;
  await writeFile(join(paths.nativeComfy, "extra_model_paths.yaml"), yaml, "utf8");
}

async function waitForComfy(comfyUrl: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await checkComfy(comfyUrl)) return true;
    await delay(2000);
  }
  return false;
}

async function waitForOllama(ollamaUrl: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if ((await getOllamaStatus(ollamaUrl)).reachable) return true;
    await delay(1000);
  }
  return false;
}

async function readNativePid(): Promise<number | null> {
  return readPidFile(getRuntimePaths().nativePid);
}

async function readPidFile(path: string): Promise<number | null> {
  try {
    const text = await readFile(path, "utf8");
    const pid = Number(text.trim());
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

async function isNativeRunning(): Promise<boolean> {
  const pid = await readNativePid();
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  await pipeline(createReadStream(path), hash);
  return hash.digest("hex");
}

async function downloadFileWithProgress(url: string, targetPath: string, onLog: RuntimeLogSink): Promise<void> {
  const response = await fetch(url, { headers: { "User-Agent": "illustMK/0.1" } });
  if (!response.ok || !response.body) {
    throw new Error(`ダウンロードに失敗しました: HTTP ${response.status}`);
  }
  const totalBytes = Number(response.headers.get("content-length") ?? 0);
  let receivedBytes = 0;
  let lastProgressLog = 0;
  const progressStream = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      receivedBytes += chunk.length;
      const now = Date.now();
      if (now - lastProgressLog >= 1000) {
        lastProgressLog = now;
        onLog({ level: "info", line: downloadProgressLine(receivedBytes, totalBytes) });
      }
      callback(null, chunk);
    }
  });
  await pipeline(response.body as unknown as NodeJS.ReadableStream, progressStream, createWriteStream(targetPath));
  onLog({ level: "info", line: downloadProgressLine(receivedBytes, totalBytes) });
}

function parseMountPoint(output: string): string | null {
  const matches = [...output.matchAll(/(\/Volumes\/[^\n]+)/g)].map((match) => match[1].trim());
  return matches.at(-1) ?? null;
}

async function copyOllamaApp(sourceApp: string, onLog: RuntimeLogSink): Promise<string> {
  const destinations = ["/Applications/Ollama.app", join(homedir(), "Applications", "Ollama.app")];
  let lastError: unknown = null;
  for (const destination of destinations) {
    try {
      await mkdir(destination.replace(/\/Ollama\.app$/, ""), { recursive: true });
      onLog({ level: "info", line: `コピー: ${sourceApp} -> ${destination}` });
      await rm(destination, { recursive: true, force: true });
      await execFileAsync("cp", ["-R", sourceApp, destination], { timeout: 120000 });
      return destination;
    } catch (error) {
      lastError = error;
      onLog({ level: "stderr", line: normalizeCommandError(error) });
    }
  }
  throw new Error(`Ollama.app のコピーに失敗しました: ${normalizeCommandError(lastError)}`);
}

function sanitizeFileName(name: string): string {
  return name.replace(/[^\w.\-+()[\] ]/g, "_").trim();
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 || unitIndex === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function downloadProgressLine(receivedBytes: number, totalBytes: number): string {
  if (totalBytes > 0) {
    const percent = Math.min(100, (receivedBytes / totalBytes) * 100);
    return `ダウンロード中: ${formatBytes(receivedBytes)} / ${formatBytes(totalBytes)} (${percent.toFixed(1)}%)`;
  }
  return `ダウンロード中: ${formatBytes(receivedBytes)}`;
}

function emitOllamaPullLine(line: string, onLog: RuntimeLogSink): void {
  try {
    const event = JSON.parse(line) as { status?: string; digest?: string; total?: number; completed?: number; error?: string };
    if (event.error) {
      onLog({ level: "stderr", line: event.error });
      return;
    }
    const status = event.status ?? "pulling";
    if (event.total && event.completed) {
      const percent = Math.min(100, (event.completed / event.total) * 100);
      onLog({
        level: "info",
        line: `${status}: ${formatBytes(event.completed)} / ${formatBytes(event.total)} (${percent.toFixed(1)}%)`
      });
      return;
    }
    onLog({ level: "info", line: event.digest ? `${status}: ${event.digest}` : status });
  } catch {
    onLog({ level: "stdout", line });
  }
}

function parsePromptEnhanceResponse(text: string): PromptEnhanceResult {
  try {
    const parsed = JSON.parse(text) as Partial<PromptEnhanceResult>;
    return {
      positivePrompt: String(parsed.positivePrompt ?? "").trim(),
      negativePrompt: String(parsed.negativePrompt ?? "").trim(),
      notes: String(parsed.notes ?? "").trim()
    };
  } catch {
    return {
      positivePrompt: text.trim(),
      negativePrompt: "low quality, worst quality, blurry, bad anatomy",
      notes: "LLM応答がJSONではなかったため、応答本文をpositive promptとして扱いました。"
    };
  }
}

function normalizeCivitaiModel(item: unknown): CivitaiModelCard[] {
  if (!item || typeof item !== "object") return [];
  const model = item as Record<string, unknown>;
  const versions = Array.isArray(model.modelVersions) ? model.modelVersions : [];
  const version = versions[0] as Record<string, unknown> | undefined;
  if (!version) return [];
  const files = Array.isArray(version.files) ? version.files : [];
  const file = files.find((candidate) => {
    const record = candidate as Record<string, unknown>;
    return String(record.name ?? "").toLowerCase().endsWith(".safetensors");
  }) as Record<string, unknown> | undefined;
  const downloadUrl = String(version.downloadUrl ?? "");
  if (!file || !downloadUrl) return [];

  const stats = (model.stats ?? {}) as Record<string, unknown>;
  const versionStats = (version.stats ?? {}) as Record<string, unknown>;
  const creator = (model.creator ?? {}) as Record<string, unknown>;
  const images = Array.isArray(version.images) ? version.images : [];
  const image = images[0] as Record<string, unknown> | undefined;
  const hashes = (file.hashes ?? {}) as Record<string, unknown>;
  const modelId = Number(model.id ?? 0);
  const versionId = Number(version.id ?? 0);
  return [
    {
      id: modelId,
      name: String(model.name ?? "Unknown"),
      type: String(model.type ?? "Checkpoint"),
      nsfw: Boolean(model.nsfw),
      creator: String(creator.username ?? "unknown"),
      versionId,
      versionName: String(version.name ?? ""),
      baseModel: String(version.baseModel ?? ""),
      fileName: String(file.name ?? `${modelId}-${versionId}.safetensors`),
      sizeKB: Number(file.sizeKB ?? 0),
      downloadUrl,
      pageUrl: `https://civitai.com/models/${modelId}?modelVersionId=${versionId}`,
      imageUrl: image?.url ? String(image.url) : null,
      downloadCount: Number(stats.downloadCount ?? versionStats.downloadCount ?? 0),
      rating: Number(stats.rating ?? 0),
      sha256: hashes.SHA256 ? String(hashes.SHA256) : undefined
    }
  ];
}

function normalizeHuggingFaceModelCard(item: Record<string, unknown>): HuggingFaceLlmModelCard {
  const id = String(item.id ?? item.modelId ?? "");
  const name = id.split("/").pop() ?? id;
  const author = id.includes("/") ? id.split("/")[0] : "";
  const siblings = Array.isArray(item.siblings) ? item.siblings : [];
  const ggufFiles = siblings
    .map((sibling) => {
      if (!sibling || typeof sibling !== "object") return null;
      const record = sibling as { rfilename?: unknown; size?: unknown; lfs?: { size?: unknown } };
      const name = typeof record.rfilename === "string" ? record.rfilename : "";
      if (!name.toLowerCase().endsWith(".gguf")) return null;
      const sizeBytes = Number(record.size ?? record.lfs?.size ?? NaN);
      return {
        name,
        sizeBytes: Number.isFinite(sizeBytes) && sizeBytes > 0 ? sizeBytes : null,
        sizeLabel: Number.isFinite(sizeBytes) && sizeBytes > 0 ? formatBytes(sizeBytes) : "サイズ不明"
      };
    })
    .filter((file): file is { name: string; sizeBytes: number | null; sizeLabel: string } => Boolean(file));
  const tags = Array.isArray(item.tags) ? item.tags.map(String) : [];
  const sizedFiles = ggufFiles.filter((file) => file.sizeBytes !== null);
  const totalSizeBytes = sizedFiles.length ? sizedFiles.reduce((sum, file) => sum + (file.sizeBytes ?? 0), 0) : null;
  return {
    id,
    name,
    author,
    pullName: `hf.co/${id}`,
    downloads: Number(item.downloads ?? 0),
    likes: Number(item.likes ?? 0),
    updatedAt: item.lastModified ? String(item.lastModified) : null,
    tags,
    ggufFiles,
    totalSizeBytes,
    totalSizeLabel: totalSizeBytes ? formatBytes(totalSizeBytes) : "サイズ不明"
  };
}

async function withHuggingFaceModelFileSizes(card: HuggingFaceLlmModelCard): Promise<HuggingFaceLlmModelCard> {
  if (card.totalSizeBytes !== null) return card;
  try {
    const response = await fetch(`https://huggingface.co/api/models/${card.id}?blobs=true`, {
      headers: {
        Accept: "application/json",
        ...(await huggingFaceAuthHeader())
      },
      signal: AbortSignal.timeout(20000)
    });
    if (!response.ok) return card;
    return normalizeHuggingFaceModelCard((await response.json()) as Record<string, unknown>);
  } catch {
    return card;
  }
}

const defaultBaseModelFilters = ["SDXL", "Pony", "Illustrious", "Animagine"];

function isAllowedBaseModel(model: CivitaiModelCard, filters: string[]): boolean {
  if (!filters.length) return true;
  const base = model.baseModel.toLowerCase();
  return filters.some((filter) => base.includes(filter.toLowerCase()));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeSeed(seed: number): number {
  if (Number.isFinite(seed) && seed >= 0) return Math.floor(seed);
  return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}

function dockerfileText(): string {
  return `FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

RUN apt-get update \\
  && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \\
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI
WORKDIR /opt/ComfyUI
RUN pip install --no-cache-dir --upgrade pip \\
  && pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 comfy
RUN chown -R comfy:comfy /opt/ComfyUI
USER comfy

EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--output-directory", "/data/outputs"]
`;
}

function composeText(paths: RuntimePaths): string {
  return `services:
  comfyui:
    build:
      context: .
    image: illustmk-comfyui:phase1
    container_name: illustmk-comfyui
    restart: unless-stopped
    ports:
      - "${comfyPort}:8188"
    volumes:
      - "${paths.checkpoints}:/opt/ComfyUI/models/checkpoints:ro"
      - "${paths.loras}:/opt/ComfyUI/models/loras:ro"
      - "${paths.vae}:/opt/ComfyUI/models/vae:ro"
      - "${paths.outputs}:/data/outputs:rw"
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
`;
}

function testWorkflowTemplate(): Record<string, { class_type: string; inputs: Record<string, unknown> }> {
  return {
    "3": {
      class_type: "KSampler",
      inputs: {
        seed: -1,
        steps: 20,
        cfg: 7,
        sampler_name: "euler",
        scheduler: "normal",
        denoise: 1,
        model: ["4", 0],
        positive: ["6", 0],
        negative: ["7", 0],
        latent_image: ["5", 0]
      }
    },
    "4": {
      class_type: "CheckpointLoaderSimple",
      inputs: {
        ckpt_name: ""
      }
    },
    "5": {
      class_type: "EmptyLatentImage",
      inputs: {
        width: 768,
        height: 1024,
        batch_size: 1
      }
    },
    "6": {
      class_type: "CLIPTextEncode",
      inputs: {
        text: "masterpiece, best quality, 1girl, anime style",
        clip: ["4", 1]
      }
    },
    "7": {
      class_type: "CLIPTextEncode",
      inputs: {
        text: "low quality, worst quality",
        clip: ["4", 1]
      }
    },
    "8": {
      class_type: "VAEDecode",
      inputs: {
        samples: ["3", 0],
        vae: ["4", 2]
      }
    },
    "9": {
      class_type: "SaveImage",
      inputs: {
        filename_prefix: "illustmk_phase1",
        images: ["8", 0]
      }
    }
  };
}
