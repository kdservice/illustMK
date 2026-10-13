import { useEffect, useMemo, useRef, useState, type DragEvent, type UIEvent } from "react";
import { createRoot } from "react-dom/client";
import {
  Boxes,
  Bot,
  CheckCircle2,
  CircleAlert,
  Download,
  ExternalLink,
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
  CivitaiRequiredComponent,
  DataPathEntry,
  DownloadRequest,
  GenerationLora,
  GenerationSeedMode,
  GenerationSceneBackground,
  GenerationSceneCharacter,
  GenerationSceneSettings,
  HuggingFaceLlmModelCard,
  IllustrationTag,
  LlmChatMessage,
  LoraCaptionCustomTag,
  LoraCaptionTag,
  LoraCaptionWorkspace,
  LoraTrainingRunParams,
  LoraTrainingProject,
  ModelAssetType,
  ModelLibraryEntry,
  OllamaStatus,
  PromptEnhanceMode,
  PromptGenerationSettings,
  PromptHistoryEntry,
  PoseConditioning,
  RuntimePaths,
  RuntimeStatus,
  SceneCharacterSettings,
  SavedPose,
  TestGenerationRequest,
  TestGenerationResult
} from "../../main/runtime";
import { createDefaultPoseConditioning, PoseEditor } from "./PoseEditor";
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

type RepeatGenerationItem = {
  id: string;
  seed: number;
  startedAt: string;
  durationText?: string;
  result?: TestGenerationResult;
  historyId?: string;
  memo?: string;
  error?: string;
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
const recommendedVaeModels = [
  {
    name: "SDXL VAE fp16 fix",
    fileName: "sdxl.vae.safetensors",
    url: "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors",
    source: "Hugging Face",
    note: "SDXL / Illustrious / Pony系で外部VAEが必要な時の基本候補。"
  }
];
const recommendedClipModels = [
  {
    name: "CLIP-L",
    fileName: "clip_l.safetensors",
    url: "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    source: "Hugging Face",
    note: "単体CLIPLoaderで試す候補。専用workflowが必要なモデルでは別CLIPが必要な場合あり。"
  }
];
type CheckpointExternalSelection = {
  vae: string;
  clip: string;
};
type CheckpointRequiredComponents = Record<string, CivitaiRequiredComponent[]>;
type CheckpointBaseModels = Record<string, string>;
type PromptKind = "positive" | "negative" | "background-positive" | "background-negative";
type TagPromptTarget = "japanese" | "positive" | "background-japanese" | "background-positive";
type OptionPromptTag = {
  id: string;
  japanese: string;
  english: string;
  color: string;
  category: string;
};
type AppliedLoraTrigger = {
  loraName: string;
  triggerId: string;
  title: string;
  triggerWord: string;
};
type UsageSample = {
  cpu: number | null;
  gpu: number | null;
};
type SavedSceneCharacter = GenerationSceneCharacter & { savedAt: string };
type SavedSceneBackground = GenerationSceneBackground & { savedAt: string };
type GenerationScope = "character" | "background" | "scene";

const defaultSeedRange = "0-10";
const backgroundOnlyPositivePrompt = "scenery, environment, empty scene, uninhabited";
const backgroundOnlyNegativePrompt = "person, people, human, character, 1girl, 1boy, boy, girl, man, woman";

const savedSceneCharactersKey = "illustmk.savedSceneCharacters";
const savedSceneBackgroundsKey = "illustmk.savedSceneBackgrounds";
const defaultCharacterPlacement = { x: 50, y: 50, scale: 100, zIndex: 1 };

function createDefaultSceneBackground(id: string = crypto.randomUUID()): GenerationSceneBackground {
  return {
    id,
    title: "背景",
    japanesePrompt: "",
    prompt: "",
    negativePrompt: "low quality, worst quality, blurry",
    promptEnhanceMode: "balanced-detail",
    promptNotes: "",
    japanesePromptTagIds: [],
    positivePromptTagIds: [],
    positiveOptionTagIds: [],
    negativeOptionTagIds: [],
    referenceImageDataUrl: "",
    referenceStrength: 0.7,
    imageSizePreset: "768x1024",
    width: 768,
    height: 1024,
    steps: 20,
    cfg: 7,
    seedMode: "random",
    seedValue: -1,
    seedRange: defaultSeedRange
  };
}

function normalizeSceneBackground(background: Partial<GenerationSceneBackground>): GenerationSceneBackground {
  return {
    ...createDefaultSceneBackground(background.id),
    ...background,
    seedRange: background.seedRange ?? defaultSeedRange
  };
}

function parseSeedRange(value: string): { start: number; end: number } {
  const match = value.trim().match(/^(\d+)\s*-\s*(\d+)$/);
  if (!match) throw new Error("Seed Valueは 0-10 の形式で入力してください。");
  const start = Number(match[1]);
  const end = Number(match[2]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) throw new Error("Seed Valueは安全な整数範囲で入力してください。");
  if (start > end) throw new Error("Seed Valueは開始値を終了値以下にしてください。");
  return { start, end };
}

function automaticCharacterX(index: number, total: number): number {
  return total > 0 ? Math.round(((index + 1) * 100) / (total + 1)) : 50;
}
const recommendedLlmModels = [
  { name: "qwen2.5vl:3b", source: "Ollama", size: "軽量", note: "画像対応VLM。デッサン人形のポーズ解析向け。" },
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
  const [dataPaths, setDataPaths] = useState<DataPathEntry[]>([]);
  const [selectedDataPathId, setSelectedDataPathId] = useState<string | null>(null);
  const [editingDataPath, setEditingDataPath] = useState<DataPathEntry | null>(null);
  const [dataPathMemo, setDataPathMemo] = useState("");
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [comfyUrl, setComfyUrl] = useState(defaultComfyUrl);
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadFileName, setDownloadFileName] = useState("");
  const [downloadSha, setDownloadSha] = useState("");
  const [vaeUrl, setVaeUrl] = useState("https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors");
  const [clipUrl, setClipUrl] = useState("");
  const [selectedVae, setSelectedVae] = useState("");
  const [selectedClip, setSelectedClip] = useState("");
  const [checkpointExternalSelections, setCheckpointExternalSelections] = useState<Record<string, CheckpointExternalSelection>>({});
  const [checkpointRequiredComponents, setCheckpointRequiredComponents] = useState<CheckpointRequiredComponents>({});
  const [checkpointBaseModels, setCheckpointBaseModels] = useState<CheckpointBaseModels>({});
  const [catalogQuery, setCatalogQuery] = useState("");
  const [catalogModels, setCatalogModels] = useState<CivitaiModelCard[]>([]);
  const [catalogPeriod, setCatalogPeriod] = useState<"AllTime" | "Year" | "Month" | "Week" | "Day">("Month");
  const [catalogSort, setCatalogSort] = useState<"Most Downloaded" | "Highest Rated" | "Newest">("Most Downloaded");
  const [baseFilters, setBaseFilters] = useState<string[]>(defaultBaseFilters);
  const [civitaiApiKey, setCivitaiApiKey] = useState("");
  const [huggingFaceApiKey, setHuggingFaceApiKey] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("http://127.0.0.1:11434");
  const [ollamaModel, setOllamaModel] = useState("qwen2.5:3b-instruct");
  const [llmPullName, setLlmPullName] = useState("qwen2.5:3b-instruct");
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);
  const [llmCatalogQuery, setLlmCatalogQuery] = useState("");
  const [hfLlmModels, setHfLlmModels] = useState<HuggingFaceLlmModelCard[]>([]);
  const [llmPrompt, setLlmPrompt] = useState("");
  const [llmChatHistory, setLlmChatHistory] = useState<LlmChatMessage[]>([]);
  const [illustrationTags, setIllustrationTags] = useState<IllustrationTag[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [japanesePromptTagIds, setJapanesePromptTagIds] = useState<string[]>([]);
  const [positivePromptTagIds, setPositivePromptTagIds] = useState<string[]>([]);
  const [positiveLoraTriggerIds, setPositiveLoraTriggerIds] = useState<string[]>([]);
  const [loraTriggerDialogOpen, setLoraTriggerDialogOpen] = useState(false);
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [tagEditingId, setTagEditingId] = useState<string | null>(null);
  const [tagName, setTagName] = useState("");
  const [tagEnglishName, setTagEnglishName] = useState("");
  const [tagColor, setTagColor] = useState("#3388ff");
  const [tagCategory, setTagCategory] = useState("");
  const [tagPrompt, setTagPrompt] = useState("");
  const [tagJapaneseDescription, setTagJapaneseDescription] = useState("");
  const [tagEnglishDescription, setTagEnglishDescription] = useState("");
  const [tagPromptTarget, setTagPromptTarget] = useState<TagPromptTarget | null>(null);
  const [optionPromptTags, setOptionPromptTags] = useState<OptionPromptTag[]>([]);
  const [positiveOptionTagIds, setPositiveOptionTagIds] = useState<string[]>([]);
  const [negativeOptionTagIds, setNegativeOptionTagIds] = useState<string[]>([]);
  const [optionTagDialogTarget, setOptionTagDialogTarget] = useState<PromptKind | null>(null);
  const [optionTagEditorOpen, setOptionTagEditorOpen] = useState(false);
  const [optionTagEditingId, setOptionTagEditingId] = useState<string | null>(null);
  const [optionTagJapanese, setOptionTagJapanese] = useState("");
  const [optionTagEnglish, setOptionTagEnglish] = useState("");
  const [optionTagColor, setOptionTagColor] = useState("#3388ff");
  const [optionTagCategory, setOptionTagCategory] = useState("品質");
  const [japanesePrompt, setJapanesePrompt] = useState("");
  const [promptEnhanceMode, setPromptEnhanceMode] = useState<PromptEnhanceMode>("balanced-detail");
  const [promptNotes, setPromptNotes] = useState("");
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [selectedLoras, setSelectedLoras] = useState<GenerationLora[]>([]);
  const [prompt, setPrompt] = useState("masterpiece, best quality, 1girl, anime style, soft light");
  const [negativePrompt, setNegativePrompt] = useState("low quality, worst quality, blurry, bad anatomy");
  const [lastImage, setLastImage] = useState<string | null>(null);
  const [lastHistoryId, setLastHistoryId] = useState<string | null>(null);
  const [imageMemo, setImageMemo] = useState("");
  const [generationResult, setGenerationResult] = useState<TestGenerationResult | null>(null);
  const [loraCompareResult, setLoraCompareResult] = useState<{ seed: number; withoutLora: TestGenerationResult; withLora: TestGenerationResult } | null>(null);
  const [repeatDialogOpen, setRepeatDialogOpen] = useState(false);
  const [repeatItems, setRepeatItems] = useState<RepeatGenerationItem[]>([]);
  const [repeatRunning, setRepeatRunning] = useState(false);
  const [repeatPaused, setRepeatPaused] = useState(false);
  const [repeatPreviewImage, setRepeatPreviewImage] = useState<string | null>(null);
  const [repeatMemoEditingItemId, setRepeatMemoEditingItemId] = useState<string | null>(null);
  const [repeatMemoText, setRepeatMemoText] = useState("");
  const [imageSizePreset, setImageSizePreset] = useState("768x1024");
  const [generationWidth, setGenerationWidth] = useState(768);
  const [generationHeight, setGenerationHeight] = useState(1024);
  const [generationSteps, setGenerationSteps] = useState(20);
  const [generationCfg, setGenerationCfg] = useState(7);
  const [seedMode, setSeedMode] = useState<GenerationSeedMode>("random");
  const [seedValue, setSeedValue] = useState(-1);
  const [seedRange, setSeedRange] = useState(defaultSeedRange);
  const [poseConditioning, setPoseConditioning] = useState<PoseConditioning>(() => createDefaultPoseConditioning());
  const [sceneCharacters, setSceneCharacters] = useState<GenerationSceneCharacter[]>([]);
  const [sceneBackground, setSceneBackground] = useState<GenerationSceneBackground | null>(null);
  const [activeSceneItemId, setActiveSceneItemId] = useState<string | null>(null);
  const [sceneAddOpen, setSceneAddOpen] = useState(false);
  const [savedSceneCharacters, setSavedSceneCharacters] = useState<SavedSceneCharacter[]>([]);
  const [savedSceneBackgrounds, setSavedSceneBackgrounds] = useState<SavedSceneBackground[]>([]);
  const [characterLibraryOpen, setCharacterLibraryOpen] = useState(false);
  const [backgroundLibraryOpen, setBackgroundLibraryOpen] = useState(false);
  const [savedPoses, setSavedPoses] = useState<SavedPose[]>([]);
  const [selectedSavedPoseId, setSelectedSavedPoseId] = useState("");
  const [poseTitle, setPoseTitle] = useState("");
  const [promptHistory, setPromptHistory] = useState<PromptHistoryEntry[]>([]);
  const [promptHistoryOpen, setPromptHistoryOpen] = useState(false);
  const [restoredGenerationSettings, setRestoredGenerationSettings] = useState<PromptGenerationSettings | null>(null);
  const [imagePreviewOpen, setImagePreviewOpen] = useState(false);
  const [loraProjects, setLoraProjects] = useState<LoraTrainingProject[]>([]);
  const [selectedLoraProjectId, setSelectedLoraProjectId] = useState<string | null>(null);
  const [loraCreateDialogOpen, setLoraCreateDialogOpen] = useState(false);
  const [loraProjectDialogMode, setLoraProjectDialogMode] = useState<"create" | "edit">("create");
  const [editingLoraProjectId, setEditingLoraProjectId] = useState<string | null>(null);
  const [loraName, setLoraName] = useState("");
  const [loraTriggerWord, setLoraTriggerWord] = useState("");
  const [loraTriggerWords, setLoraTriggerWords] = useState<string[]>([""]);
  const [loraDetailName, setLoraDetailName] = useState("");
  const [loraDetailTriggerWord, setLoraDetailTriggerWord] = useState("");
  const [loraDetailTriggerWords, setLoraDetailTriggerWords] = useState<string[]>([""]);
  const [loraProjectBaseFilters, setLoraProjectBaseFilters] = useState<string[]>([]);
  const [loraAdditionalTraining, setLoraAdditionalTraining] = useState(false);
  const [loraCatalogOpen, setLoraCatalogOpen] = useState(false);
  const [loraCatalogQuery, setLoraCatalogQuery] = useState("");
  const [loraCatalogModels, setLoraCatalogModels] = useState<CivitaiModelCard[]>([]);
  const [loraCatalogPeriod, setLoraCatalogPeriod] = useState<"Month" | "AllTime">("Month");
  const [loraCatalogPage, setLoraCatalogPage] = useState(1);
  const [loraCatalogCursor, setLoraCatalogCursor] = useState<string | null>(null);
  const [loraCatalogHasMore, setLoraCatalogHasMore] = useState(false);
  const [loraCatalogLoadingMore, setLoraCatalogLoadingMore] = useState(false);
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
  const [usageHistory, setUsageHistory] = useState<UsageSample[]>([]);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [autoFollowLogs, setAutoFollowLogs] = useState(true);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [mainTab, setMainTab] = useState<"environment" | "models" | "llm" | "lora" | "generate">("environment");
  const [modelLibrary, setModelLibrary] = useState<ModelLibraryEntry[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelTypeFilters, setModelTypeFilters] = useState<ModelAssetType[]>([]);
  const [modelCatalogOpen, setModelCatalogOpen] = useState(false);
  const [modelCatalogMode, setModelCatalogMode] = useState<"civitai" | "url">("civitai");
  const [modelCatalogTypes, setModelCatalogTypes] = useState<ModelAssetType[]>(["checkpoint"]);
  const [modelCatalogResults, setModelCatalogResults] = useState<CivitaiModelCard[]>([]);
  const [modelCatalogQuery, setModelCatalogQuery] = useState("");
  const [modelCatalogPeriod, setModelCatalogPeriod] = useState<"AllTime" | "Year" | "Month" | "Week" | "Day">("Month");
  const [modelCatalogSort, setModelCatalogSort] = useState<"Most Downloaded" | "Highest Rated" | "Newest">("Most Downloaded");
  const [modelDirectType, setModelDirectType] = useState<ModelAssetType>("checkpoint");
  const [modelDirectUrl, setModelDirectUrl] = useState("");
  const [modelDirectFileName, setModelDirectFileName] = useState("");
  const [modelDirectBase, setModelDirectBase] = useState("");
  const [modelDirectUsage, setModelDirectUsage] = useState("");
  const [modelImportOpen, setModelImportOpen] = useState(false);
  const [modelImportType, setModelImportType] = useState<ModelAssetType>("checkpoint");
  const [runtimeTab, setRuntimeTab] = useState<"macos" | "docker">("macos");
  const [modelAddMode, setModelAddMode] = useState<"catalog" | "url">("catalog");
  const [generateTab, setGenerateTab] = useState<"character" | "prompt" | "lora" | "options" | "pose">("prompt");
  const logViewRef = useRef<HTMLDivElement | null>(null);
  const currentOperationId = useRef<string | null>(null);
  const repeatStopRequested = useRef(false);
  const repeatPausedRef = useRef(false);

  const selectedCheckpointInfo = useMemo(
    () => status?.checkpoints.find((checkpoint) => checkpoint.name === selectedCheckpoint) ?? null,
    [selectedCheckpoint, status?.checkpoints]
  );
  const activeSceneCharacter = useMemo(
    () => sceneCharacters.find((character) => character.id === activeSceneItemId) ?? null,
    [activeSceneItemId, sceneCharacters]
  );
  const selectedTags = useMemo(
    () => illustrationTags.filter((tag) => new Set([...selectedTagIds, ...japanesePromptTagIds, ...positivePromptTagIds]).has(tag.id)),
    [illustrationTags, japanesePromptTagIds, positivePromptTagIds, selectedTagIds]
  );
  const selectedLoraProject = useMemo(
    () => loraProjects.find((project) => project.id === selectedLoraProjectId) ?? loraProjects[0] ?? null,
    [loraProjects, selectedLoraProjectId]
  );
  const selectedDataPath = useMemo(
    () => dataPaths.find((entry) => entry.id === selectedDataPathId) ?? dataPaths.find((entry) => entry.active) ?? dataPaths[0] ?? null,
    [dataPaths, selectedDataPathId]
  );
  const loraProjectBaseOptions = useMemo(
    () => uniqueCategories(loraProjects.map((project) => inferCheckpointBaseModel(project.baseCheckpoint))),
    [loraProjects]
  );
  const filteredLoraProjects = useMemo(
    () =>
      loraProjectBaseFilters.length
        ? loraProjects.filter((project) => loraProjectBaseFilters.includes(inferCheckpointBaseModel(project.baseCheckpoint)))
        : loraProjects,
    [loraProjectBaseFilters, loraProjects]
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
  const selectedModel = useMemo(
    () => modelLibrary.find((entry) => entry.id === selectedModelId) ?? null,
    [modelLibrary, selectedModelId]
  );
  const filteredModelLibrary = useMemo(
    () => modelTypeFilters.length ? modelLibrary.filter((entry) => modelTypeFilters.includes(entry.type)) : modelLibrary,
    [modelLibrary, modelTypeFilters]
  );
  const japanesePromptTags = useMemo(
    () => illustrationTags.filter((tag) => japanesePromptTagIds.includes(tag.id)),
    [illustrationTags, japanesePromptTagIds]
  );
  const positivePromptTags = useMemo(
    () => illustrationTags.filter((tag) => positivePromptTagIds.includes(tag.id)),
    [illustrationTags, positivePromptTagIds]
  );
  const positiveOptionTags = useMemo(
    () => optionPromptTags.filter((tag) => positiveOptionTagIds.includes(tag.id)),
    [optionPromptTags, positiveOptionTagIds]
  );
  const selectedLoraTriggers = useMemo<AppliedLoraTrigger[]>(
    () =>
      selectedLoras
        .flatMap((lora) => {
          const project = loraProjects.find((item) => item.outputName === lora.name || item.sourceLora === lora.name);
          const triggerWords = getLoraTriggerWords(project);
          return triggerWords.map((triggerWord, index) => ({
            loraName: lora.name,
            triggerId: `${lora.name}::${index}::${triggerWord}`,
            title: project?.name || lora.name,
            triggerWord
          }));
        }),
    [loraProjects, selectedLoras]
  );
  const positiveLoraTriggers = useMemo(
    () => selectedLoraTriggers.filter((item) => positiveLoraTriggerIds.includes(item.triggerId)),
    [positiveLoraTriggerIds, selectedLoraTriggers]
  );
  const negativeOptionTags = useMemo(
    () => optionPromptTags.filter((tag) => negativeOptionTagIds.includes(tag.id)),
    [negativeOptionTagIds, optionPromptTags]
  );
  const backgroundJapanesePromptTags = useMemo(
    () => illustrationTags.filter((tag) => sceneBackground?.japanesePromptTagIds.includes(tag.id)),
    [illustrationTags, sceneBackground?.japanesePromptTagIds]
  );
  const backgroundPositivePromptTags = useMemo(
    () => illustrationTags.filter((tag) => sceneBackground?.positivePromptTagIds.includes(tag.id)),
    [illustrationTags, sceneBackground?.positivePromptTagIds]
  );
  const backgroundPositiveOptionTags = useMemo(
    () => optionPromptTags.filter((tag) => sceneBackground?.positiveOptionTagIds.includes(tag.id)),
    [optionPromptTags, sceneBackground?.positiveOptionTagIds]
  );
  const backgroundNegativeOptionTags = useMemo(
    () => optionPromptTags.filter((tag) => sceneBackground?.negativeOptionTagIds.includes(tag.id)),
    [optionPromptTags, sceneBackground?.negativeOptionTagIds]
  );
  const availableLoraNames = useMemo(
    () => {
      const existingFiles = new Set((status?.loras ?? []).filter((lora) => !isStepLoraName(lora.name)).map((lora) => lora.name));
      return new Set(loraProjects.filter((project) => existingFiles.has(project.outputName)).map((project) => project.outputName));
    },
    [loraProjects, status?.loras]
  );
  const activeGenerationLoras = useMemo(
    () => selectedLoras.filter((lora) => availableLoraNames.has(lora.name)),
    [availableLoraNames, selectedLoras]
  );
  const effectivePrompt = composeGenerationPrompt(prompt, [
    ...positivePromptTags.map((tag) => tag.englishDescription),
    ...positiveOptionTags.map((tag) => tag.english),
    ...positiveLoraTriggers.map((item) => item.triggerWord)
  ]);
  const effectiveNegativePrompt = composeGenerationPrompt(negativePrompt, negativeOptionTags.map((tag) => tag.english));
  const selectedCheckpointGenerationReady = Boolean(
    selectedCheckpointInfo && (selectedCheckpointInfo.hasClip || selectedClip) && (selectedCheckpointInfo.hasVae || selectedVae)
  );
  const selectedCheckpointCompatibilityLabel = selectedCheckpointInfo
    ? [
        selectedCheckpointInfo.hasClip ? "CLIP内蔵" : selectedClip ? `外部CLIP: ${selectedClip}` : "外部CLIPが必要",
        selectedCheckpointInfo.hasVae ? "VAE内蔵" : selectedVae ? `外部VAE: ${selectedVae}` : "外部VAEが必要"
      ].join(" / ")
    : "未選択";
  const poseConditioningReady = !poseConditioning.enabled || Boolean(
    poseConditioning.imageDataUrl &&
    poseConditioning.controlNet &&
    status?.controlnet.some((file) => file.name === poseConditioning.controlNet) &&
    (!poseConditioning.depthControlNet || (
      poseConditioning.depthImageDataUrl &&
      status?.controlnet.some((file) => file.name === poseConditioning.depthControlNet)
    ))
  );
  const canGenerate = Boolean(status?.comfyReachable && selectedCheckpointGenerationReady && poseConditioningReady && !action.busy);
  const canGenerateCharacter = Boolean(canGenerate && activeSceneCharacter);
  const canGenerateBackground = Boolean(canGenerate && sceneBackground);
  const canGenerateScene = Boolean(canGenerate && (sceneCharacters.length || sceneBackground));
  const canDownloadLlm = Boolean(ollamaStatus?.reachable && ollamaStatus.dataLinked && llmPullName.trim() && !action.busy);
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
  const canEnhanceBackgroundPrompt = Boolean(
    ollamaStatus?.reachable && ollamaStatus.dataLinked && ollamaModel.trim() && sceneBackground?.japanesePrompt.trim() && !action.busy
  );
  const enhanceBackgroundPromptDisabledReason = useMemo(() => {
    if (action.busy) return "処理中です。";
    if (!ollamaStatus?.reachable) return "Ollama APIが未接続です。";
    if (!ollamaStatus.dataLinked) return "Ollama保存先がDataPathに切替わっていません。";
    if (!ollamaModel.trim()) return "LLMモデルを選択してください。";
    if (!sceneBackground?.japanesePrompt.trim()) return "日本語プロンプトを入力してください。";
    return "変換できます。";
  }, [action.busy, ollamaModel, ollamaStatus, sceneBackground?.japanesePrompt]);
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

  useEffect(() => {
    void refreshStatus();
    void loadSettings();
    void loadLlmChatHistory();
    void loadIllustrationTags();
    void loadPromptHistory();
    void loadLoraProjects();
    void loadModelLibrary();
    void loadSavedPoses();
    setSavedSceneCharacters(readLocalArray<SavedSceneCharacter>(savedSceneCharactersKey));
    setSavedSceneBackgrounds(readLocalArray<SavedSceneBackground>(savedSceneBackgroundsKey).map((item) => ({
      ...normalizeSceneBackground(item),
      savedAt: item.savedAt
    })));
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void refreshStatus();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [comfyUrl]);

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
    if (!settingsLoaded) return;
    const timer = window.setTimeout(() => {
      void saveSettings();
    }, 600);
    return () => window.clearTimeout(timer);
  }, [ollamaUrl, settingsLoaded]);

  useEffect(() => {
    if (!status) return;
    if (!status.checkpoints.length) {
      if (selectedCheckpoint) setSelectedCheckpoint("");
      return;
    }
    if (!selectedCheckpoint || !status.checkpoints.some((checkpoint) => checkpoint.name === selectedCheckpoint)) {
      setSelectedCheckpoint(status.checkpoints.find((checkpoint) => checkpoint.simpleWorkflowReady)?.name ?? status.checkpoints[0].name);
    }
  }, [selectedCheckpoint, status?.checkpoints]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("illustmk.checkpointExternalSelections");
      if (raw) setCheckpointExternalSelections(JSON.parse(raw) as Record<string, CheckpointExternalSelection>);
    } catch {
      setCheckpointExternalSelections({});
    }
    try {
      const rawRequired = localStorage.getItem("illustmk.checkpointRequiredComponents");
      if (rawRequired) setCheckpointRequiredComponents(JSON.parse(rawRequired) as CheckpointRequiredComponents);
    } catch {
      setCheckpointRequiredComponents({});
    }
    try {
      const rawBaseModels = localStorage.getItem("illustmk.checkpointBaseModels");
      if (rawBaseModels) setCheckpointBaseModels(JSON.parse(rawBaseModels) as CheckpointBaseModels);
    } catch {
      setCheckpointBaseModels({});
    }
    try {
      const rawOptionTags = localStorage.getItem("illustmk.optionPromptTags");
      if (rawOptionTags) setOptionPromptTags(normalizeOptionPromptTags(JSON.parse(rawOptionTags)));
    } catch {
      setOptionPromptTags([]);
    }
  }, []);

  useEffect(() => {
    if (!selectedCheckpoint) return;
    if (modelLibrary.some((entry) => entry.type === "checkpoint" && entry.active)) return;
    const saved = checkpointExternalSelections[selectedCheckpoint];
    if (!saved) return;
    setSelectedVae(saved.vae);
    setSelectedClip(saved.clip);
  }, [checkpointExternalSelections, modelLibrary, selectedCheckpoint]);

  useEffect(() => {
    if (!status) return;
    if (!status.vae.length) {
      if (selectedVae) setSelectedVae("");
      return;
    }
    if (selectedVae && !status.vae.some((file) => file.name === selectedVae)) {
      setSelectedVae("");
    }
  }, [selectedVae, status?.vae]);

  useEffect(() => {
    if (!status) return;
    if (!status.clip.length) {
      if (selectedClip) setSelectedClip("");
      return;
    }
    if (selectedClip && !status.clip.some((file) => file.name === selectedClip)) {
      setSelectedClip("");
    }
  }, [selectedClip, status?.clip]);

  useEffect(() => {
    if (!status) return;
    setSelectedLoras((current) => {
      const next = current.filter((lora) => availableLoraNames.has(lora.name));
      return sameGenerationLoras(current, next) ? current : next;
    });
  }, [availableLoraNames, status]);

  useEffect(() => {
    if (!positiveLoraTriggerIds.length) return;
    const availableTriggerIds = new Set(selectedLoraTriggers.map((trigger) => trigger.triggerId));
    setPositiveLoraTriggerIds((current) => {
      const next = current.filter((id) => availableTriggerIds.has(id));
      return sameStringArray(current, next) ? current : next;
    });
  }, [positiveLoraTriggerIds.length, selectedLoraTriggers]);

  useEffect(() => {
    if (!loraProjects.length) {
      setSelectedLoraProjectId(null);
      return;
    }
    if (!selectedLoraProjectId || !loraProjects.some((project) => project.id === selectedLoraProjectId)) {
      setSelectedLoraProjectId(loraProjects[0].id);
    }
  }, [loraProjects, selectedLoraProjectId]);

  useEffect(() => {
    if (!dataPaths.length) {
      setSelectedDataPathId(null);
      return;
    }
    if (!selectedDataPathId || !dataPaths.some((entry) => entry.id === selectedDataPathId)) {
      setSelectedDataPathId(dataPaths.find((entry) => entry.active)?.id ?? dataPaths[0].id);
    }
  }, [dataPaths, selectedDataPathId]);

  useEffect(() => {
    setLoraDetailName(selectedLoraProject?.name ?? "");
    setLoraDetailTriggerWord(selectedLoraProject?.triggerWord ?? "");
    setLoraDetailTriggerWords(getLoraTriggerWords(selectedLoraProject).length ? getLoraTriggerWords(selectedLoraProject) : [""]);
  }, [selectedLoraProject?.id, selectedLoraProject?.name, selectedLoraProject?.triggerWord, selectedLoraProject?.triggerWords]);

  useEffect(() => {
    repeatPausedRef.current = repeatPaused;
  }, [repeatPaused]);

  async function runAction<T>(message: string, task: () => Promise<T>, after?: (value: T) => void | Promise<void>): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    appendLog("info", message, id);
    setAction({ busy: true, message, error: null });
    setDownloadProgress(null);
    try {
      const value = await task();
      await after?.(value);
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

  async function runOllamaPull(modelName = llmPullName): Promise<void> {
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

  async function loadCivitaiModels(query?: string): Promise<void> {
    const trimmedQuery = query?.trim();
    await runAction(
      trimmedQuery ? "Civitaiを検索中" : "Civitai人気Top10を取得中",
      () =>
        window.illustMK.runtime.searchCivitai({
          query: trimmedQuery || undefined,
          sort: catalogSort,
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
      if (model.baseModel) {
        setCheckpointBaseModels((current) => {
          const next = { ...current, [model.fileName]: model.baseModel };
          localStorage.setItem("illustmk.checkpointBaseModels", JSON.stringify(next));
          return next;
        });
      }
      if (model.requiredComponents.length) {
        setCheckpointRequiredComponents((current) => {
          const next = { ...current, [model.fileName]: model.requiredComponents };
          localStorage.setItem("illustmk.checkpointRequiredComponents", JSON.stringify(next));
          return next;
        });
      }
    });
  }

  async function searchLoraCatalog(page = 1): Promise<void> {
    const append = page > 1;
    if (append) setLoraCatalogLoadingMore(true);
    await runAction(
      append ? "Civitai LoRAを追加取得中" : "Civitai LoRAを検索中",
      () =>
        window.illustMK.runtime.searchCivitaiPage({
          query: loraCatalogQuery.trim() || undefined,
          sort: "Most Downloaded",
          period: loraCatalogPeriod,
          limit: 20,
          page,
          cursor: append ? loraCatalogCursor ?? undefined : undefined,
          nsfw: false,
          type: "LORA"
        }),
      (result) => {
        const models = result.models;
        setLoraCatalogPage(page);
        setLoraCatalogCursor(result.nextCursor);
        setLoraCatalogHasMore(Boolean(result.nextCursor));
        setLoraCatalogModels((current) => {
          if (!append) return models;
          const seen = new Set(current.map((model) => `${model.id}-${model.versionId}`));
          return [...current, ...models.filter((model) => !seen.has(`${model.id}-${model.versionId}`))];
        });
        appendLog("info", `${models.length}件のLoRAを取得しました。`);
      }
    );
    setLoraCatalogLoadingMore(false);
  }

  function handleLoraCatalogScroll(event: UIEvent<HTMLDivElement>): void {
    const target = event.currentTarget;
    const nearBottom = target.scrollHeight - target.scrollTop - target.clientHeight < 320;
    if (!nearBottom || action.busy || loraCatalogLoadingMore || !loraCatalogHasMore) return;
    void searchLoraCatalog(loraCatalogPage + 1);
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
    await installModelAsset("vae", vaeUrl, "external.vae.safetensors");
  }

  async function installModelAsset(targetType: "vae" | "clip", url: string, fallbackFileName: string, explicitFileName?: string): Promise<void> {
    const fileName = explicitFileName || url.split("/").pop()?.split("?")[0] || fallbackFileName;
    const label = targetType === "vae" ? "VAE" : "CLIP/text encoder";
    await runDownloadAction(`${label}をインストール中`, {
      url: url.trim(),
      targetType,
      fileName,
      source: url.includes("civitai.com") ? "civitai" : "direct"
    }, () => {
      if (targetType === "vae") setSelectedVae(fileName);
      if (targetType === "clip") setSelectedClip(fileName);
    });
  }

  function componentTargetType(component: CivitaiRequiredComponent): "vae" | "clip" | null {
    const type = component.type.toLowerCase();
    if (type === "vae") return "vae";
    if (type === "text encoder") return "clip";
    return null;
  }

  function isComponentInstalled(component: CivitaiRequiredComponent): boolean {
    const targetType = componentTargetType(component);
    if (targetType === "vae") return Boolean(status?.vae.some((file) => file.name === component.name));
    if (targetType === "clip") return Boolean(status?.clip.some((file) => file.name === component.name));
    return false;
  }

  async function installRequiredComponent(checkpoint: string, component: CivitaiRequiredComponent): Promise<void> {
    const targetType = componentTargetType(component);
    if (!targetType) return;
    await installModelAsset(targetType, component.downloadUrl, component.name, component.name);
    const current = checkpointExternalSelections[checkpoint] ?? { vae: "", clip: "" };
    saveCheckpointExternalSelection(checkpoint, {
      ...current,
      [targetType]: component.name
    });
  }

  async function deleteSelectedVae(): Promise<void> {
    if (!selectedVae) return;
    await deleteModelAsset("vae", selectedVae);
  }

  async function importModelAsset(targetType: "vae" | "clip" | "controlnet"): Promise<void> {
    const label = targetType === "vae" ? "VAE" : targetType === "clip" ? "CLIP/text encoder" : "ControlNet";
    await runAction(`${label}をインポート中`, () => window.illustMK.runtime.importModelAsset({ targetType }), (result) => {
      if (!result) return;
      const fileName = result.path.split(/[\\/]/).pop() ?? "";
      if (targetType === "vae") setSelectedVae(fileName);
      if (targetType === "clip") setSelectedClip(fileName);
      if (targetType === "controlnet") {
        setPoseConditioning((current) => fileName.toLowerCase().includes("depth")
          ? { ...current, depthControlNet: fileName }
          : { ...current, controlNet: fileName });
      }
      appendLog("info", result.message);
      void refreshStatus();
    });
  }

  async function deleteModelAsset(targetType: "vae" | "clip", fileName: string): Promise<void> {
    if (!fileName) return;
    if (!window.confirm(`${fileName} を削除します。元に戻せません。`)) return;
    await runAction(`${fileName} を削除中`, () =>
      window.illustMK.runtime.deleteModel({ targetType, fileName }), () => {
        if (targetType === "vae" && selectedVae === fileName) {
          setSelectedVae("");
        }
        if (targetType === "clip" && selectedClip === fileName) {
          setSelectedClip("");
        }
      }
    );
    await refreshStatus();
  }

  async function deleteSelectedClip(): Promise<void> {
    if (!selectedClip) return;
    await deleteModelAsset("clip", selectedClip);
  }

  async function installClip(): Promise<void> {
    await installModelAsset("clip", clipUrl, "external_clip.safetensors");
  }

  function saveCheckpointExternalSelection(checkpoint: string, value: CheckpointExternalSelection): void {
    if (!checkpoint) return;
    setCheckpointExternalSelections((current) => {
      const next = { ...current, [checkpoint]: value };
      localStorage.setItem("illustmk.checkpointExternalSelections", JSON.stringify(next));
      return next;
    });
    if (checkpoint === selectedCheckpoint) {
      setSelectedVae(value.vae);
      setSelectedClip(value.clip);
    }
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
    const [nextPaths, nextStatus, nextDataPaths] = await Promise.all([
      window.illustMK.runtime.paths(),
      window.illustMK.runtime.status(comfyUrl),
      window.illustMK.runtime.dataPaths()
    ]);
    setPaths(nextPaths);
    setStatus(nextStatus);
    setDataPaths(nextDataPaths);
    setUsageHistory((current) =>
      [...current, { cpu: nextStatus.systemUsage.cpuPercent, gpu: nextStatus.systemUsage.gpuPercent }].slice(-36)
    );
  }

  async function loadModelLibrary(): Promise<void> {
    const entries = await window.illustMK.models.list();
    setModelLibrary(entries);
    setSelectedModelId((current) => current && entries.some((entry) => entry.id === current) ? current : entries[0]?.id ?? null);
    const active = entries.find((entry) => entry.type === "checkpoint" && entry.active);
    if (active) {
      setSelectedCheckpoint(active.fileName);
      setSelectedVae(active.selectedVae ?? "");
      setSelectedClip(active.selectedClip ?? "");
    }
  }

  async function loadSavedPoses(): Promise<void> {
    const poses = await window.illustMK.poses.list();
    setSavedPoses(poses);
    setSelectedSavedPoseId((current) => current && poses.some((pose) => pose.id === current) ? current : "");
  }

  async function saveCurrentPose(): Promise<void> {
    await runAction("ポーズを保存中", () => window.illustMK.poses.save({
      id: selectedSavedPoseId || undefined,
      title: poseTitle,
      pose: poseConditioning
    }), (poses) => {
      setSavedPoses(poses);
      const saved = poses.find((pose) => pose.id === selectedSavedPoseId) ?? poses[0];
      setSelectedSavedPoseId(saved?.id ?? "");
      setPoseTitle(saved?.title ?? poseTitle);
    });
  }

  function loadSelectedPose(id: string): void {
    const saved = savedPoses.find((pose) => pose.id === id);
    setSelectedSavedPoseId(id);
    if (!saved) return;
    setPoseTitle(saved.title);
    setPoseConditioning(normalizePoseConditioning(saved.pose));
    setRestoredGenerationSettings(null);
  }

  async function deleteSelectedPose(): Promise<void> {
    if (!selectedSavedPoseId || !window.confirm("選択中の保存ポーズを削除します。")) return;
    await runAction("保存ポーズを削除中", () => window.illustMK.poses.delete(selectedSavedPoseId), (poses) => {
      setSavedPoses(poses);
      setSelectedSavedPoseId("");
      setPoseTitle("");
    });
  }

  async function ensurePoseAnalysis(pose: PoseConditioning): Promise<PoseConditioning> {
    return pose;
  }

  async function installPosePreprocessor(): Promise<void> {
    const id = crypto.randomUUID();
    await runAction("DWPoseをインストール中", () => window.illustMK.runtime.runOperation({ id, operation: "pose-preprocessor-install" }), () => {
      void refreshStatus();
    });
  }

  function updateModelDraft(patch: Partial<ModelLibraryEntry>): void {
    if (!selectedModelId) return;
    setModelLibrary((current) => current.map((entry) => entry.id === selectedModelId ? { ...entry, ...patch } : entry));
  }

  async function saveSelectedModel(): Promise<void> {
    const entry = modelLibrary.find((item) => item.id === selectedModelId);
    if (!entry) return;
    await runAction("モデル情報を保存中", () => window.illustMK.models.save({
      id: entry.id,
      name: entry.name,
      baseModel: entry.baseModel,
      usage: entry.usage,
      selectedVae: entry.selectedVae,
      selectedClip: entry.selectedClip
    }), (entries) => {
      setModelLibrary(entries);
      if (entry.active) {
        setSelectedVae(entry.selectedVae ?? "");
        setSelectedClip(entry.selectedClip ?? "");
      }
    });
  }

  async function activateSelectedModel(): Promise<void> {
    const entry = modelLibrary.find((item) => item.id === selectedModelId);
    if (!entry || entry.type !== "checkpoint") return;
    await runAction("Checkpointを切替中", () => window.illustMK.models.setActive(entry.id), (entries) => {
      setModelLibrary(entries);
      const active = entries.find((item) => item.active);
      setSelectedCheckpoint(active?.fileName ?? "");
      setSelectedVae(active?.selectedVae ?? "");
      setSelectedClip(active?.selectedClip ?? "");
    });
  }

  async function importModelLibraryFile(): Promise<void> {
    await runAction("モデルをインポート中", () => window.illustMK.models.importFile(modelImportType), (entries) => {
      setModelLibrary(entries);
      setModelImportOpen(false);
      setSelectedModelId(entries.find((entry) => entry.type === modelImportType)?.id ?? entries[0]?.id ?? null);
      void refreshStatus();
    });
  }

  function civitaiTypesForFilter(type: ModelAssetType): Array<"Checkpoint" | "Controlnet" | "VAE" | "TextualInversion" | "Hypernetwork" | "Poses"> {
    if (type === "checkpoint") return ["Checkpoint"];
    if (type === "controlnet") return ["Controlnet"];
    if (type === "vae") return ["VAE"];
    if (type === "other") return ["TextualInversion", "Hypernetwork", "Poses"];
    return [];
  }

  function targetTypeForCatalogModel(model: CivitaiModelCard): ModelAssetType {
    const type = model.type.toLowerCase();
    if (type === "checkpoint") return "checkpoint";
    if (type === "controlnet") return "controlnet";
    if (type === "vae") return "vae";
    return "other";
  }

  async function searchModelCatalog(top10 = false): Promise<void> {
    const selectedTypes = modelCatalogTypes.filter((type) => type !== "clip");
    if (!selectedTypes.length) {
      setModelCatalogResults([]);
      return;
    }
    await runAction(top10 ? "Civitai人気モデルを取得中" : "Civitaiモデルを検索中", async () => {
      const query = top10 ? undefined : modelCatalogQuery.trim() || undefined;
      const results = await Promise.all(selectedTypes.flatMap(civitaiTypesForFilter).map((type) =>
        window.illustMK.runtime.searchCivitaiPage({
          query,
          sort: modelCatalogSort,
          period: modelCatalogPeriod,
          limit: top10 ? 10 : 24,
          nsfw: false,
          baseModels: type === "Checkpoint" ? baseFilters : undefined,
          type
        })
      ));
      const seen = new Set<string>();
      return results.flatMap((result) => result.models).filter((model) => {
        const key = `${model.id}-${model.versionId}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).sort((a, b) => b.downloadCount - a.downloadCount).slice(0, top10 ? 10 : 60);
    }, (models) => setModelCatalogResults(models));
  }

  async function downloadCatalogModelToLibrary(model: CivitaiModelCard): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    setAction({ busy: true, message: `${model.name} をダウンロード中`, error: null });
    setDownloadProgress(0);
    try {
      const entries = await window.illustMK.models.downloadCivitai({
        model,
        targetType: targetTypeForCatalogModel(model),
        operationId: id
      });
      setModelLibrary(entries);
      setSelectedModelId(entries.find((entry) => entry.civitaiVersionId === model.versionId)?.id ?? null);
      setDownloadProgress(100);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
    } finally {
      currentOperationId.current = null;
    }
  }

  async function downloadDirectModelToLibrary(): Promise<void> {
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    setAction({ busy: true, message: "モデルをダウンロード中", error: null });
    setDownloadProgress(0);
    try {
      const entries = await window.illustMK.models.downloadDirect({
        url: modelDirectUrl,
        fileName: modelDirectFileName || undefined,
        targetType: modelDirectType,
        baseModel: modelDirectBase,
        usage: modelDirectUsage,
        operationId: id
      });
      setModelLibrary(entries);
      setSelectedModelId(entries[0]?.id ?? null);
      setDownloadProgress(100);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
    } finally {
      currentOperationId.current = null;
    }
  }

  async function installModelLibraryRequiredComponent(checkpoint: ModelLibraryEntry, component: CivitaiRequiredComponent): Promise<void> {
    const componentType = component.type.toLowerCase() === "vae" ? "vae" : component.type.toLowerCase() === "text encoder" ? "clip" : null;
    if (!componentType) return;
    const id = crypto.randomUUID();
    currentOperationId.current = id;
    setAction({ busy: true, message: `${component.name} をダウンロード中`, error: null });
    setDownloadProgress(0);
    try {
      let entries = await window.illustMK.models.downloadDirect({
        url: component.downloadUrl,
        fileName: component.name,
        targetType: componentType,
        operationId: id
      });
      entries = await window.illustMK.models.save({
        id: checkpoint.id,
        [componentType === "vae" ? "selectedVae" : "selectedClip"]: component.name
      });
      setModelLibrary(entries);
      if (checkpoint.active) {
        if (componentType === "vae") setSelectedVae(component.name);
        else setSelectedClip(component.name);
      }
      setDownloadProgress(100);
      setAction({ busy: false, message: "完了しました", error: null });
      await refreshStatus();
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText, id);
      setAction({ busy: false, message: "失敗しました", error: errorText });
    } finally {
      currentOperationId.current = null;
    }
  }

  async function deleteSelectedModelLibraryEntry(): Promise<void> {
    const entry = modelLibrary.find((item) => item.id === selectedModelId);
    if (!entry) return;
    if (entry.active) {
      window.alert("アクティブなCheckpointは削除できません。先に別のCheckpointをアクティブにしてください。");
      return;
    }
    const historyCount = promptHistory.filter((history) => JSON.stringify(history).includes(entry.fileName)).length;
    const reference = historyCount ? `\n${historyCount}件の生成履歴がこのファイル名を参照しています。` : "";
    if (!window.confirm(`${entry.name} を削除します。元に戻せません。${reference}`)) return;
    await runAction("モデルを削除中", () => window.illustMK.models.delete(entry.id), (entries) => {
      setModelLibrary(entries);
      setSelectedModelId(entries[0]?.id ?? null);
      void refreshStatus();
    });
  }

  async function reloadDataPaths(): Promise<void> {
    setDataPaths(await window.illustMK.runtime.dataPaths());
  }

  async function addDataPath(): Promise<void> {
    await runAction("DataPathを追加中", () => window.illustMK.runtime.addDataPath(), async (entries) => {
      setDataPaths(entries);
      resetDataPathDependentState();
      await refreshAfterDataPathChange();
    });
  }

  async function importDataPath(): Promise<void> {
    await runAction("DataPathをインポート中", () => window.illustMK.runtime.importDataPath(), (entries) => {
      setDataPaths(entries);
    });
  }

  async function switchActiveDataPath(id: string): Promise<void> {
    await runAction("DataPathを切替中", () => window.illustMK.runtime.switchDataPath(id), async (entries) => {
      setDataPaths(entries);
      resetDataPathDependentState();
      await refreshAfterDataPathChange();
    });
  }

  async function deleteDataPathEntry(id: string): Promise<void> {
    await runAction("DataPath登録を削除中", () => window.illustMK.runtime.deleteDataPath(id), async (entries) => {
      setDataPaths(entries);
      resetDataPathDependentState();
      await refreshAfterDataPathChange();
    });
  }

  function openDataPathMemoEditor(entry: DataPathEntry): void {
    setEditingDataPath(entry);
    setDataPathMemo(entry.memo);
  }

  async function saveDataPathMemo(): Promise<void> {
    if (!editingDataPath) return;
    await runAction(
      "DataPathメモを保存中",
      () => window.illustMK.runtime.updateDataPathMemo({ id: editingDataPath.id, memo: dataPathMemo }),
      (entries) => {
        setDataPaths(entries);
        setEditingDataPath(null);
      }
    );
  }

  async function refreshAfterDataPathChange(): Promise<void> {
    await Promise.all([refreshStatus(), refreshOllamaStatus(), loadLoraProjects(), loadPromptHistory(), loadModelLibrary(), loadSavedPoses()]);
  }

  function resetDataPathDependentState(): void {
    setStatus(null);
    setPaths(null);
    setSelectedCheckpoint("");
    setSelectedVae("");
    setSelectedClip("");
    setSelectedLoras([]);
    setPoseConditioning(createDefaultPoseConditioning());
    setSavedPoses([]);
    setSelectedSavedPoseId("");
    setPoseTitle("");
    setModelLibrary([]);
    setSelectedModelId(null);
    setPositiveLoraTriggerIds([]);
    setSelectedLoraProjectId(null);
    setLoraProjects([]);
    setGenerationResult(null);
    setLoraCompareResult(null);
    setLastImage(null);
    setLastHistoryId(null);
    setRestoredGenerationSettings(null);
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
    setLlmPullName(settings.ollamaModel);
    setOllamaStatus(await window.illustMK.ollama.status(settings.ollamaUrl));
    setSettingsLoaded(true);
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
    if (entry.restorable === false) {
      window.alert(`この履歴は現在のDataPathで不足があります。\n${(entry.missingAssets ?? []).join("\n")}`);
      return;
    }
    const settings = entry.generationSettings;
    let restoredActiveCharacter: GenerationSceneCharacter | undefined;
    if (settings?.scene) {
      const restoredScene = structuredClone(settings.scene);
      setSceneCharacters(restoredScene.characters);
      setSceneBackground(restoredScene.background ? normalizeSceneBackground(restoredScene.background) : null);
      const nextActiveId = restoredScene.activeItemId ?? restoredScene.characters[0]?.id ?? restoredScene.background?.id ?? null;
      setActiveSceneItemId(nextActiveId);
      restoredActiveCharacter = restoredScene.characters.find((character) => character.id === nextActiveId);
    }
    setRestoredGenerationSettings(settings ?? null);
    const restoresBackground = settings?.generationScope === "background";
    if (!restoresBackground) {
      setJapanesePrompt(settings?.japanesePrompt ?? entry.japanesePrompt ?? "");
      setPrompt(settings?.prompt ?? entry.prompt);
      setNegativePrompt(settings?.negativePrompt ?? entry.negativePrompt);
      setPromptEnhanceMode(settings?.promptEnhanceMode ?? "balanced-detail");
    }
    setSelectedCheckpoint(settings?.checkpoint ?? entry.checkpoint);
    setSelectedVae(normalizeExistingAssetName(settings?.vae ?? entry.vae ?? "", status?.vae ?? []));
    setSelectedClip(normalizeExistingAssetName(settings?.clip ?? entry.clip ?? "", status?.clip ?? []));
    const nextOllamaUrl = settings?.ollamaUrl ?? entry.llmUrl;
    const nextOllamaModel = settings?.ollamaModel ?? entry.llmModel;
    if (nextOllamaUrl) setOllamaUrl(nextOllamaUrl);
    if (nextOllamaModel) setOllamaModel(nextOllamaModel);
    if (!restoresBackground) {
      setJapanesePromptTagIds(settings?.japanesePromptTagIds ?? entry.japanesePromptTagIds ?? []);
      setPositivePromptTagIds(settings?.positivePromptTagIds ?? entry.positivePromptTagIds ?? []);
      setPositiveOptionTagIds(settings?.positiveOptionTagIds ?? entry.positiveOptionTagIds ?? []);
      setNegativeOptionTagIds(settings?.negativeOptionTagIds ?? entry.negativeOptionTagIds ?? []);
      setPositiveLoraTriggerIds(settings?.positiveLoraTriggerIds ?? entry.positiveLoraTriggerIds ?? []);
      setSelectedLoras(settings?.loras ?? entry.loras ?? []);
      setGenerationWidth(settings?.width ?? entry.width);
      setGenerationHeight(settings?.height ?? entry.height);
      setImageSizePreset(settings?.imageSizePreset ?? `${entry.width}x${entry.height}`);
      setGenerationSteps(settings?.steps ?? entry.steps);
      setGenerationCfg(settings?.cfg ?? entry.cfg);
      setSeedMode(settings?.seedMode ?? (entry.seed >= 0 ? "fixed" : "random"));
      setSeedValue(settings?.seedValue ?? entry.seed);
      setSeedRange(settings?.seedRange ?? defaultSeedRange);
      setPoseConditioning(normalizePoseConditioning(
        settings?.poseConditioning ?? settings?.generationRequest?.poseConditioning
      ));
    }
    if (settings?.baseFilters) setBaseFilters(settings.baseFilters);
    if (settings?.selectedModelAddMode === "catalog" || settings?.selectedModelAddMode === "url") setModelAddMode(settings.selectedModelAddMode);
    if (
      settings?.generationTab === "character" ||
      settings?.generationTab === "prompt" ||
      settings?.generationTab === "lora" ||
      settings?.generationTab === "options" ||
      settings?.generationTab === "pose"
    ) setGenerateTab(settings.generationTab);
    setImageMemo(entry.memo ?? "");
    setLastHistoryId(entry.id);
    if (entry.imageDataUrl || entry.imagePath) setLastImage(entry.imageDataUrl ?? entry.imagePath ?? null);
    if (restoredActiveCharacter) applyCharacterSettings(restoredActiveCharacter.settings, true);
    setPromptHistoryOpen(false);
  }

  function captureCharacterSettings(): SceneCharacterSettings {
    return {
      japanesePrompt,
      prompt,
      negativePrompt,
      promptEnhanceMode,
      promptNotes,
      loras: structuredClone(selectedLoras),
      japanesePromptTagIds: [...japanesePromptTagIds],
      positivePromptTagIds: [...positivePromptTagIds],
      positiveOptionTagIds: [...positiveOptionTagIds],
      negativeOptionTagIds: [...negativeOptionTagIds],
      positiveLoraTriggerIds: [...positiveLoraTriggerIds],
      poseConditioning: structuredClone(poseConditioning),
      seedMode,
      seedValue,
      seedRange
    };
  }

  function blankCharacterSettings(): SceneCharacterSettings {
    return {
      japanesePrompt: "",
      prompt: "masterpiece, best quality, 1girl, anime style, soft light",
      negativePrompt: "low quality, worst quality, blurry, bad anatomy",
      promptEnhanceMode: "balanced-detail",
      promptNotes: "",
      loras: [],
      japanesePromptTagIds: [],
      positivePromptTagIds: [],
      positiveOptionTagIds: [],
      negativeOptionTagIds: [],
      positiveLoraTriggerIds: [],
      poseConditioning: createDefaultPoseConditioning(),
      seedMode: "random",
      seedValue: -1,
      seedRange: defaultSeedRange
    };
  }

  function applyCharacterSettings(settings: SceneCharacterSettings, preserveRestoredSettings = false): void {
    if (!preserveRestoredSettings) setRestoredGenerationSettings(null);
    setJapanesePrompt(settings.japanesePrompt);
    setPrompt(settings.prompt);
    setNegativePrompt(settings.negativePrompt);
    setPromptEnhanceMode(settings.promptEnhanceMode);
    setPromptNotes(settings.promptNotes);
    setSelectedLoras(structuredClone(settings.loras));
    setJapanesePromptTagIds([...settings.japanesePromptTagIds]);
    setPositivePromptTagIds([...settings.positivePromptTagIds]);
    setPositiveOptionTagIds([...settings.positiveOptionTagIds]);
    setNegativeOptionTagIds([...settings.negativeOptionTagIds]);
    setPositiveLoraTriggerIds([...settings.positiveLoraTriggerIds]);
    setPoseConditioning(normalizePoseConditioning(settings.poseConditioning));
    setSeedMode(settings.seedMode);
    setSeedValue(settings.seedValue);
    setSeedRange(settings.seedRange ?? defaultSeedRange);
    setGenerateTab("prompt");
  }

  function persistActiveSceneCharacter(characters = sceneCharacters): GenerationSceneCharacter[] {
    if (!activeSceneCharacter) return characters;
    const settings = captureCharacterSettings();
    return characters.map((character) => character.id === activeSceneCharacter.id ? { ...character, settings } : character);
  }

  function addSceneCharacter(): void {
    const current = persistActiveSceneCharacter();
    const nextNumber = current.length + 1;
    const usesAutomaticPlacement = current.every((character, index) =>
      character.placement.x === automaticCharacterX(index, current.length) &&
      character.placement.y === defaultCharacterPlacement.y &&
      character.placement.scale === defaultCharacterPlacement.scale
    );
    const character: GenerationSceneCharacter = {
      id: crypto.randomUUID(),
      title: `キャラ${nextNumber}`,
      settings: blankCharacterSettings(),
      placement: {
        ...defaultCharacterPlacement,
        x: automaticCharacterX(nextNumber - 1, nextNumber),
        zIndex: nextNumber
      }
    };
    const nextCharacters = [...current, character];
    setSceneCharacters(usesAutomaticPlacement
      ? nextCharacters.map((item, index) => ({
          ...item,
          placement: { ...item.placement, x: automaticCharacterX(index, nextCharacters.length) }
        }))
      : nextCharacters);
    setActiveSceneItemId(character.id);
    applyCharacterSettings(character.settings);
    setSceneAddOpen(false);
  }

  function activateSceneCharacter(characterId: string): void {
    const current = persistActiveSceneCharacter();
    const target = current.find((character) => character.id === characterId);
    if (!target) return;
    setSceneCharacters(current);
    setActiveSceneItemId(characterId);
    applyCharacterSettings(target.settings);
  }

  function addSceneBackground(): void {
    const current = persistActiveSceneCharacter();
    setSceneCharacters(current);
    const background = sceneBackground ?? createDefaultSceneBackground();
    setSceneBackground(background);
    setActiveSceneItemId(background.id);
    setSceneAddOpen(false);
  }

  function updateActiveCharacterPlacement(patch: Partial<GenerationSceneCharacter["placement"]>): void {
    if (!activeSceneCharacter) return;
    const current = persistActiveSceneCharacter();
    setSceneCharacters(current.map((character) => character.id === activeSceneCharacter.id
      ? { ...character, placement: { ...character.placement, ...patch } }
      : character));
  }

  function updateActiveCharacterTitle(title: string): void {
    if (!activeSceneCharacter) return;
    const current = persistActiveSceneCharacter();
    setSceneCharacters(current.map((character) => character.id === activeSceneCharacter.id ? { ...character, title } : character));
  }

  function saveActiveCharacterPreset(): void {
    if (!activeSceneCharacter) return;
    const saved: SavedSceneCharacter = {
      ...activeSceneCharacter,
      id: crypto.randomUUID(),
      settings: captureCharacterSettings(),
      savedAt: new Date().toISOString()
    };
    const next = [saved, ...savedSceneCharacters].slice(0, 100);
    setSavedSceneCharacters(next);
    localStorage.setItem(savedSceneCharactersKey, JSON.stringify(next));
  }

  function restoreCharacterPreset(saved: SavedSceneCharacter): void {
    if (!activeSceneCharacter) return;
    const restored: GenerationSceneCharacter = {
      ...activeSceneCharacter,
      title: saved.title,
      settings: structuredClone(saved.settings),
      placement: { ...defaultCharacterPlacement, zIndex: activeSceneCharacter.placement.zIndex }
    };
    setSceneCharacters(persistActiveSceneCharacter().map((character) => character.id === restored.id ? restored : character));
    applyCharacterSettings(restored.settings);
    setCharacterLibraryOpen(false);
  }

  function saveBackgroundPreset(): void {
    if (!sceneBackground) return;
    const saved: SavedSceneBackground = { ...structuredClone(sceneBackground), id: crypto.randomUUID(), savedAt: new Date().toISOString() };
    const next = [saved, ...savedSceneBackgrounds].slice(0, 100);
    setSavedSceneBackgrounds(next);
    localStorage.setItem(savedSceneBackgroundsKey, JSON.stringify(next));
  }

  function restoreBackgroundPreset(saved: SavedSceneBackground): void {
    const restored = normalizeSceneBackground({ ...structuredClone(saved), id: sceneBackground?.id ?? crypto.randomUUID() });
    setSceneBackground(restored);
    setActiveSceneItemId(restored.id);
    setSceneCharacters((current) => current.map((character, index) => ({
      ...character,
      placement: { ...defaultCharacterPlacement, zIndex: index + 1 }
    })));
    setBackgroundLibraryOpen(false);
  }

  function deleteCharacterPreset(id: string): void {
    const next = savedSceneCharacters.filter((character) => character.id !== id);
    setSavedSceneCharacters(next);
    localStorage.setItem(savedSceneCharactersKey, JSON.stringify(next));
  }

  function deleteBackgroundPreset(id: string): void {
    const next = savedSceneBackgrounds.filter((background) => background.id !== id);
    setSavedSceneBackgrounds(next);
    localStorage.setItem(savedSceneBackgroundsKey, JSON.stringify(next));
  }

  function removeActiveSceneItem(): void {
    if (!activeSceneItemId) return;
    if (activeSceneCharacter) {
      const next = sceneCharacters.filter((character) => character.id !== activeSceneCharacter.id);
      setSceneCharacters(next);
      const nextItem = next[0] ?? sceneBackground;
      setActiveSceneItemId(nextItem?.id ?? null);
      if (next[0]) applyCharacterSettings(next[0].settings);
      return;
    }
    if (sceneBackground?.id === activeSceneItemId) {
      setSceneBackground(null);
      const nextCharacter = sceneCharacters[0];
      setActiveSceneItemId(nextCharacter?.id ?? null);
      if (nextCharacter) applyCharacterSettings(nextCharacter.settings);
    }
  }

  function restoreCharacterFromHistory(character: GenerationSceneCharacter): void {
    const current = persistActiveSceneCharacter();
    const restored: GenerationSceneCharacter = {
      ...structuredClone(character),
      id: crypto.randomUUID(),
      placement: { ...defaultCharacterPlacement, zIndex: current.length + 1 }
    };
    setSceneCharacters([...current, restored]);
    setActiveSceneItemId(restored.id);
    applyCharacterSettings(restored.settings);
    setPromptHistoryOpen(false);
  }

  function restoreBackgroundFromHistory(background: GenerationSceneBackground): void {
    const restored = normalizeSceneBackground({ ...structuredClone(background), id: sceneBackground?.id ?? crypto.randomUUID() });
    setSceneBackground(restored);
    setSceneCharacters((current) => current.map((character, index) => ({
      ...character,
      placement: { ...defaultCharacterPlacement, zIndex: index + 1 }
    })));
    setActiveSceneItemId(restored.id);
    setPromptHistoryOpen(false);
  }

  function updateSceneBackground(patch: Partial<GenerationSceneBackground>): void {
    setSceneBackground((current) => current ? { ...current, ...patch } : current);
  }

  function importBackgroundReference(file: File | undefined): void {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => updateSceneBackground({ referenceImageDataUrl: String(reader.result ?? "") });
    reader.readAsDataURL(file);
  }

  function currentGenerationScene(): GenerationSceneSettings {
    return {
      characters: persistActiveSceneCharacter().map((character) => structuredClone(character)),
      background: sceneBackground ? structuredClone(sceneBackground) : undefined,
      activeItemId: activeSceneItemId ?? undefined
    };
  }

  function resetGenerationParameters(): void {
    setRestoredGenerationSettings(null);
    setJapanesePrompt("");
    setPromptNotes("");
    setPromptEnhanceMode("balanced-detail");
    setPrompt("masterpiece, best quality, 1girl, anime style, soft light");
    setNegativePrompt("low quality, worst quality, blurry, bad anatomy");
    setJapanesePromptTagIds([]);
    setPositivePromptTagIds([]);
    setPositiveOptionTagIds([]);
    setNegativeOptionTagIds([]);
    setPositiveLoraTriggerIds([]);
    setSelectedLoras([]);
    setImageSizePreset("768x1024");
    setGenerationWidth(768);
    setGenerationHeight(1024);
    setGenerationSteps(20);
    setGenerationCfg(7);
    setSeedMode("random");
    setSeedValue(-1);
    setSeedRange(defaultSeedRange);
    setPoseConditioning(createDefaultPoseConditioning());
    setGenerationResult(null);
    setLastImage(null);
    setLastHistoryId(null);
    setImageMemo("");
    setRepeatItems([]);
    setRepeatPreviewImage(null);
    setGenerateTab("prompt");
    setSelectedCheckpoint(status?.checkpoints[0]?.name ?? "");
    setSelectedVae("");
    setSelectedClip("");
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

  function updateBackgroundImageSizePreset(value: string): void {
    if (!sceneBackground) return;
    if (value === "custom") {
      updateSceneBackground({ imageSizePreset: value });
      return;
    }
    const [width, height] = value.split("x").map(Number);
    updateSceneBackground({
      imageSizePreset: value,
      ...(Number.isFinite(width) && Number.isFinite(height) ? { width, height } : {})
    });
  }

  function openNewTagDialog(category = ""): void {
    setTagEditingId(null);
    setTagName("");
    setTagEnglishName("");
    setTagColor("#3388ff");
    setTagCategory(category);
    setTagPrompt("");
    setTagJapaneseDescription("");
    setTagEnglishDescription("");
    setTagDialogOpen(true);
  }

  function openEditTagDialog(tag: IllustrationTag): void {
    setTagEditingId(tag.id);
    setTagName(tag.name);
    setTagEnglishName(tag.englishName || tag.name);
    setTagColor(tag.color || "#3388ff");
    setTagCategory(tag.category || "未分類");
    setTagPrompt(tag.sourcePrompt);
    setTagJapaneseDescription(tag.japaneseDescription);
    setTagEnglishDescription(tag.englishDescription);
    setTagDialogOpen(true);
  }

  function openNewTagDialogForCategory(category: string): void {
    openNewTagDialog(category);
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
          color: tagColor,
          category: tagCategory.trim() || "未分類",
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
      setJapanesePromptTagIds((current) => current.filter((tagId) => tagId !== id));
      setPositivePromptTagIds((current) => current.filter((tagId) => tagId !== id));
      if (tagEditingId === id) {
        setTagDialogOpen(false);
        setTagEditingId(null);
      }
    });
  }

  function toggleIllustrationTag(id: string): void {
    setSelectedTagIds((current) => (current.includes(id) ? current.filter((tagId) => tagId !== id) : [...current, id]));
  }

  function applyIllustrationTagToPrompt(tag: IllustrationTag): void {
    if (tagPromptTarget === "background-positive") {
      updateSceneBackground({
        positivePromptTagIds: sceneBackground?.positivePromptTagIds.includes(tag.id)
          ? sceneBackground.positivePromptTagIds
          : [...(sceneBackground?.positivePromptTagIds ?? []), tag.id]
      });
      return;
    }
    if (tagPromptTarget === "background-japanese") {
      updateSceneBackground({
        japanesePromptTagIds: sceneBackground?.japanesePromptTagIds.includes(tag.id)
          ? sceneBackground.japanesePromptTagIds
          : [...(sceneBackground?.japanesePromptTagIds ?? []), tag.id]
      });
      return;
    }
    if (tagPromptTarget === "positive") {
      setPositivePromptTagIds((current) => (current.includes(tag.id) ? current : [...current, tag.id]));
      return;
    }
    setJapanesePromptTagIds((current) => (current.includes(tag.id) ? current : [...current, tag.id]));
  }

  function persistOptionPromptTags(tags: OptionPromptTag[]): void {
    setOptionPromptTags(tags);
    localStorage.setItem("illustmk.optionPromptTags", JSON.stringify(tags));
  }

  function openOptionTagEditor(tag?: OptionPromptTag): void {
    setOptionTagEditingId(tag?.id ?? null);
    setOptionTagJapanese(tag?.japanese ?? "");
    setOptionTagEnglish(tag?.english ?? "");
    setOptionTagColor(tag?.color ?? "#3388ff");
    setOptionTagCategory(tag ? tag.category || "未分類" : "");
    setOptionTagEditorOpen(true);
  }

  function saveOptionPromptTag(): void {
    const japanese = optionTagJapanese.trim();
    const english = optionTagEnglish.trim();
    if (!japanese || !english) return;
    const nextTag: OptionPromptTag = {
      id: optionTagEditingId ?? crypto.randomUUID(),
      japanese,
      english,
      color: optionTagColor || "#3388ff",
      category: optionTagCategory.trim() || "未分類"
    };
    persistOptionPromptTags(
      optionTagEditingId
        ? optionPromptTags.map((tag) => (tag.id === optionTagEditingId ? nextTag : tag))
        : [...optionPromptTags, nextTag]
    );
    setOptionTagEditorOpen(false);
  }

  function deleteOptionPromptTag(): void {
    if (!optionTagEditingId) return;
    const target = optionPromptTags.find((tag) => tag.id === optionTagEditingId);
    persistOptionPromptTags(optionPromptTags.filter((tag) => tag.id !== optionTagEditingId));
    if (target) {
      setPositiveOptionTagIds((current) => current.filter((id) => id !== target.id));
      setNegativeOptionTagIds((current) => current.filter((id) => id !== target.id));
      updateSceneBackground({
        positiveOptionTagIds: sceneBackground?.positiveOptionTagIds.filter((id) => id !== target.id) ?? [],
        negativeOptionTagIds: sceneBackground?.negativeOptionTagIds.filter((id) => id !== target.id) ?? []
      });
    }
    setOptionTagEditorOpen(false);
  }

  function applyOptionPromptTag(tag: OptionPromptTag): void {
    if (optionTagDialogTarget === "background-negative") {
      updateSceneBackground({
        negativeOptionTagIds: sceneBackground?.negativeOptionTagIds.includes(tag.id)
          ? sceneBackground.negativeOptionTagIds
          : [...(sceneBackground?.negativeOptionTagIds ?? []), tag.id]
      });
      return;
    }
    if (optionTagDialogTarget === "background-positive") {
      updateSceneBackground({
        positiveOptionTagIds: sceneBackground?.positiveOptionTagIds.includes(tag.id)
          ? sceneBackground.positiveOptionTagIds
          : [...(sceneBackground?.positiveOptionTagIds ?? []), tag.id]
      });
      return;
    }
    if (optionTagDialogTarget === "negative") {
      setNegativeOptionTagIds((current) => (current.includes(tag.id) ? current : [...current, tag.id]));
      return;
    }
    setPositiveOptionTagIds((current) => (current.includes(tag.id) ? current : [...current, tag.id]));
  }

  function removeOptionPromptTag(target: PromptKind, tag: OptionPromptTag): void {
    if (target === "background-negative") {
      updateSceneBackground({ negativeOptionTagIds: sceneBackground?.negativeOptionTagIds.filter((id) => id !== tag.id) ?? [] });
      return;
    }
    if (target === "background-positive") {
      updateSceneBackground({ positiveOptionTagIds: sceneBackground?.positiveOptionTagIds.filter((id) => id !== tag.id) ?? [] });
      return;
    }
    if (target === "negative") {
      setNegativeOptionTagIds((current) => current.filter((id) => id !== tag.id));
      return;
    }
    setPositiveOptionTagIds((current) => current.filter((id) => id !== tag.id));
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
    setLoraTriggerWords([""]);
    setLoraCreateDialogOpen(true);
  }

  function openEditLoraDialog(): void {
    if (!selectedLoraProject) return;
    setLoraProjectDialogMode("edit");
    setEditingLoraProjectId(selectedLoraProject.id);
    setLoraName(selectedLoraProject.name);
    setLoraTriggerWord(selectedLoraProject.triggerWord);
    setLoraTriggerWords(getLoraTriggerWords(selectedLoraProject).length ? getLoraTriggerWords(selectedLoraProject) : [""]);
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
          triggerWord: normalizedTriggerWords(loraTriggerWords, loraTriggerWord)[0] ?? "",
          triggerWords: normalizedTriggerWords(loraTriggerWords, loraTriggerWord),
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

  async function saveSelectedLoraDetails(): Promise<void> {
    if (!selectedLoraProject) return;
    const targetProjectId = selectedLoraProject.id;
    await runAction(
      "LoRA設定を更新中",
      () =>
        window.illustMK.loraProjects.save({
          id: targetProjectId,
          name: loraDetailName,
          triggerWord: normalizedTriggerWords(loraDetailTriggerWords, loraDetailTriggerWord)[0] ?? "",
          triggerWords: normalizedTriggerWords(loraDetailTriggerWords, loraDetailTriggerWord),
          baseCheckpoint: selectedLoraProject.baseCheckpoint
        }),
      (projects) => {
        setLoraProjects(projects);
        setSelectedLoraProjectId(targetProjectId);
      }
    );
  }

  function allowLoraThumbnailDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }

  async function updateSelectedLoraThumbnail(event: DragEvent<HTMLDivElement>): Promise<void> {
    event.preventDefault();
    event.stopPropagation();
    if (!selectedLoraProject) return;
    const file = event.dataTransfer.files[0];
    const droppedUrl = extractDroppedImageUrl(event.dataTransfer);
    let request:
      | { id: string; dataUrl: string }
      | { id: string; sourceUrl: string }
      | null = null;
    if (file) {
      if (!isSupportedImageFile(file)) {
        appendLog("stderr", "PNG / JPEG / WebP の画像をドロップしてください。");
        return;
      }
      request = { id: selectedLoraProject.id, dataUrl: await fileToDataUrl(file) };
    } else if (droppedUrl) {
      request = { id: selectedLoraProject.id, sourceUrl: droppedUrl };
    }
    if (!request) {
      appendLog("stderr", "サムネイル画像を取得できませんでした。画像ファイルまたは画像URLをドロップしてください。");
      return;
    }
    const targetProjectId = selectedLoraProject.id;
    await runAction(
      "LoRAサムネイルを更新中",
      () => window.illustMK.loraProjects.updateThumbnail(request),
      (projects) => {
        setLoraProjects(projects);
        setSelectedLoraProjectId(targetProjectId);
      }
    );
  }

  function isSupportedImageFile(file: File): boolean {
    if (["image/png", "image/jpeg", "image/webp"].includes(file.type)) return true;
    return /\.(png|jpe?g|webp)$/i.test(file.name);
  }

  function fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          resolve(reader.result);
          return;
        }
        reject(new Error("サムネイル画像の読み込みに失敗しました。"));
      };
      reader.onerror = () => reject(new Error("サムネイル画像の読み込みに失敗しました。"));
      reader.readAsDataURL(file);
    });
  }

  function extractDroppedImageUrl(dataTransfer: DataTransfer): string {
    const uriList = dataTransfer
      .getData("text/uri-list")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line && !line.startsWith("#"));
    if (uriList) return uriList;
    const html = dataTransfer.getData("text/html");
    const imageSource = html.match(/<img[^>]+src=["']([^"']+)["']/i)?.[1];
    if (imageSource) return imageSource;
    const plainText = dataTransfer.getData("text/plain").trim();
    return /^https?:\/\//i.test(plainText) ? plainText : "";
  }

  async function deleteSelectedLoraProject(): Promise<void> {
    if (!selectedLoraProject) return;
    if (!window.confirm(`${selectedLoraProject.name} を削除します。\nDatasetを含むプロジェクトディレクトリと、生成タブに表示される出力LoRAファイルも削除します。元に戻せません。`)) return;
    const deletedOutputName = selectedLoraProject.outputName;
    await runAction("LoRAを削除中", () => window.illustMK.loraProjects.delete({ id: selectedLoraProject.id, deleteOutputFile: true }), (projects) => {
      setLoraProjects(projects);
      setSelectedLoraProjectId(projects[0]?.id ?? null);
      setSelectedLoras((current) => current.filter((lora) => lora.name !== deletedOutputName));
    });
    await refreshStatus();
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

  async function enhanceBackgroundPrompt(): Promise<void> {
    if (!sceneBackground) return;
    await runAction(
      "背景の日本語プロンプトを生成向けに変換中",
      () => window.illustMK.ollama.enhancePrompt({
        ollamaUrl,
        model: ollamaModel,
        japanesePrompt: sceneBackground.japanesePrompt,
        currentPrompt: sceneBackground.prompt,
        currentNegativePrompt: sceneBackground.negativePrompt,
        selectedTags: backgroundJapanesePromptTags.map((tag) => ({
          name: tag.name,
          englishName: tag.englishName || tag.name,
          englishDescription: tag.englishDescription
        })),
        mode: sceneBackground.promptEnhanceMode
      }),
      (result) => updateSceneBackground({
        prompt: result.positivePrompt,
        negativePrompt: result.negativePrompt,
        promptNotes: result.notes
      })
    );
  }

  function imageUrl(path: string): string {
    if (path.startsWith("data:")) return path;
    return `file://${path}`;
  }

  function updateTriggerWordList(target: "dialog" | "detail", index: number, value: string): void {
    const setter = target === "dialog" ? setLoraTriggerWords : setLoraDetailTriggerWords;
    setter((current) => current.map((item, itemIndex) => (itemIndex === index ? value : item)));
    if (index === 0) {
      if (target === "dialog") setLoraTriggerWord(value);
      else setLoraDetailTriggerWord(value);
    }
  }

  function addTriggerWordRow(target: "dialog" | "detail"): void {
    const setter = target === "dialog" ? setLoraTriggerWords : setLoraDetailTriggerWords;
    setter((current) => [...current, ""]);
  }

  function removeTriggerWordRow(target: "dialog" | "detail", index: number): void {
    const setter = target === "dialog" ? setLoraTriggerWords : setLoraDetailTriggerWords;
    setter((current) => {
      const next = current.filter((_item, itemIndex) => itemIndex !== index);
      return next.length ? next : [""];
    });
  }

  async function saveCurrentHistoryMemo(): Promise<void> {
    if (!lastHistoryId) return;
    await runAction("生成メモを更新中", () => window.illustMK.promptHistory.updateMemo({ id: lastHistoryId, memo: imageMemo }), (history) => {
      setPromptHistory(history);
    });
  }

  function openRepeatMemoEditor(item: RepeatGenerationItem): void {
    if (!item.historyId) return;
    setRepeatMemoEditingItemId(item.id);
    setRepeatMemoText(item.memo ?? "");
  }

  async function saveRepeatHistoryMemo(): Promise<void> {
    const item = repeatItems.find((current) => current.id === repeatMemoEditingItemId);
    if (!item?.historyId) return;
    const memo = repeatMemoText;
    await runAction("Repeat生成メモを更新中", () => window.illustMK.promptHistory.updateMemo({ id: item.historyId!, memo }), (history) => {
      setPromptHistory(history);
      setRepeatItems((current) => current.map((repeatItem) => (repeatItem.id === item.id ? { ...repeatItem, memo } : repeatItem)));
      if (lastHistoryId === item.historyId) setImageMemo(memo);
      setRepeatMemoEditingItemId(null);
      setRepeatMemoText("");
    });
  }

  async function deletePromptHistory(entry: PromptHistoryEntry): Promise<void> {
    if (!window.confirm("この履歴を削除します。元に戻せません。")) return;
    await runAction("プロンプト履歴を削除中", () => window.illustMK.promptHistory.delete({ id: entry.id }), (history) => {
      setPromptHistory(history);
      setRepeatItems((current) => current.map((item) => (item.historyId === entry.id ? { ...item, historyId: undefined } : item)));
      if (lastHistoryId === entry.id) {
        setLastHistoryId(null);
        setImageMemo("");
      }
    });
  }

  function toggleBaseFilter(filter: string): void {
    setBaseFilters((current) => (current.includes(filter) ? current.filter((item) => item !== filter) : [...current, filter]));
  }

  function composeCharacterRequest(character: GenerationSceneCharacter, index: number, includePlacement = true): { prompt: string; negativePrompt: string } {
    const settings = character.settings;
    const positiveTags = illustrationTags
      .filter((tag) => settings.positivePromptTagIds.includes(tag.id))
      .map((tag) => tag.englishDescription);
    const positiveOptions = optionPromptTags
      .filter((tag) => settings.positiveOptionTagIds.includes(tag.id))
      .map((tag) => tag.english);
    const negativeOptions = optionPromptTags
      .filter((tag) => settings.negativeOptionTagIds.includes(tag.id))
      .map((tag) => tag.english);
    const triggerWords = settings.loras.flatMap((lora) => {
      const project = loraProjects.find((item) => item.outputName === lora.name || item.sourceLora === lora.name);
      return getLoraTriggerWords(project).filter((word, triggerIndex) =>
        settings.positiveLoraTriggerIds.includes(`${lora.name}::${triggerIndex}::${word}`)
      );
    });
    const placement = character.placement;
    const placementPrompt = includePlacement
      ? `character ${index + 1} ${character.title}, position x ${placement.x} percent, y ${placement.y} percent, scale ${placement.scale} percent, layer ${placement.zIndex}`
      : `character ${character.title}`;
    return {
      prompt: composeGenerationPrompt(settings.prompt, [...positiveTags, ...positiveOptions, ...triggerWords, placementPrompt]),
      negativePrompt: composeGenerationPrompt(settings.negativePrompt, negativeOptions)
    };
  }

  function composeBackgroundRequest(background: GenerationSceneBackground): { prompt: string; negativePrompt: string } {
    const positiveTags = illustrationTags
      .filter((tag) => background.positivePromptTagIds.includes(tag.id))
      .map((tag) => tag.englishDescription);
    const positiveOptions = optionPromptTags
      .filter((tag) => background.positiveOptionTagIds.includes(tag.id))
      .map((tag) => tag.english);
    const negativeOptions = optionPromptTags
      .filter((tag) => background.negativeOptionTagIds.includes(tag.id))
      .map((tag) => tag.english);
    return {
      prompt: composeGenerationPrompt(background.prompt, [...positiveTags, ...positiveOptions]),
      negativePrompt: composeGenerationPrompt(background.negativePrompt, negativeOptions)
    };
  }

  function createGenerationRequest(scope: GenerationScope, seed: number, activePose = poseConditioning): TestGenerationRequest {
    const poseEnabled = activePose.enabled;
    const scene = currentGenerationScene();
    const targetCharacter = scene.characters.find((character) => character.id === activeSceneItemId);
    if (scope === "character" && targetCharacter) {
      const characterRequest = composeCharacterRequest(targetCharacter, 0, false);
      return {
        prompt: composeGenerationPrompt(characterRequest.prompt, poseEnabled ? [activePose.positivePrompt ?? ""] : []),
        negativePrompt: composeGenerationPrompt(characterRequest.negativePrompt, poseEnabled ? [activePose.negativePrompt ?? ""] : []),
        checkpoint: selectedCheckpoint,
        loras: targetCharacter.settings.loras.filter((lora) => availableLoraNames.has(lora.name)),
        width: generationWidth,
        height: generationHeight,
        steps: generationSteps,
        cfg: generationCfg,
        seed,
        comfyUrl,
        vae: selectedVae || undefined,
        clip: selectedClip || undefined,
        poseConditioning: poseEnabled ? activePose : undefined
      };
    }
    if (scope === "background" && scene.background) {
      const backgroundRequest = composeBackgroundRequest(scene.background);
      return {
        prompt: composeGenerationPrompt(backgroundRequest.prompt, [backgroundOnlyPositivePrompt]),
        negativePrompt: composeGenerationPrompt(backgroundRequest.negativePrompt, [backgroundOnlyNegativePrompt]),
        checkpoint: selectedCheckpoint,
        loras: [],
        width: scene.background.width,
        height: scene.background.height,
        steps: scene.background.steps,
        cfg: scene.background.cfg,
        seed,
        comfyUrl,
        vae: selectedVae || undefined,
        clip: selectedClip || undefined
      };
    }
    const sceneCharactersForRequest = scene.characters.map((character, index) => composeCharacterRequest(character, index));
    const regionalCharacters = scene.characters.map((character, index) => {
      const characterRequest = composeCharacterRequest(character, index, false);
      return {
        id: character.id,
        title: character.title,
        prompt: characterRequest.prompt,
        negativePrompt: characterRequest.negativePrompt,
        placement: character.placement
      };
    });
    const hasScene = sceneCharactersForRequest.length > 0 || Boolean(scene.background);
    const backgroundRequest = scene.background ? composeBackgroundRequest(scene.background) : { prompt: "", negativePrompt: "" };
    const scenePrompt = composeGenerationPrompt(
      backgroundRequest.prompt,
      sceneCharactersForRequest.map((character) => character.prompt)
    );
    const sceneNegativePrompt = composeGenerationPrompt(
      backgroundRequest.negativePrompt,
      sceneCharactersForRequest.map((character) => character.negativePrompt)
    );
    const sceneLoras = Array.from(new Map(
      scene.characters.flatMap((character) => character.settings.loras).map((lora) => [lora.name, lora])
    ).values()).filter((lora) => availableLoraNames.has(lora.name));
    return {
      prompt: composeGenerationPrompt(hasScene ? scenePrompt : effectivePrompt, poseEnabled ? [activePose.positivePrompt ?? ""] : []),
      negativePrompt: composeGenerationPrompt(hasScene ? sceneNegativePrompt : effectiveNegativePrompt, poseEnabled ? [activePose.negativePrompt ?? ""] : []),
      checkpoint: selectedCheckpoint,
      loras: hasScene ? sceneLoras : activeGenerationLoras,
      width: generationWidth,
      height: generationHeight,
      steps: generationSteps,
      cfg: generationCfg,
      seed,
      comfyUrl,
      vae: selectedVae || undefined,
      clip: selectedClip || undefined,
      poseConditioning: poseEnabled ? activePose : undefined,
      sceneConditioning: hasScene ? {
        backgroundPrompt: backgroundRequest.prompt,
        backgroundNegativePrompt: backgroundRequest.negativePrompt,
        characters: regionalCharacters
      } : undefined
    };
  }

  function nextGenerationSeed(): number {
    if (seedMode === "fixed" && seedValue >= 0) return seedValue;
    if (seedMode === "sequence") return parseSeedRange(seedRange).start;
    return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
  }

  function nextGenerationSeedForScope(scope: GenerationScope): number {
    if (scope === "background" && sceneBackground) {
      if (sceneBackground.seedMode === "fixed" && sceneBackground.seedValue >= 0) return sceneBackground.seedValue;
      if (sceneBackground.seedMode === "sequence") return parseSeedRange(sceneBackground.seedRange).start;
      return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    }
    return nextGenerationSeed();
  }

  function nextRandomSeed(excludes = new Set<number>()): number {
    let seed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    while (excludes.has(seed)) {
      seed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    }
    return seed;
  }

  function requestWithConcreteSeed(request: TestGenerationRequest): TestGenerationRequest {
    return request.seed >= 0 ? request : { ...request, seed: nextGenerationSeed() };
  }

  async function generateAndSave(request: TestGenerationRequest, startedAt: number, scope: GenerationScope): Promise<{ result: TestGenerationResult; history: PromptHistoryEntry[]; durationText: string }> {
    const result = await window.illustMK.runtime.testGenerate(request);
    appendLog("info", `ComfyUI Prompt ID: ${result.promptId ?? "なし"}`);
    const durationText = formatGenerationDuration(performance.now() - startedAt);
    const resultWithDuration = {
      ...result,
      message: result.message ? `${result.message} (生成時間 ${durationText})` : `(生成時間 ${durationText})`
    };
    const backgroundHistory = scope === "background" ? sceneBackground : null;
    const history = await window.illustMK.promptHistory.save({
      ...request,
      prompt: backgroundHistory?.prompt ?? prompt,
      negativePrompt: backgroundHistory?.negativePrompt ?? negativePrompt,
      japanesePrompt: backgroundHistory?.japanesePrompt ?? japanesePrompt,
      llmModel: ollamaModel,
      llmUrl: ollamaUrl,
      japanesePromptTagIds: backgroundHistory?.japanesePromptTagIds ?? japanesePromptTagIds,
      positivePromptTagIds: backgroundHistory?.positivePromptTagIds ?? positivePromptTagIds,
      positiveOptionTagIds: backgroundHistory?.positiveOptionTagIds ?? positiveOptionTagIds,
      negativeOptionTagIds: backgroundHistory?.negativeOptionTagIds ?? negativeOptionTagIds,
      positiveLoraTriggerIds: backgroundHistory ? [] : positiveLoraTriggerIds,
      memo: imageMemo,
      generationSettings: buildGenerationHistorySettings(request, scope),
      imagePath: result.imagePath,
      imageDataUrl: result.imageDataUrl
    });
    return { result: resultWithDuration, history, durationText };
  }

  async function waitWhileRepeatPaused(): Promise<void> {
    while (repeatPausedRef.current && !repeatStopRequested.current) {
      await new Promise((resolve) => window.setTimeout(resolve, 300));
    }
  }

  async function runSingleGenerate(scope: GenerationScope): Promise<void> {
    const allowed = scope === "character" ? canGenerateCharacter : scope === "background" ? canGenerateBackground : canGenerateScene;
    if (!allowed) return;
    await runAction(
      scope === "character" ? "キャラクタを生成中" : scope === "background" ? "背景を生成中" : "シーンを総合生成中",
      async () => {
        const startedAt = performance.now();
        const restoredScope = restoredGenerationSettings?.generationScope ?? "character";
        const usesRestoredRequest = Boolean(
          restoredRequestApplies &&
          restoredGenerationSettings?.generationRequest &&
          restoredScope === scope &&
          (scope !== "character" || restoredGenerationSettings?.targetCharacterId === activeSceneCharacter?.id)
        );
        const sourcePose = normalizePoseConditioning(
          scope === "background" ? undefined : usesRestoredRequest ? testRequest.poseConditioning : poseConditioning
        );
        const analyzedPose = await ensurePoseAnalysis(sourcePose);
        const rawRequest = usesRestoredRequest
          ? { ...testRequest, poseConditioning: analyzedPose.enabled ? analyzedPose : undefined }
          : createGenerationRequest(scope, nextGenerationSeedForScope(scope), analyzedPose);
        const request = requestWithConcreteSeed(rawRequest);
        const seedSource = usesRestoredRequest
          ? rawRequest.seed >= 0
            ? "history.generationSettings.generationRequest.seed"
            : "history.generationSettings.generationRequest.seed was negative; app generated concrete seed"
          : scope === "background" && sceneBackground?.seedMode === "fixed" && sceneBackground.seedValue >= 0
            ? "background.seedValue"
            : scope === "background" && sceneBackground?.seedMode === "sequence"
              ? "background.seedRange first value"
            : seedMode === "fixed" && seedValue >= 0
            ? "ui.seedValue"
            : seedMode === "sequence"
              ? "ui.seedRange first value"
            : "app generated random seed";
        const historySettings = buildGenerationHistorySettings(request, scope);
        appendLog("info", `生成種別: ${scope === "character" ? `キャラクタ (${activeSceneCharacter?.title ?? "未選択"})` : scope === "background" ? `背景 (${sceneBackground?.title ?? "未選択"})` : "総合"}`);
        appendGenerationDiagnostics(usesRestoredRequest ? "history-restored" : "manual-ui", rawRequest, request, seedSource, historySettings);
        const requestLoras = request.loras ?? [];
        appendLog("info", `ComfyUI Positive Prompt: ${request.prompt}`);
        appendLog("info", `ComfyUI Negative Prompt: ${request.negativePrompt}`);
        appendLog("info", `ComfyUI LoRA: ${requestLoras.length ? requestLoras.map((lora) => `${lora.name} model=${lora.modelStrength} clip=${lora.clipStrength}`).join(", ") : "なし"}`);
        appendLog("info", `ComfyUI Size/Steps/CFG/Seed: ${request.width}x${request.height} / ${request.steps} / ${request.cfg} / ${request.seed}`);
        return generateAndSave(request, startedAt, scope);
      },
      ({ result, history }) => {
        setGenerationResult(result);
        setPromptHistory(history);
        setLastHistoryId(history[0]?.id ?? null);
        if (result.imageDataUrl || result.imagePath) setLastImage(result.imageDataUrl ?? result.imagePath ?? null);
      }
    );
  }

  async function runRepeatGenerate(scope: GenerationScope): Promise<void> {
    if (repeatRunning || (scope === "character" ? !canGenerateCharacter : scope === "background" ? !canGenerateBackground : !canGenerateScene)) return;
    let repeatPose: PoseConditioning;
    let sequenceRange: { current: number; end: number } | null = null;
    try {
      repeatPose = await ensurePoseAnalysis(poseConditioning);
      const scopeSeedMode = scope === "background" ? sceneBackground?.seedMode : seedMode;
      const scopeSeedRange = scope === "background" ? sceneBackground?.seedRange : seedRange;
      if (scopeSeedMode === "sequence") {
        const parsed = parseSeedRange(scopeSeedRange ?? defaultSeedRange);
        sequenceRange = { current: parsed.start, end: parsed.end };
      }
    } catch (error) {
      const errorText = error instanceof Error ? error.message : String(error);
      appendLog("stderr", errorText);
      setAction({ busy: false, message: "失敗しました", error: errorText });
      return;
    }
    setRepeatDialogOpen(true);
    setRepeatItems([]);
    setRepeatPreviewImage(null);
    setRepeatRunning(true);
    setRepeatPaused(false);
    repeatStopRequested.current = false;
    repeatPausedRef.current = false;
    const scopeLabel = scope === "character" ? "キャラクタ" : scope === "background" ? "背景" : "総合";
    appendLog("info", `${scopeLabel}リピート生成を開始しました。`);
    const usedSeeds = new Set<number>();
    try {
      while (!repeatStopRequested.current) {
        await waitWhileRepeatPaused();
        if (repeatStopRequested.current) break;
        if (sequenceRange && sequenceRange.current > sequenceRange.end) break;
        const seed = sequenceRange ? sequenceRange.current++ : nextRandomSeed(usedSeeds);
        usedSeeds.add(seed);
        const request = createGenerationRequest(scope, seed, repeatPose);
        const itemId = crypto.randomUUID();
        setRepeatItems((current) => [{ id: itemId, seed, startedAt: new Date().toISOString() }, ...current]);
        try {
          const startedAt = performance.now();
          appendLog("info", `${scope === "character" ? "Character" : scope === "background" ? "Background" : "Scene"} Repeat Generate Seed: ${seed}`);
          appendGenerationDiagnostics(
            "manual-ui",
            request,
            request,
            sequenceRange ? "repeat sequence seed generated by app" : "repeat random seed generated by app",
            buildGenerationHistorySettings(request, scope)
          );
          const requestLoras = request.loras ?? [];
          appendLog("info", `ComfyUI Positive Prompt: ${request.prompt}`);
          appendLog("info", `ComfyUI Negative Prompt: ${request.negativePrompt}`);
          appendLog("info", `ComfyUI LoRA: ${requestLoras.length ? requestLoras.map((lora) => `${lora.name} model=${lora.modelStrength} clip=${lora.clipStrength}`).join(", ") : "なし"}`);
          appendLog("info", `ComfyUI Size/Steps/CFG/Seed: ${request.width}x${request.height} / ${request.steps} / ${request.cfg} / ${request.seed}`);
          const { result, history, durationText } = await generateAndSave(request, startedAt, scope);
          setGenerationResult(result);
          setPromptHistory(history);
          setLastHistoryId(history[0]?.id ?? null);
          if (result.imageDataUrl || result.imagePath) {
            setLastImage(result.imageDataUrl ?? result.imagePath ?? null);
          }
          setRepeatItems((current) =>
            current.map((item) => (item.id === itemId ? { ...item, result, durationText, historyId: history[0]?.id, memo: imageMemo } : item))
          );
        } catch (error) {
          const errorText = error instanceof Error ? error.message : String(error);
          appendLog("stderr", errorText);
          setRepeatItems((current) => current.map((item) => (item.id === itemId ? { ...item, error: errorText } : item)));
        }
      }
    } finally {
      setRepeatRunning(false);
      setRepeatPaused(false);
      repeatStopRequested.current = false;
      repeatPausedRef.current = false;
      appendLog("info", `${scopeLabel}リピート生成を停止しました。`);
    }
  }

  function generationDiagnosticsJson(value: unknown): string {
    return JSON.stringify(value, null, 2);
  }

  function appendGenerationDiagnostics(
    route: "manual-ui" | "history-restored",
    rawRequest: TestGenerationRequest,
    finalRequest: TestGenerationRequest,
    seedSource: string,
    historySettings: PromptGenerationSettings
  ): void {
    const manualPrefix = historySettings.generationScope === "background" ? "sceneBackground" : "ui";
    const sourceMap = {
      route,
      prompt: route === "history-restored" ? "history.generationSettings.generationRequest.prompt" : `${manualPrefix}.effectivePrompt`,
      negativePrompt: route === "history-restored" ? "history.generationSettings.generationRequest.negativePrompt" : `${manualPrefix}.effectiveNegativePrompt`,
      checkpoint: route === "history-restored" ? "history.generationSettings.generationRequest.checkpoint" : "ui.selectedCheckpoint",
      vae: route === "history-restored" ? "history.generationSettings.generationRequest.vae" : "ui.selectedVae",
      clip: route === "history-restored" ? "history.generationSettings.generationRequest.clip" : "ui.selectedClip",
      loras: route === "history-restored" ? "history.generationSettings.generationRequest.loras" : historySettings.generationScope === "background" ? "background generation does not apply LoRA" : "ui.activeGenerationLoras",
      width: route === "history-restored" ? "history.generationSettings.generationRequest.width" : `${manualPrefix}.width`,
      height: route === "history-restored" ? "history.generationSettings.generationRequest.height" : `${manualPrefix}.height`,
      steps: route === "history-restored" ? "history.generationSettings.generationRequest.steps" : `${manualPrefix}.steps`,
      cfg: route === "history-restored" ? "history.generationSettings.generationRequest.cfg" : `${manualPrefix}.cfg`,
      seed: seedSource,
      comfyUrl: route === "history-restored" ? "history.generationSettings.generationRequest.comfyUrl" : "ui.comfyUrl"
    };
    const loraProjectByOutput = new Map(loraProjects.map((project) => [project.outputName, project]));
    const diagnostics = {
      route,
      restored: {
        restoredRequestApplies,
        hasRestoredGenerationSettings: Boolean(restoredGenerationSettings),
        restoredGenerationSettings,
        lastHistoryId
      },
      sourceMap,
      rawRequestBeforeSeedFix: rawRequest,
      finalComfyRequest: finalRequest,
      historySettingsToSave: historySettings,
      uiState: {
        japanesePrompt,
        prompt,
        negativePrompt,
        promptEnhanceMode,
        selectedCheckpoint,
        selectedVae,
        selectedClip,
        generationWidth,
        generationHeight,
        imageSizePreset,
        generationSteps,
        generationCfg,
        seedMode,
        seedValue,
        seedRange,
        comfyUrl,
        ollamaUrl,
        ollamaModel,
        generateTab,
        poseConditioning,
        modelAddMode,
        baseFilters
      },
      promptComposition: {
        basePositivePrompt: prompt,
        baseNegativePrompt: negativePrompt,
        effectivePositivePrompt: effectivePrompt,
        effectiveNegativePrompt,
        japanesePromptTags: japanesePromptTags.map((tag) => ({
          id: tag.id,
          name: tag.name,
          englishName: tag.englishName,
          insertedJapaneseDescription: tag.japaneseDescription,
          insertedEnglishDescription: tag.englishDescription
        })),
        positivePromptTags: positivePromptTags.map((tag) => ({
          id: tag.id,
          name: tag.name,
          englishName: tag.englishName,
          insertedEnglishDescription: tag.englishDescription
        })),
        positiveOptionTags: positiveOptionTags.map((tag) => ({
          id: tag.id,
          japanese: tag.japanese,
          english: tag.english,
          category: tag.category
        })),
        negativeOptionTags: negativeOptionTags.map((tag) => ({
          id: tag.id,
          japanese: tag.japanese,
          english: tag.english,
          category: tag.category
        })),
        positiveLoraTriggers
      },
      loraState: {
        selectedLoras,
        activeGenerationLoras,
        availableLoraNames: [...availableLoraNames],
        requestLoras: finalRequest.loras ?? [],
        selectedLoraProjects: (finalRequest.loras ?? []).map((lora) => {
          const project = loraProjectByOutput.get(lora.name);
          return {
            loraName: lora.name,
            projectId: project?.id,
            projectName: project?.name,
            outputName: project?.outputName,
            sourceLora: project?.sourceLora,
            triggerWords: getLoraTriggerWords(project),
            baseCheckpoint: project?.baseCheckpoint,
            status: project?.status
          };
        })
      },
      installedAssetsReferenceOnly: {
        note: "参照用のインストール済み一覧です。生成へ適用したLoRAは loraState.requestLoras のみです。",
        checkpoints: status?.checkpoints.map((file) => file.name) ?? [],
        availableLorasNotAppliedAutomatically: status?.loras.map((file) => file.name) ?? [],
        vae: status?.vae.map((file) => file.name) ?? [],
        clip: status?.clip.map((file) => file.name) ?? []
      }
    };
    appendLog("info", "=== GENERATION DIAGNOSTICS START ===");
    appendLog("info", generationDiagnosticsJson(diagnostics));
    appendLog("info", "=== GENERATION DIAGNOSTICS END ===");
  }

  const downloadRequest: DownloadRequest = {
    url: downloadUrl.trim(),
    targetType: "checkpoints",
    fileName: downloadFileName.trim() || undefined,
    sha256: downloadSha.trim() || undefined
  };

  const currentTestRequest = createGenerationRequest("character", seedMode === "fixed" ? seedValue : 0);
  const restoredRequestApplies = Boolean(
    restoredGenerationSettings?.generationRequest &&
    restoredGenerationSettings.prompt === prompt &&
    restoredGenerationSettings.negativePrompt === negativePrompt &&
    restoredGenerationSettings.checkpoint === selectedCheckpoint &&
    restoredGenerationSettings.vae === selectedVae &&
    restoredGenerationSettings.clip === selectedClip &&
    restoredGenerationSettings.width === generationWidth &&
    restoredGenerationSettings.height === generationHeight &&
    restoredGenerationSettings.steps === generationSteps &&
    restoredGenerationSettings.cfg === generationCfg &&
    restoredGenerationSettings.seedMode === seedMode &&
    restoredGenerationSettings.seedValue === seedValue &&
    restoredGenerationSettings.comfyUrl === comfyUrl &&
    sameGenerationLoras(restoredGenerationSettings.loras, selectedLoras) &&
    sameStringArray(restoredGenerationSettings.japanesePromptTagIds, japanesePromptTagIds) &&
    sameStringArray(restoredGenerationSettings.positivePromptTagIds, positivePromptTagIds) &&
    sameStringArray(restoredGenerationSettings.positiveOptionTagIds, positiveOptionTagIds) &&
    sameStringArray(restoredGenerationSettings.negativeOptionTagIds, negativeOptionTagIds) &&
    sameStringArray(restoredGenerationSettings.positiveLoraTriggerIds, positiveLoraTriggerIds) &&
    samePoseConditioning(restoredGenerationSettings.poseConditioning, poseConditioning)
  );
  const testRequest: TestGenerationRequest = restoredRequestApplies && restoredGenerationSettings?.generationRequest
    ? restoredGenerationSettings.generationRequest
    : currentTestRequest;

  function buildGenerationHistorySettings(request: TestGenerationRequest, scope: GenerationScope): PromptGenerationSettings {
    const background = scope === "background" ? sceneBackground : null;
    const historyScene = currentGenerationScene();
    if (scope === "background" && historyScene.background) {
      historyScene.background.seedMode = "fixed";
      historyScene.background.seedValue = request.seed;
    }
    if (scope === "character") {
      const historyCharacter = historyScene.characters.find((character) => character.id === activeSceneCharacter?.id);
      if (historyCharacter) {
        historyCharacter.settings.seedMode = "fixed";
        historyCharacter.settings.seedValue = request.seed;
      }
    }
    if (scope === "scene") {
      if (historyScene.background) {
        historyScene.background.seedMode = "fixed";
        historyScene.background.seedValue = request.seed;
      }
      historyScene.characters.forEach((character) => {
        character.settings.seedMode = "fixed";
        character.settings.seedValue = request.seed;
      });
    }
    return {
      generationRequest: request,
      generationScope: scope,
      targetCharacterId: scope === "character" ? activeSceneCharacter?.id : undefined,
      targetCharacterTitle: scope === "character" ? activeSceneCharacter?.title : undefined,
      japanesePrompt: background?.japanesePrompt ?? japanesePrompt,
      prompt: background?.prompt ?? prompt,
      effectivePrompt: request.prompt,
      negativePrompt: background?.negativePrompt ?? negativePrompt,
      effectiveNegativePrompt: request.negativePrompt,
      promptEnhanceMode: background?.promptEnhanceMode ?? promptEnhanceMode,
      checkpoint: selectedCheckpoint,
      vae: selectedVae,
      clip: selectedClip,
      loras: background ? [] : selectedLoras,
      japanesePromptTagIds: background?.japanesePromptTagIds ?? japanesePromptTagIds,
      positivePromptTagIds: background?.positivePromptTagIds ?? positivePromptTagIds,
      positiveOptionTagIds: background?.positiveOptionTagIds ?? positiveOptionTagIds,
      negativeOptionTagIds: background?.negativeOptionTagIds ?? negativeOptionTagIds,
      positiveLoraTriggerIds: background ? [] : positiveLoraTriggerIds,
      width: background?.width ?? generationWidth,
      height: background?.height ?? generationHeight,
      imageSizePreset: background?.imageSizePreset ?? imageSizePreset,
      steps: request.steps,
      cfg: request.cfg,
      seedMode: "fixed",
      seedValue: request.seed,
      seedRange: background?.seedRange ?? seedRange,
      requestSeed: request.seed,
      comfyUrl,
      ollamaUrl,
      ollamaModel,
      selectedModelAddMode: modelAddMode,
      baseFilters,
      generationTab: generateTab,
      poseConditioning: request.poseConditioning,
      scene: historyScene
    };
  }

  async function runLoraOnOffCompare(): Promise<void> {
    const compareSeed = nextGenerationSeed();
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
        <div className="topbar-brand">
          <div className="brand-mark">IM</div>
          <div>
            <h1>illustMK</h1>
            <p>AI Illustration Studio</p>
          </div>
        </div>
        <StatusGrid
          status={status}
          ollamaStatus={ollamaStatus}
          runtimeTab={runtimeTab}
          ollamaModel={ollamaModel}
          selectedCheckpoint={selectedCheckpointInfo}
          checkpointBaseModel={modelLibrary.find((entry) => entry.type === "checkpoint" && entry.active)?.baseModel || (selectedCheckpoint ? checkpointBaseModels[selectedCheckpoint] : "")}
          usageHistory={usageHistory}
        />
      </header>

      <nav className="main-tabs">
        <button className={mainTab === "environment" ? "tab active" : "tab"} onClick={() => setMainTab("environment")}>
          環境
        </button>
        <button className={mainTab === "models" ? "tab active" : "tab"} onClick={() => setMainTab("models")}>
          モデル
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
            <section className="panel data-path-panel">
              <PanelTitle icon={<FolderOpen size={18} />} title="PATH" />
              <div className="data-path-list">
                {dataPaths.map((entry) => (
                  <article
                    className={[
                      "data-path-card",
                      entry.active ? "active" : "",
                      selectedDataPath?.id === entry.id ? "selected" : ""
                    ].filter(Boolean).join(" ")}
                    key={entry.id}
                    onClick={() => setSelectedDataPathId(entry.id)}
                  >
                    <header>
                      <strong>{entry.path}</strong>
                      <div className="lora-project-meta">
                        <span className={entry.status === "インストール済み" ? "meta-tag ok" : entry.status === "エラー" ? "meta-tag error" : "meta-tag warn"}>
                          {entry.status}
                        </span>
                        {entry.active ? <span className="meta-tag active">ACTIVE</span> : null}
                      </div>
                    </header>
                    <p>{entry.memo || "メモなし"}</p>
                    <small>使用量: {entry.diskUsageLabel}</small>
                  </article>
                ))}
                {!dataPaths.length ? <p className="empty">DataPathが未登録です。</p> : null}
              </div>
              <div className="button-grid">
                <button disabled={action.busy} onClick={() => void addDataPath()}>
                  <Plus size={17} />
                  新規
                </button>
                <button disabled={action.busy || dataPaths.length <= 1 || !selectedDataPath} onClick={() => {
                  if (selectedDataPath) void deleteDataPathEntry(selectedDataPath.id);
                }}>
                  <Trash2 size={17} />
                  削除
                </button>
                <button disabled={action.busy} onClick={() => void importDataPath()}>
                  <Download size={17} />
                  インポート
                </button>
                <button
                  disabled={action.busy || !selectedDataPath || selectedDataPath.active}
                  onClick={() => {
                    if (selectedDataPath) void switchActiveDataPath(selectedDataPath.id);
                  }}
                >
                  <RefreshCw size={17} />
                  切替
                </button>
                <button
                  disabled={action.busy || !selectedDataPath}
                  onClick={() => {
                    if (selectedDataPath) openDataPathMemoEditor(selectedDataPath);
                  }}
                >
                  <Pencil size={17} />
                  編集
                </button>
                <button disabled={action.busy} onClick={() => void reloadDataPaths()}>
                  <RefreshCw size={17} />
                  更新
                </button>
              </div>
            </section>

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
                <div className="button-row compact-actions">
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
              <PanelTitle icon={<Download size={18} />} title="VAE" />
              <div className="asset-catalog-list">
                {recommendedVaeModels.map((item) => (
                  <article className="llm-catalog-card" key={item.fileName}>
                    <div className="llm-select-button">
                      <span>{item.name}</span>
                      <small>{item.source}</small>
                    </div>
                    <p>{item.note}</p>
                    <button disabled={action.busy} onClick={() => void installModelAsset("vae", item.url, item.fileName, item.fileName)}>
                      <Download size={15} />
                      Download
                    </button>
                  </article>
                ))}
              </div>
              <div className="field">
                <label>VAE URL</label>
                <div className="inline">
                  <input value={vaeUrl} onChange={(event) => setVaeUrl(event.target.value)} />
                  <button className="icon-button" disabled={action.busy || !vaeUrl.trim()} onClick={() => void installVae()} title="VAEインストール">
                    <Download size={17} />
                  </button>
                </div>
              </div>
              <div className="model-list">
                <div className="model-list-header">
                  <h2>インストール済みVAE</h2>
                  <button disabled={action.busy} onClick={() => void importModelAsset("vae")}>
                    <FolderOpen size={15} />
                    インポート
                  </button>
                </div>
                {status?.vae.length ? (
                  status.vae.map((file) => (
                    <div className="model-row" key={file.name}>
                      <button className={selectedVae === file.name ? "model selected" : "model"} onClick={() => setSelectedVae(file.name)}>
                        <span>{file.name}</span>
                        <small>{file.sizeLabel}</small>
                      </button>
                      <button className="icon-button danger" disabled={action.busy} onClick={() => void deleteModelAsset("vae", file.name)} title="削除">
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))
                ) : (
                  <p className="empty">まだVAEがありません。</p>
                )}
              </div>
            </section>

            <section className="panel model-runtime-panel">
              <PanelTitle icon={<Download size={18} />} title="CLIP" />
              <div className="asset-catalog-list">
                {recommendedClipModels.map((item) => (
                  <article className="llm-catalog-card" key={item.fileName}>
                    <div className="llm-select-button">
                      <span>{item.name}</span>
                      <small>{item.source}</small>
                    </div>
                    <p>{item.note}</p>
                    <button disabled={action.busy} onClick={() => void installModelAsset("clip", item.url, item.fileName, item.fileName)}>
                      <Download size={15} />
                      Download
                    </button>
                  </article>
                ))}
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
              <div className="model-list">
                <div className="model-list-header">
                  <h2>インストール済みCLIP</h2>
                  <button disabled={action.busy} onClick={() => void importModelAsset("clip")}>
                    <FolderOpen size={15} />
                    インポート
                  </button>
                </div>
                {status?.clip.length ? (
                  status.clip.map((file) => (
                    <div className="model-row" key={file.name}>
                      <button className={selectedClip === file.name ? "model selected" : "model"} onClick={() => setSelectedClip(file.name)}>
                        <span>{file.name}</span>
                        <small>{file.sizeLabel}</small>
                      </button>
                      <button className="icon-button danger" disabled={action.busy} onClick={() => void deleteModelAsset("clip", file.name)} title="削除">
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))
                ) : (
                  <p className="empty">まだCLIPがありません。</p>
                )}
              </div>
            </section>
          </div>

          <section className="panel llm-panel">
            <PanelTitle icon={<Bot size={18} />} title="LLM / Ollama" />
            <div className="field">
              <label>LLM API URL</label>
              <div className="inline">
                <input
                  placeholder="http://127.0.0.1:11434"
                  value={ollamaUrl}
                  onChange={(event) => setOllamaUrl(event.target.value)}
                  onBlur={() => void refreshOllamaStatus()}
                />
                <button className="icon-button" onClick={() => void refreshOllamaStatus()} title="接続確認">
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
            <div className="button-row compact-actions">
              <button onClick={() => void runStreamOperation("ollama-install", "Ollama環境をインストール中")}>
                <Download size={17} />
                Ollamaインストール
              </button>
              <button onClick={() => void runAction("Ollamaを起動中", () => window.illustMK.ollama.start(), () => void refreshOllamaStatus())}>
                <Play size={17} />
                起動
              </button>
              <button onClick={() => void runAction("Ollamaを停止中", () => window.illustMK.ollama.stop(), () => void refreshOllamaStatus())}>
                <Square size={17} />
                停止
              </button>
            </div>
            <div className="field">
              <label>LLMダウンロード</label>
              <div className="inline">
                <input placeholder="qwen2.5:7b-instruct / hf.co/..." value={llmPullName} onChange={(event) => setLlmPullName(event.target.value)} />
                <button className="icon-button" onClick={() => void runOllamaPull(llmPullName)} disabled={!canDownloadLlm} title="LLMダウンロード">
                  <Download size={17} />
                </button>
              </div>
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
                    <div className="llm-search-select">
                      <strong>{model.id}</strong>
                      <span>
                        HF GGUF / DL {model.downloads.toLocaleString()} / {model.totalSizeLabel}
                      </span>
                    </div>
                    <p>{model.ggufFiles.slice(0, 4).map((file) => `${file.name} (${file.sizeLabel})`).join(", ") || "GGUF repo"}</p>
                    <button
                      className="llm-download-button"
                      disabled={!ollamaStatus?.reachable || !ollamaStatus.dataLinked || action.busy}
                      onClick={() => {
                        setLlmPullName(model.pullName);
                        void runOllamaPull(model.pullName);
                      }}
                    >
                      <Download size={15} />
                      Download
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
                  <div className="llm-select-button">
                    <span>{model.name}</span>
                    <small>{model.source} / {model.size}</small>
                  </div>
                  <p>{model.note}</p>
                  <button
                    disabled={!ollamaStatus?.reachable || !ollamaStatus.dataLinked || action.busy}
                    onClick={() => {
                      setLlmPullName(model.name);
                      void runOllamaPull(model.name);
                    }}
                  >
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
                    <button
                      className={ollamaModel === model.name ? "model selected" : "model"}
                      onClick={() => {
                        setOllamaModel(model.name);
                        setLlmPullName(model.name);
                      }}
                    >
                      <span>{model.name}</span>
                      {ollamaModel === model.name ? <em>ACTIVE</em> : null}
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
              <button className={modelAddMode === "catalog" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelAddMode("catalog")}>
                Civitaiカタログ
              </button>
              <button className={modelAddMode === "url" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelAddMode("url")}>
                URL直接
              </button>
            </div>

            {modelAddMode === "catalog" ? (
              <>
                <div className="catalog-control-grid">
                  <div className="field">
                    <label>期間</label>
                    <select value={catalogPeriod} onChange={(event) => setCatalogPeriod(event.target.value as "AllTime" | "Year" | "Month" | "Week" | "Day")}>
                      <option value="Day">今日</option>
                      <option value="Week">直近1週間</option>
                      <option value="Month">直近1ヶ月</option>
                      <option value="Year">直近1年</option>
                      <option value="AllTime">累計</option>
                    </select>
                  </div>
                  <div className="field">
                    <label>条件</label>
                    <select value={catalogSort} onChange={(event) => setCatalogSort(event.target.value as "Most Downloaded" | "Highest Rated" | "Newest")}>
                      <option value="Most Downloaded">人気順</option>
                      <option value="Highest Rated">評価順</option>
                      <option value="Newest">新着順</option>
                    </select>
                  </div>
                  <button className="primary catalog-action" disabled={action.busy} onClick={() => void loadCivitaiModels()}>
                    <Download size={17} />
                    人気Top10を表示
                  </button>
                </div>
                <div className="field">
                  <label>検索キーワード</label>
                  <div className="inline">
                    <input placeholder="anime, illustrious..." value={catalogQuery} onChange={(event) => setCatalogQuery(event.target.value)} />
                    <button className="icon-button" disabled={action.busy} onClick={() => void loadCivitaiModels(catalogQuery)} title="検索">
                      <RefreshCw size={17} />
                    </button>
                  </div>
                </div>
                <div className="filter-row">
                  <label>BASE MODEL FILTER</label>
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
                <div className="catalog-list">
                  {catalogModels.map((model) => (
                    <article className="catalog-card" key={`${model.id}-${model.versionId}`}>
                      {model.imageUrl ? <img src={model.imageUrl} alt="" /> : <div className="catalog-no-image">No Image</div>}
                      <div>
                        <h3>{model.name}</h3>
                        <p>{model.creator} / {model.baseModel || "base不明"}</p>
                        <p>{Math.round(model.sizeKB / 1024)} MB / DL {model.downloadCount.toLocaleString()}</p>
                        <div className="catalog-card-actions">
                          <button disabled={action.busy} onClick={() => void downloadCivitaiModel(model)}>
                            <Download size={15} />
                            Download
                          </button>
                          <button disabled={!model.pageUrl} onClick={() => void window.illustMK.app.openExternal(model.pageUrl)} title="Civitaiページを開く">
                            <ExternalLink size={15} />
                            Web
                          </button>
                        </div>
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
                status.checkpoints.map((file) => {
                  const external = checkpointExternalSelections[file.name] ?? { vae: "", clip: "" };
                  const requiredComponents = checkpointRequiredComponents[file.name] ?? [];
                  const needsVae = !file.hasVae;
                  const needsClip = !file.hasClip;
                  const suggestedVae = needsVae ? status.vae.find((item) => item.hasVae)?.name ?? status.vae[0]?.name ?? "" : "";
                  const suggestedClip = needsClip ? status.clip.find((item) => item.hasClip)?.name ?? status.clip[0]?.name ?? "" : "";
                  const ready = (!needsVae || external.vae) && (!needsClip || external.clip);
                  return (
                    <div className={needsVae || needsClip ? "model-row checkpoint-row expanded" : "model-row checkpoint-row"} key={file.name}>
                      <button
                        className={selectedCheckpoint === file.name ? "model selected" : "model"}
                        onClick={() => setSelectedCheckpoint(file.name)}
                      >
                        <span>{file.name}</span>
                        <small>{file.sizeLabel} / {needsVae || needsClip ? (ready ? "外部設定OK" : "追加ファイル必要") : file.compatibilityLabel}</small>
                      </button>
                      <button
                        className="icon-button danger"
                        disabled={action.busy}
                        onClick={() => void deleteCheckpoint(file.name)}
                        title="削除"
                      >
                        <Trash2 size={16} />
                      </button>
                      {needsVae || needsClip ? (
                        <div className="checkpoint-external-settings">
                          <div className="field">
                            <label>VAE</label>
                            <select
                              value={external.vae}
                              disabled={!needsVae}
                              onChange={(event) => saveCheckpointExternalSelection(file.name, { ...external, vae: event.target.value })}
                            >
                              <option value="">{needsVae ? "未選択" : "内蔵"}</option>
                              {status.vae.map((item) => (
                                <option key={item.name} value={item.name}>
                                  {item.name}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div className="field">
                            <label>CLIP</label>
                            <select
                              value={external.clip}
                              disabled={!needsClip}
                              onChange={(event) => saveCheckpointExternalSelection(file.name, { ...external, clip: event.target.value })}
                            >
                              <option value="">{needsClip ? "未選択" : "内蔵"}</option>
                              {status.clip.map((item) => (
                                <option key={item.name} value={item.name}>
                                  {item.name}
                                </option>
                              ))}
                            </select>
                          </div>
                          <button
                            disabled={action.busy || (!suggestedVae && needsVae) || (!suggestedClip && needsClip)}
                            onClick={() => saveCheckpointExternalSelection(file.name, { vae: needsVae ? suggestedVae : "", clip: needsClip ? suggestedClip : "" })}
                          >
                            推奨を設定
                          </button>
                          {requiredComponents.length ? (
                            <div className="required-component-list">
                              <strong>Required Components</strong>
                              {requiredComponents.map((component) => {
                                const targetType = componentTargetType(component);
                                const installed = isComponentInstalled(component);
                                const selected =
                                  (targetType === "vae" && external.vae === component.name) ||
                                  (targetType === "clip" && external.clip === component.name);
                                return (
                                  <button
                                    key={`${component.type}-${component.name}`}
                                    disabled={action.busy}
                                    className={selected ? "component-option selected" : "component-option"}
                                    onClick={() => {
                                      if (installed && targetType) {
                                        saveCheckpointExternalSelection(file.name, { ...external, [targetType]: component.name });
                                        return;
                                      }
                                      void installRequiredComponent(file.name, component);
                                    }}
                                  >
                                    <span>{component.name}</span>
                                    <small>{component.type} / {Math.round(component.sizeKB / 1024)} MB / {installed ? "installed" : "download"}</small>
                                  </button>
                                );
                              })}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  );
                })
              ) : (
                <p className="empty">まだ checkpoint がありません。</p>
              )}
            </div>
          </section>
        </section>
      ) : mainTab === "models" ? (
        <section className="model-library-layout">
          <section className="panel model-library-sidebar">
            <PanelTitle icon={<Boxes size={18} />} title="モデル" />
            <div className="model-library-filters">
              {(["checkpoint", "controlnet", "vae", "clip", "other"] as ModelAssetType[]).map((type) => (
                <button
                  key={type}
                  className={modelTypeFilters.includes(type) ? "filter-chip active" : "filter-chip"}
                  onClick={() => setModelTypeFilters((current) => current.includes(type) ? current.filter((item) => item !== type) : [...current, type])}
                >
                  {modelTypeLabel(type)}
                </button>
              ))}
              <button className="filter-chip" onClick={() => setModelTypeFilters([])}>リセット</button>
            </div>
            <div className="model-library-list">
              {filteredModelLibrary.length ? filteredModelLibrary.map((entry) => (
                <button
                  type="button"
                  key={entry.id}
                  className={["model-library-card", entry.id === selectedModelId ? "selected" : "", entry.active ? "active-model" : ""].filter(Boolean).join(" ")}
                  onClick={() => setSelectedModelId(entry.id)}
                >
                  <span className="model-library-thumb">
                    {entry.thumbnailDataUrl ? <img src={entry.thumbnailDataUrl} alt="" /> : <ImageIcon size={22} />}
                  </span>
                  <span className="model-library-card-body">
                    <strong>{entry.name}</strong>
                    <span className="model-library-tags">
                      <em>{modelTypeLabel(entry.type)}</em>
                      {entry.baseModel ? <em>{entry.baseModel}</em> : null}
                      {entry.usage ? <em>{entry.usage}</em> : null}
                      <em>{entry.format}</em>
                      <em>{entry.sizeLabel}</em>
                    </span>
                  </span>
                  {entry.active ? <b>ACTIVE</b> : null}
                </button>
              )) : <p className="empty">このDataPathにはモデルがありません。</p>}
            </div>
            <div className="model-library-actions">
              <button onClick={() => setModelImportOpen(true)}><Plus size={16} />インポート</button>
              <button onClick={() => setModelCatalogOpen(true)}><Download size={16} />カタログ</button>
              <button onClick={() => void loadModelLibrary()}><RefreshCw size={16} />更新</button>
              <button className="danger" disabled={!selectedModel || selectedModel.active} onClick={() => void deleteSelectedModelLibraryEntry()}><Trash2 size={16} />削除</button>
            </div>
          </section>
          <section className="panel model-library-detail">
            <PanelTitle icon={<Boxes size={18} />} title={selectedModel ? modelTypeLabel(selectedModel.type) : "モデル詳細"} />
            {selectedModel ? (
              <div className="model-detail-content">
                <div className="model-detail-hero">
                  <div className="model-detail-image">
                    {selectedModel.thumbnailDataUrl ? <img src={selectedModel.thumbnailDataUrl} alt="" /> : <ImageIcon size={46} />}
                  </div>
                  <div className="model-detail-fields">
                    <div className="field">
                      <label>タイトル</label>
                      <input value={selectedModel.name} onChange={(event) => updateModelDraft({ name: event.target.value })} />
                    </div>
                    <div className="two-columns">
                      <div className="field">
                        <label>BASE MODEL</label>
                        <input value={selectedModel.baseModel} onChange={(event) => updateModelDraft({ baseModel: event.target.value })} placeholder="Illustrious / SDXL / Anima..." />
                      </div>
                      <div className="field">
                        <label>用途</label>
                        <input value={selectedModel.usage} onChange={(event) => updateModelDraft({ usage: event.target.value })} placeholder="OpenPose / Depth..." />
                      </div>
                    </div>
                    <div className="model-file-summary">
                      <span>{selectedModel.fileName}</span>
                      <span>{selectedModel.format}</span>
                      <span>{selectedModel.sizeLabel}</span>
                    </div>
                  </div>
                </div>
                {selectedModel.type === "checkpoint" ? (
                  <>
                    <div className="checkpoint-linked-assets">
                      <div className="field">
                        <label>VAE</label>
                        <select value={selectedModel.selectedVae ?? ""} onChange={(event) => updateModelDraft({ selectedVae: event.target.value })}>
                          <option value="">内蔵 / 未指定</option>
                          {modelLibrary.filter((item) => item.type === "vae").map((item) => <option value={item.fileName} key={item.id}>{item.name}</option>)}
                        </select>
                      </div>
                      <div className="field">
                        <label>CLIP</label>
                        <select value={selectedModel.selectedClip ?? ""} onChange={(event) => updateModelDraft({ selectedClip: event.target.value })}>
                          <option value="">内蔵 / 未指定</option>
                          {modelLibrary.filter((item) => item.type === "clip").map((item) => <option value={item.fileName} key={item.id}>{item.name}</option>)}
                        </select>
                      </div>
                    </div>
                    {selectedModel.requiredComponents?.length ? (
                      <div className="model-required-components">
                        <strong>推奨コンポーネント</strong>
                        {selectedModel.requiredComponents.map((component) => {
                          const targetType = component.type.toLowerCase() === "vae" ? "vae" : component.type.toLowerCase() === "text encoder" ? "clip" : null;
                          const installed = targetType ? modelLibrary.some((entry) => entry.type === targetType && entry.fileName === component.name) : false;
                          return (
                            <div className="model-required-row" key={`${component.type}-${component.name}`}>
                              <span>{component.name}<small>{component.type} / {Math.round(component.sizeKB / 1024)} MB</small></span>
                              <button disabled={action.busy || installed || !targetType} onClick={() => void installModelLibraryRequiredComponent(selectedModel, component)}>{installed ? "導入済み" : "推奨ダウンロード"}</button>
                            </div>
                          );
                        })}
                      </div>
                    ) : null}
                  </>
                ) : null}
                <div className="model-detail-actions">
                  {selectedModel.sourcePageUrl ? (
                    <button onClick={() => void window.illustMK.app.openExternal(selectedModel.sourcePageUrl!)}><ExternalLink size={16} />Webページ</button>
                  ) : null}
                  {selectedModel.type === "checkpoint" ? (
                    <button className={selectedModel.active ? "active" : ""} disabled={selectedModel.active} onClick={() => void activateSelectedModel()}>
                      <CheckCircle2 size={16} />{selectedModel.active ? "使用中" : "このCheckpointを使用"}
                    </button>
                  ) : null}
                  <button className="primary" onClick={() => void saveSelectedModel()}><Save size={16} />更新</button>
                </div>
              </div>
            ) : (
              <div className="placeholder compact"><Boxes size={40} /><p>左の一覧からモデルを選択してください。</p></div>
            )}
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
            <div className="base-filter-row lora-base-filter-row">
              {loraProjectBaseOptions.map((filter) => (
                <button
                  className={loraProjectBaseFilters.includes(filter) ? "active" : ""}
                  key={filter}
                  onClick={() =>
                    setLoraProjectBaseFilters((current) =>
                      current.includes(filter) ? current.filter((item) => item !== filter) : [...current, filter]
                    )
                  }
                >
                  {filter}
                </button>
              ))}
              <button className="tiny-button" disabled={!loraProjectBaseFilters.length} onClick={() => setLoraProjectBaseFilters([])}>
                リセット
              </button>
            </div>
            <div className="lora-project-list">
              {filteredLoraProjects.length ? (
                filteredLoraProjects.map((project) => (
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
                    <div className="lora-project-thumb">
                      <ImageIcon className="lora-project-placeholder-icon" size={22} />
                      {project.thumbnailDataUrl || project.thumbnailPath ? (
                        <img src={project.thumbnailDataUrl || imageUrl(project.thumbnailPath ?? "")} alt="" onError={(event) => { event.currentTarget.remove(); }} />
                      ) : null}
                    </div>
                    <div className="lora-project-body">
                      <header>
                        <strong>{project.name}</strong>
                        <span>{project.status}</span>
                      </header>
                      <div className="lora-project-meta">
                        {getLoraTriggerWords(project).map((word) => <span className="meta-tag trigger" key={word}>{word}</span>)}
                        <span className="meta-tag base">{inferCheckpointBaseModel(project.baseCheckpoint)}</span>
                        {status?.loras.find((file) => file.name === project.outputName)?.sizeLabel ? (
                          <span className="meta-tag size">{status.loras.find((file) => file.name === project.outputName)?.sizeLabel}</span>
                        ) : null}
                      </div>
                      <p>{project.message}</p>
                    </div>
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
              <PanelTitle icon={<TerminalSquare size={18} />} title="LoRA" />
            </div>
            {selectedLoraProject ? (
              <div className="lora-detail-panel">
                <div
                  className="lora-detail-thumb"
                  onDragOver={allowLoraThumbnailDrop}
                  onDrop={(event) => void updateSelectedLoraThumbnail(event)}
                >
                  <ImageIcon className="lora-project-placeholder-icon" size={34} />
                  {selectedLoraProject.thumbnailDataUrl || selectedLoraProject.thumbnailPath ? (
                    <img
                      src={selectedLoraProject.thumbnailDataUrl || imageUrl(selectedLoraProject.thumbnailPath ?? "")}
                      alt=""
                      onError={(event) => {
                        event.currentTarget.remove();
                      }}
                    />
                  ) : null}
                  <span>画像をD&D</span>
                </div>
                <div className="lora-detail-fields">
                  <div className="field">
                    <label>タイトル</label>
                    <input value={loraDetailName} onChange={(event) => setLoraDetailName(event.target.value)} />
                  </div>
                  <div className="field">
                    <label>トリガーワード</label>
                    <div className="trigger-word-list">
                      {loraDetailTriggerWords.map((word, index) => (
                        <div className="trigger-word-row" key={index}>
                          <input value={word} onChange={(event) => updateTriggerWordList("detail", index, event.target.value)} />
                          <button className="icon-button" type="button" onClick={() => addTriggerWordRow("detail")} title="追加">
                            <Plus size={15} />
                          </button>
                          <button className="icon-button" type="button" disabled={loraDetailTriggerWords.length <= 1} onClick={() => removeTriggerWordRow("detail", index)} title="削除">
                            <Trash2 size={15} />
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="lora-detail-actions">
                    <button
                      disabled={!selectedLoraProject.sourcePageUrl}
                      onClick={() => selectedLoraProject.sourcePageUrl && void window.illustMK.app.openExternal(selectedLoraProject.sourcePageUrl)}
                    >
                      <ExternalLink size={17} />
                      Webページ
                    </button>
                    <button className="primary" disabled={action.busy || !loraDetailName.trim()} onClick={() => void saveSelectedLoraDetails()}>
                      <Save size={17} />
                      更新
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <p className="empty">LoRAプロジェクトを選択してください。</p>
            )}
            <div className="lora-control-panel">
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
            <div className="image-memo-panel">
              <textarea value={imageMemo} onChange={(event) => setImageMemo(event.target.value)} placeholder="この画像のメモ" />
              <button className="tiny-button" disabled={!lastHistoryId || action.busy} onClick={() => void saveCurrentHistoryMemo()}>
                <Save size={14} />
                更新
              </button>
            </div>
            <div className="preview-generation-actions">
              <button className="generate" disabled={!canGenerateScene} onClick={() => void runSingleGenerate("scene")}>
                <WandSparkles size={18} />
                総合生成
              </button>
              <button className="generate secondary-generate" disabled={!canGenerateScene || repeatRunning} onClick={() => void runRepeatGenerate("scene")}>
                <RefreshCw size={18} />
                総合リピート生成
              </button>
            </div>
          </section>

          <section className="panel generate-panel">
            {sceneCharacters.length || sceneBackground ? (
              <div className="scene-item-tabs">
                {sceneCharacters.map((character) => (
                  <button
                    key={character.id}
                    className={activeSceneItemId === character.id ? "scene-item-tab active" : "scene-item-tab"}
                    onClick={() => activateSceneCharacter(character.id)}
                  >
                    {character.title}
                  </button>
                ))}
                {sceneBackground ? (
                  <button
                    className={activeSceneItemId === sceneBackground.id ? "scene-item-tab active" : "scene-item-tab"}
                    onClick={() => {
                      setSceneCharacters(persistActiveSceneCharacter());
                      setActiveSceneItemId(sceneBackground.id);
                    }}
                  >
                    {sceneBackground.title}
                  </button>
                ) : null}
                <button className="scene-add-tab" onClick={() => setSceneAddOpen(true)} title="シーン項目を追加">
                  <Plus size={18} />
                </button>
              </div>
            ) : null}
            {!sceneCharacters.length && !sceneBackground ? (
              <div className="scene-empty-state">
                <button onClick={() => setSceneAddOpen(true)} title="キャラまたは背景を追加">
                  <Plus size={44} />
                </button>
              </div>
            ) : sceneBackground && activeSceneItemId === sceneBackground.id ? (
              <div className="scene-background-editor">
                <div className="scene-editor-heading">
                  <div>
                    <strong>背景設定</strong>
                    <span>背景単体の復元時は、キャラの配置情報を初期位置へ戻します。</span>
                  </div>
                  <button className="icon-button danger" onClick={removeActiveSceneItem} title="背景をシーンから削除">
                    <Trash2 size={16} />
                  </button>
                </div>
                <div className="field">
                  <label>タイトル</label>
                  <input value={sceneBackground.title} onChange={(event) => updateSceneBackground({ title: event.target.value })} />
                </div>
                <div className="field">
                  <div className="prompt-field-header">
                    <label>Japanese Prompt</label>
                    <button onClick={() => setPromptHistoryOpen(true)}><RefreshCw size={15} />履歴</button>
                    <button onClick={() => setTagPromptTarget("background-japanese")}><WandSparkles size={15} />背景タグ</button>
                  </div>
                  <textarea rows={6} value={sceneBackground.japanesePrompt} onChange={(event) => updateSceneBackground({ japanesePrompt: event.target.value })} />
                  <AppliedIllustrationTags
                    target="background-japanese"
                    tags={backgroundJapanesePromptTags}
                    onRemove={(_, tag) => updateSceneBackground({ japanesePromptTagIds: sceneBackground.japanesePromptTagIds.filter((id) => id !== tag.id) })}
                  />
                </div>
                <div className="field">
                  <label>変換モード</label>
                  <select value={sceneBackground.promptEnhanceMode} onChange={(event) => updateSceneBackground({ promptEnhanceMode: event.target.value as PromptEnhanceMode })}>
                    <option value="balanced-detail">おすすめ: 本文 + 脚注/定義を反映</option>
                    <option value="footnote-strict">脚注重視: 定義文を強制採用</option>
                    <option value="main-subject">本文重視: 主文を優先</option>
                    <option value="tag-dense">タグ密度重視: 細かいタグへ分解</option>
                  </select>
                </div>
                <button className="primary" disabled={!canEnhanceBackgroundPrompt} title={enhanceBackgroundPromptDisabledReason} onClick={() => void enhanceBackgroundPrompt()}>
                  <Bot size={17} />日本語から背景Positive / Negativeへ変換
                </button>
                <p className={canEnhanceBackgroundPrompt ? "result-message ready" : "result-message"}>{enhanceBackgroundPromptDisabledReason}</p>
                {sceneBackground.promptNotes ? <p className="result-message">{sceneBackground.promptNotes}</p> : null}
                <div className="field">
                  <div className="prompt-field-header">
                    <label>Positive Prompt</label>
                    <button onClick={() => setTagPromptTarget("background-positive")}><WandSparkles size={15} />背景タグ</button>
                    <button onClick={() => setOptionTagDialogTarget("background-positive")}><WandSparkles size={15} />オプションタグ</button>
                  </div>
                  <textarea rows={6} value={sceneBackground.prompt} onChange={(event) => updateSceneBackground({ prompt: event.target.value })} />
                  <AppliedIllustrationTags
                    target="background-positive"
                    tags={backgroundPositivePromptTags}
                    onRemove={(_, tag) => updateSceneBackground({ positivePromptTagIds: sceneBackground.positivePromptTagIds.filter((id) => id !== tag.id) })}
                  />
                  <AppliedOptionTags target="background-positive" tags={backgroundPositiveOptionTags} onRemove={removeOptionPromptTag} />
                </div>
                <div className="field">
                  <div className="prompt-field-header">
                    <label>Negative Prompt</label>
                    <button onClick={() => setOptionTagDialogTarget("background-negative")}><WandSparkles size={15} />オプションタグ</button>
                  </div>
                  <textarea rows={4} value={sceneBackground.negativePrompt} onChange={(event) => updateSceneBackground({ negativePrompt: event.target.value })} />
                  <AppliedOptionTags target="background-negative" tags={backgroundNegativeOptionTags} onRemove={removeOptionPromptTag} />
                </div>
                <section className="background-generation-options">
                  <strong>生成オプション</strong>
                  <div className="field">
                    <label>画像サイズ</label>
                    <select value={sceneBackground.imageSizePreset} onChange={(event) => updateBackgroundImageSizePreset(event.target.value)}>
                      <option value="1024x1024">正方形 1024 x 1024</option>
                      <option value="768x1024">縦 768 x 1024</option>
                      <option value="1024x768">横 1024 x 768</option>
                      <option value="custom">カスタム</option>
                    </select>
                  </div>
                  <div className="background-options-grid">
                    <div className="field"><label>Width</label><input type="number" min="64" step="64" value={sceneBackground.width} onChange={(event) => updateSceneBackground({ width: Number(event.target.value), imageSizePreset: "custom" })} /></div>
                    <div className="field"><label>Height</label><input type="number" min="64" step="64" value={sceneBackground.height} onChange={(event) => updateSceneBackground({ height: Number(event.target.value), imageSizePreset: "custom" })} /></div>
                    <div className="field"><label>Steps</label><input type="number" min="1" max="150" value={sceneBackground.steps} onChange={(event) => updateSceneBackground({ steps: Number(event.target.value) })} /></div>
                    <div className="field"><label>CFG</label><input type="number" min="1" max="30" step="0.5" value={sceneBackground.cfg} onChange={(event) => updateSceneBackground({ cfg: Number(event.target.value) })} /></div>
                  </div>
                  <div className="background-options-grid seed-options">
                    <div className="field"><label>Seed</label><select value={sceneBackground.seedMode} onChange={(event) => updateSceneBackground({ seedMode: event.target.value as GenerationSeedMode })}><option value="random">Random</option><option value="fixed">Fixed</option><option value="sequence">連番</option></select></div>
                    <div className="field">
                      <label>Seed Value</label>
                      {sceneBackground.seedMode === "sequence" ? (
                        <input value={sceneBackground.seedRange} placeholder={defaultSeedRange} onChange={(event) => updateSceneBackground({ seedRange: event.target.value })} />
                      ) : (
                        <input type="number" disabled={sceneBackground.seedMode === "random"} value={sceneBackground.seedValue} onChange={(event) => updateSceneBackground({ seedValue: Number(event.target.value) })} />
                      )}
                    </div>
                  </div>
                </section>
                <div className="scene-reference-grid">
                  <div className="scene-reference-preview">
                    {sceneBackground.referenceImageDataUrl ? (
                      <img src={sceneBackground.referenceImageDataUrl} alt="背景参照画像" />
                    ) : (
                      <div className="placeholder"><ImageIcon size={34} /><span>参照画像なし</span></div>
                    )}
                  </div>
                  <div className="scene-reference-controls">
                    <label className="file-button">
                      <ImageIcon size={16} />
                      サンプル画像を読み込む
                      <input type="file" accept="image/*" onChange={(event) => importBackgroundReference(event.target.files?.[0])} />
                    </label>
                    <button disabled={!sceneBackground.referenceImageDataUrl} onClick={() => updateSceneBackground({ referenceImageDataUrl: "" })}>
                      <Trash2 size={15} />
                      画像を解除
                    </button>
                    <div className="field">
                      <label>参照強度 {sceneBackground.referenceStrength.toFixed(2)}</label>
                      <input type="range" min="0" max="1" step="0.05" value={sceneBackground.referenceStrength} onChange={(event) => updateSceneBackground({ referenceStrength: Number(event.target.value) })} />
                    </div>
                  </div>
                </div>
                <div className="button-row">
                  <button onClick={() => setBackgroundLibraryOpen(true)}><RefreshCw size={16} />保存済み背景から復元</button>
                  <button className="primary" onClick={saveBackgroundPreset}><Save size={16} />背景を保存</button>
                </div>
                <div className="background-generation-actions">
                  <button className="generate" disabled={!canGenerateBackground} onClick={() => void runSingleGenerate("background")}><WandSparkles size={18} />背景生成</button>
                  <button className="generate secondary-generate" disabled={!canGenerateBackground || repeatRunning} onClick={() => void runRepeatGenerate("background")}><RefreshCw size={18} />背景リピート生成</button>
                </div>
              </div>
            ) : activeSceneCharacter ? (
              <>
            <div className="runtime-tabs">
              <button className={generateTab === "character" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("character")}>
                キャラ設定
              </button>
              <button className={generateTab === "prompt" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("prompt")}>
                プロンプト
              </button>
              <button className={generateTab === "lora" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("lora")}>
                LoRA
                {selectedLoras.length ? <span>{selectedLoras.length}</span> : null}
              </button>
              <button className={generateTab === "pose" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("pose")}>
                デッサン
                {poseConditioning.enabled ? <span>ON</span> : null}
              </button>
              <button className={generateTab === "options" ? "runtime-tab active" : "runtime-tab"} onClick={() => setGenerateTab("options")}>
                生成オプション
              </button>
            </div>
            {generateTab === "character" ? (
              <div className="scene-character-settings">
                <div className="scene-editor-heading">
                  <div>
                    <strong>{activeSceneCharacter.title}</strong>
                    <span>キャラ単体を復元した場合、配置は中央の初期値になります。</span>
                  </div>
                  <button className="icon-button danger" onClick={removeActiveSceneItem} title="キャラをシーンから削除">
                    <Trash2 size={16} />
                  </button>
                </div>
                <div className="field">
                  <label>キャラ名</label>
                  <input value={activeSceneCharacter.title} onChange={(event) => updateActiveCharacterTitle(event.target.value)} />
                </div>
                <div className="scene-placement-grid">
                  <div className="field"><label>横位置 (%)</label><input type="number" min="0" max="100" value={activeSceneCharacter.placement.x} onChange={(event) => updateActiveCharacterPlacement({ x: Number(event.target.value) })} /></div>
                  <div className="field"><label>縦位置 (%)</label><input type="number" min="0" max="100" value={activeSceneCharacter.placement.y} onChange={(event) => updateActiveCharacterPlacement({ y: Number(event.target.value) })} /></div>
                  <div className="field"><label>拡大率 (%)</label><input type="number" min="10" max="300" value={activeSceneCharacter.placement.scale} onChange={(event) => updateActiveCharacterPlacement({ scale: Number(event.target.value) })} /></div>
                  <div className="field"><label>重なり順</label><input type="number" min="1" value={activeSceneCharacter.placement.zIndex} onChange={(event) => updateActiveCharacterPlacement({ zIndex: Number(event.target.value) })} /></div>
                </div>
                <div className="button-row">
                  <button onClick={() => setCharacterLibraryOpen(true)}><RefreshCw size={16} />保存済みキャラから復元</button>
                  <button className="primary" onClick={saveActiveCharacterPreset}><Save size={16} />キャラ設定を保存</button>
                </div>
              </div>
            ) : generateTab === "prompt" ? (
              <>
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
                optionPromptTags={optionPromptTags}
                  japanesePromptTags={japanesePromptTags}
                  positivePromptTags={positivePromptTags}
                  positiveOptionTags={positiveOptionTags}
                  negativeOptionTags={negativeOptionTags}
                  positiveLoraTriggers={positiveLoraTriggers}
                canEnhancePrompt={canEnhancePrompt}
                disabledReason={enhancePromptDisabledReason}
                onEnhance={() => void enhanceJapanesePrompt()}
                onOpenOptionTags={setOptionTagDialogTarget}
                onRemoveOptionTag={removeOptionPromptTag}
                  onOpenTagBrowser={setTagPromptTarget}
                  onReset={resetGenerationParameters}
                  onOpenHistory={() => setPromptHistoryOpen(true)}
                  onOpenLoraTriggers={() => setLoraTriggerDialogOpen(true)}
                  onRemoveIllustrationTag={(target, tag) => {
                    if (target === "japanese") {
                      setJapanesePromptTagIds((current) => current.filter((id) => id !== tag.id));
                      return;
                    }
                    setPositivePromptTagIds((current) => current.filter((id) => id !== tag.id));
                  }}
                  onRemoveLoraTrigger={(trigger) => setPositiveLoraTriggerIds((current) => current.filter((id) => id !== trigger.triggerId))}
                />
              </>
            ) : generateTab === "lora" ? (
              <GenerationLoraPanel
                loras={status?.loras ?? []}
                projects={loraProjects}
                selectedLoras={selectedLoras}
                onToggleLora={toggleGenerationLora}
                onUpdateLora={updateGenerationLora}
                onClearLoras={() => setSelectedLoras([])}
              />
            ) : generateTab === "pose" ? (
              <PoseEditor
                value={poseConditioning}
                controlNetModels={modelLibrary.filter((entry) => entry.type === "controlnet" && (entry.usage.toLowerCase().includes("openpose") || entry.fileName.toLowerCase().includes("openpose") || entry.fileName.toLowerCase().includes("pose"))).map((entry) => entry.fileName)}
                depthControlNetModels={modelLibrary.filter((entry) => entry.type === "controlnet" && (entry.usage.toLowerCase().includes("depth") || entry.fileName.toLowerCase().includes("depth"))).map((entry) => entry.fileName)}
                onChange={(next) => {
                  setRestoredGenerationSettings(null);
                  setPoseConditioning(next);
                }}
                savedPoses={savedPoses}
                selectedSavedPoseId={selectedSavedPoseId}
                poseTitle={poseTitle}
                onPoseTitleChange={setPoseTitle}
                onSelectSavedPose={loadSelectedPose}
                onSavePose={() => void saveCurrentPose()}
                onDeletePose={() => void deleteSelectedPose()}
                comfyUrl={comfyUrl}
                onInstallPreprocessor={installPosePreprocessor}
                outputWidth={generationWidth}
                outputHeight={generationHeight}
                disabled={action.busy}
              />
            ) : (
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
                <div className="two-columns">
                  <div className="field">
                    <label>Steps</label>
                    <input type="number" min="1" max="150" value={generationSteps} onChange={(event) => setGenerationSteps(Number(event.target.value))} />
                  </div>
                  <div className="field">
                    <label>CFG</label>
                    <input type="number" min="1" max="30" step="0.5" value={generationCfg} onChange={(event) => setGenerationCfg(Number(event.target.value))} />
                  </div>
                </div>
                <div className="field">
                  <label>Seed</label>
                  <select value={seedMode} onChange={(event) => setSeedMode(event.target.value as GenerationSeedMode)}>
                    <option value="random">Random</option>
                    <option value="fixed">Fixed</option>
                    <option value="sequence">連番</option>
                  </select>
                </div>
                <div className="field">
                  <label>Seed Value</label>
                  {seedMode === "sequence" ? (
                    <input value={seedRange} placeholder={defaultSeedRange} onChange={(event) => setSeedRange(event.target.value)} />
                  ) : (
                    <input type="number" disabled={seedMode === "random"} value={seedValue} onChange={(event) => setSeedValue(Number(event.target.value))} />
                  )}
                </div>
              </div>
            )}
            {selectedCheckpointInfo && !selectedCheckpointGenerationReady ? (
              <p className="result-message">
                このCheckpointは追加ファイルが必要です。{selectedCheckpointCompatibilityLabel}
              </p>
            ) : null}
            {generationResult?.message && <p className="result-message">{generationResult.message}</p>}
            <div className="character-generation-actions">
              <button className="generate" disabled={!canGenerateCharacter} onClick={() => void runSingleGenerate("character")}>
                <WandSparkles size={18} />
                キャラクタ生成
              </button>
              <button className="generate secondary-generate" disabled={!canGenerateCharacter || repeatRunning} onClick={() => void runRepeatGenerate("character")}>
                <RefreshCw size={18} />
                キャラクタリピート生成
              </button>
            </div>
              </>
            ) : null}
          </section>
        </section>
      )}
      {sceneAddOpen ? (
        <div className="modal-backdrop" onClick={() => setSceneAddOpen(false)}>
          <section className="modal-panel scene-add-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <PanelTitle icon={<Plus size={18} />} title="シーンへ追加" />
              <button className="icon-button" onClick={() => setSceneAddOpen(false)} title="閉じる"><X size={17} /></button>
            </div>
            <div className="scene-add-options">
              <button onClick={addSceneCharacter}><Plus size={24} /><strong>キャラを追加</strong></button>
              <button disabled={Boolean(sceneBackground)} onClick={addSceneBackground}><ImageIcon size={24} /><strong>背景を追加</strong></button>
            </div>
          </section>
        </div>
      ) : null}
      {characterLibraryOpen ? (
        <div className="modal-backdrop" onClick={() => setCharacterLibraryOpen(false)}>
          <section className="modal-panel scene-library-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <PanelTitle icon={<RefreshCw size={18} />} title="保存済みキャラ" />
              <button className="icon-button" onClick={() => setCharacterLibraryOpen(false)} title="閉じる"><X size={17} /></button>
            </div>
            <div className="scene-library-list">
              {savedSceneCharacters.length ? savedSceneCharacters.map((character) => (
                <article key={character.id}>
                  <div><strong>{character.title}</strong><time>{new Date(character.savedAt).toLocaleString("ja-JP", { hour12: false })}</time></div>
                  <div className="button-row">
                    <button className="primary" onClick={() => restoreCharacterPreset(character)}>復元</button>
                    <button className="danger" onClick={() => deleteCharacterPreset(character.id)}><Trash2 size={15} />削除</button>
                  </div>
                </article>
              )) : <p className="empty">保存済みキャラはありません。</p>}
            </div>
          </section>
        </div>
      ) : null}
      {backgroundLibraryOpen ? (
        <div className="modal-backdrop" onClick={() => setBackgroundLibraryOpen(false)}>
          <section className="modal-panel scene-library-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <PanelTitle icon={<RefreshCw size={18} />} title="保存済み背景" />
              <button className="icon-button" onClick={() => setBackgroundLibraryOpen(false)} title="閉じる"><X size={17} /></button>
            </div>
            <div className="scene-library-list">
              {savedSceneBackgrounds.length ? savedSceneBackgrounds.map((background) => (
                <article key={background.id}>
                  {background.referenceImageDataUrl ? <img src={background.referenceImageDataUrl} alt="背景サムネイル" /> : null}
                  <div><strong>{background.title}</strong><time>{new Date(background.savedAt).toLocaleString("ja-JP", { hour12: false })}</time></div>
                  <div className="button-row">
                    <button className="primary" onClick={() => restoreBackgroundPreset(background)}>復元</button>
                    <button className="danger" onClick={() => deleteBackgroundPreset(background.id)}><Trash2 size={15} />削除</button>
                  </div>
                </article>
              )) : <p className="empty">保存済み背景はありません。</p>}
            </div>
          </section>
        </div>
      ) : null}
      {repeatDialogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel repeat-modal">
            <div className="modal-header">
              <PanelTitle icon={<RefreshCw size={18} />} title="Repeat Generate" />
              <div className="repeat-status">
                {repeatRunning ? (repeatPaused ? "PAUSED" : "RUNNING") : "STOPPED"} / {repeatItems.length}枚
              </div>
            </div>
            <div className="repeat-grid">
              {repeatItems.length ? (
                repeatItems.map((item) => {
                  const image = item.result?.imageDataUrl ?? item.result?.imagePath ?? null;
                  return (
                    <article className={item.error ? "repeat-tile error" : "repeat-tile"} key={item.id}>
                      {image ? (
                        <button className="repeat-image-button" type="button" onClick={() => setRepeatPreviewImage(image)} title="拡大表示">
                          <img src={imageUrl(image)} alt={`Repeat seed ${item.seed}`} />
                        </button>
                      ) : (
                        <div className="repeat-placeholder">
                          {item.error ? <CircleAlert size={24} /> : <RefreshCw size={24} />}
                          <span>{item.error ? "失敗" : "生成中"}</span>
                        </div>
                      )}
                      <footer>
                        <div className="repeat-seed-row">
                          <strong>Seed {item.seed}</strong>
                          <button className="icon-button" disabled={!item.historyId} onClick={() => openRepeatMemoEditor(item)} title="メモ編集">
                            <Pencil size={14} />
                          </button>
                        </div>
                        {item.durationText ? <span>{item.durationText}</span> : null}
                        {item.memo ? <small className="repeat-memo">{item.memo}</small> : null}
                        {item.error ? <small>{item.error}</small> : null}
                      </footer>
                    </article>
                  );
                })
              ) : (
                <div className="repeat-empty">
                  <ImageIcon size={42} />
                  <p>生成結果がここに追加されます。</p>
                </div>
              )}
            </div>
            <div className="repeat-actions">
              <button
                disabled={!repeatRunning}
                onClick={() => {
                  setRepeatPaused((current) => !current);
                }}
              >
                <Pause size={20} />
                {repeatPaused ? "RESUME" : "PAUSE"}
              </button>
              <button
                className="danger"
                disabled={!repeatRunning}
                onClick={() => {
                  repeatStopRequested.current = true;
                  setRepeatPaused(false);
                }}
              >
                <Square size={20} />
                STOP
              </button>
              <button
                onClick={() => {
                  repeatStopRequested.current = true;
                  setRepeatPaused(false);
                  setRepeatDialogOpen(false);
                }}
              >
                <X size={20} />
                CLOSE
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {repeatPreviewImage ? (
        <div className="modal-backdrop stacked" onClick={() => setRepeatPreviewImage(null)}>
          <section className="modal-panel image-preview-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <PanelTitle icon={<ImageIcon size={18} />} title="Repeat Preview" />
              <button className="icon-button" onClick={() => setRepeatPreviewImage(null)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="image-preview-frame">
              <img src={imageUrl(repeatPreviewImage)} alt="Repeat generated preview enlarged" />
            </div>
          </section>
        </div>
      ) : null}
      {repeatMemoEditingItemId ? (
        <div className="modal-backdrop stacked">
          <section className="modal-panel repeat-memo-modal">
            <div className="modal-header">
              <PanelTitle icon={<Pencil size={18} />} title="Repeatメモ編集" />
              <button className="icon-button" onClick={() => setRepeatMemoEditingItemId(null)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="field">
              <label>Memo</label>
              <textarea value={repeatMemoText} onChange={(event) => setRepeatMemoText(event.target.value)} />
            </div>
            <div className="modal-actions">
              <button onClick={() => setRepeatMemoEditingItemId(null)}>キャンセル</button>
              <button className="primary" onClick={() => void saveRepeatHistoryMemo()}>
                <Save size={17} />
                更新
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {modelCatalogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel model-catalog-modal">
            <div className="modal-header">
              <PanelTitle icon={<Download size={18} />} title="モデルカタログ" />
              <button className="icon-button" onClick={() => setModelCatalogOpen(false)} title="閉じる"><X size={17} /></button>
            </div>
            <div className="model-mode-tabs">
              <button className={modelCatalogMode === "civitai" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelCatalogMode("civitai")}>Civitai</button>
              <button className={modelCatalogMode === "url" ? "runtime-tab active" : "runtime-tab"} onClick={() => setModelCatalogMode("url")}>URL直接</button>
            </div>
            {modelCatalogMode === "civitai" ? (
              <>
                <div className="catalog-control-grid">
                  <div className="field"><label>期間</label><select value={modelCatalogPeriod} onChange={(event) => setModelCatalogPeriod(event.target.value as typeof modelCatalogPeriod)}><option value="Day">今日</option><option value="Week">1週間</option><option value="Month">1ヶ月</option><option value="Year">1年</option><option value="AllTime">累計</option></select></div>
                  <div className="field"><label>条件</label><select value={modelCatalogSort} onChange={(event) => setModelCatalogSort(event.target.value as typeof modelCatalogSort)}><option value="Most Downloaded">人気順</option><option value="Highest Rated">評価順</option><option value="Newest">新着順</option></select></div>
                  <button className="primary catalog-action" disabled={action.busy} onClick={() => void searchModelCatalog(true)}>人気Top10を表示</button>
                </div>
                <div className="field">
                  <label>検索キーワード</label>
                  <div className="inline">
                    <input value={modelCatalogQuery} onChange={(event) => setModelCatalogQuery(event.target.value)} placeholder="OpenPose, Illustrious, anime..." onKeyDown={(event) => { if (event.key === "Enter") void searchModelCatalog(false); }} />
                    <button className="icon-button" disabled={action.busy} onClick={() => void searchModelCatalog(false)} title="検索"><RefreshCw size={17} /></button>
                  </div>
                </div>
                <div className="model-catalog-filter-band">
                  <label>MODEL TYPE</label>
                  <div className="filter-buttons">
                    {(["checkpoint", "controlnet", "vae", "other"] as ModelAssetType[]).map((type) => (
                      <button key={type} className={modelCatalogTypes.includes(type) ? "filter-chip active" : "filter-chip"} onClick={() => setModelCatalogTypes([type])}>{modelTypeLabel(type)}</button>
                    ))}
                    <button className="filter-chip openpose-shortcut" onClick={() => { setModelCatalogTypes(["controlnet"]); setModelCatalogPeriod("AllTime"); setModelCatalogQuery("OpenPose"); }}>OpenPose</button>
                    <button className="filter-chip" onClick={() => { setModelCatalogTypes(["controlnet"]); setModelCatalogPeriod("AllTime"); setModelCatalogQuery("depth"); }}>Depth</button>
                  </div>
                </div>
                {modelCatalogTypes.includes("checkpoint") ? (
                  <div className="model-catalog-filter-band">
                    <label>BASE MODEL</label>
                    <div className="filter-buttons">
                      {allBaseFilters.map((filter) => <button key={filter.name} className={baseFilters.includes(filter.name) ? "filter-chip active" : "filter-chip"} onClick={() => toggleBaseFilter(filter.name)}>{filter.name}</button>)}
                      <button className="filter-chip" onClick={() => setBaseFilters(defaultBaseFilters)}>リセット</button>
                    </div>
                  </div>
                ) : null}
                <div className="model-catalog-results">
                  {modelCatalogResults.map((model) => (
                    <article className="model-catalog-card" key={`${model.id}-${model.versionId}`}>
                      {model.imageUrl ? <img src={model.imageUrl} alt="" /> : <div className="catalog-no-image"><ImageIcon size={30} /></div>}
                      <div className="model-catalog-card-body">
                        <strong>{model.name}</strong>
                        <span>{model.creator} / {model.type} / {model.baseModel || "base不明"}</span>
                        <span>{Math.round(model.sizeKB / 1024)} MB / DL {model.downloadCount.toLocaleString()}</span>
                        <div className="catalog-card-actions">
                          <button disabled={action.busy} onClick={() => void downloadCatalogModelToLibrary(model)}><Download size={15} />Download</button>
                          <button disabled={!model.pageUrl} onClick={() => void window.illustMK.app.openExternal(model.pageUrl)}><ExternalLink size={15} />Web</button>
                        </div>
                      </div>
                    </article>
                  ))}
                  {!modelCatalogResults.length ? <p className="empty">条件を設定して検索してください。</p> : null}
                </div>
              </>
            ) : (
              <div className="model-direct-form">
                <div className="field"><label>種類</label><select value={modelDirectType} onChange={(event) => setModelDirectType(event.target.value as ModelAssetType)}>{(["checkpoint", "controlnet", "vae", "clip", "other"] as ModelAssetType[]).map((type) => <option value={type} key={type}>{modelTypeLabel(type)}</option>)}</select></div>
                <div className="field"><label>URL</label><input value={modelDirectUrl} onChange={(event) => setModelDirectUrl(event.target.value)} placeholder="https://.../model.safetensors" /></div>
                <div className="field"><label>保存名</label><input value={modelDirectFileName} onChange={(event) => setModelDirectFileName(event.target.value)} placeholder="未指定ならURLから決定" /></div>
                <div className="two-columns">
                  <div className="field"><label>BASE MODEL</label><input value={modelDirectBase} onChange={(event) => setModelDirectBase(event.target.value)} /></div>
                  <div className="field"><label>用途</label><input value={modelDirectUsage} onChange={(event) => setModelDirectUsage(event.target.value)} placeholder="OpenPoseなど" /></div>
                </div>
                <button className="primary" disabled={!modelDirectUrl.trim() || action.busy} onClick={() => void downloadDirectModelToLibrary()}><Download size={17} />ダウンロード</button>
              </div>
            )}
          </section>
        </div>
      ) : null}
      {modelImportOpen ? (
        <div className="modal-backdrop stacked">
          <section className="modal-panel model-import-modal">
            <div className="modal-header"><PanelTitle icon={<FolderOpen size={18} />} title="モデルをインポート" /><button className="icon-button" onClick={() => setModelImportOpen(false)} title="閉じる"><X size={17} /></button></div>
            <div className="field"><label>種類</label><select value={modelImportType} onChange={(event) => setModelImportType(event.target.value as ModelAssetType)}>{(["checkpoint", "controlnet", "vae", "clip", "other"] as ModelAssetType[]).map((type) => <option value={type} key={type}>{modelTypeLabel(type)}</option>)}</select></div>
            <div className="modal-actions"><button onClick={() => setModelImportOpen(false)}>キャンセル</button><button className="primary" onClick={() => void importModelLibraryFile()}><FolderOpen size={16} />ファイルを選択</button></div>
          </section>
        </div>
      ) : null}
      {optionTagDialogTarget ? (
        <div className="modal-backdrop">
          <section className="modal-panel option-tag-modal">
            <div className="modal-header">
              <PanelTitle
                icon={<WandSparkles size={18} />}
                title={`${optionTagDialogTarget.startsWith("background-") ? "背景 " : ""}${optionTagDialogTarget.endsWith("positive") ? "Positive" : "Negative"} オプションタグ`}
              />
              <button className="icon-button" onClick={() => setOptionTagDialogTarget(null)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="option-tag-toolbar">
              <button onClick={() => openOptionTagEditor()}>
                <Plus size={17} />
                タグ追加
              </button>
            </div>
            <OptionPromptTagBrowser
              tags={optionPromptTags}
              onApply={applyOptionPromptTag}
              onEdit={openOptionTagEditor}
            />
          </section>
        </div>
      ) : null}
      {tagPromptTarget ? (
        <div className="modal-backdrop">
          <section className="modal-panel option-tag-modal">
            <div className="modal-header">
              <PanelTitle icon={<WandSparkles size={18} />} title="タグ一覧" />
              <button className="icon-button" onClick={() => setTagPromptTarget(null)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="option-tag-toolbar">
              <button onClick={() => openNewTagDialog()}>
                <Plus size={17} />
                タグ追加
              </button>
            </div>
            <IllustrationTagBrowser tags={illustrationTags} onApply={applyIllustrationTagToPrompt} onEdit={openEditTagDialog} onAddCategory={openNewTagDialogForCategory} />
          </section>
        </div>
      ) : null}
      {loraTriggerDialogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel option-tag-modal">
            <div className="modal-header">
              <PanelTitle icon={<Boxes size={18} />} title="LoRAトリガーワード一覧" />
              <button className="icon-button" onClick={() => setLoraTriggerDialogOpen(false)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="option-tag-browser">
              {selectedLoraTriggers.length ? (
                <div className="option-tag-list">
                  {selectedLoraTriggers.map((trigger) => (
                    <button
                      className={positiveLoraTriggerIds.includes(trigger.triggerId) ? "lora-trigger-row selected" : "lora-trigger-row"}
                      key={trigger.triggerId}
                      onClick={() => setPositiveLoraTriggerIds((current) => (current.includes(trigger.triggerId) ? current : [...current, trigger.triggerId]))}
                    >
                      <strong>{trigger.title}</strong>
                      <span>{trigger.triggerWord}</span>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="empty">選択中LoRAにトリガーワードがありません。</p>
              )}
            </div>
          </section>
        </div>
      ) : null}
      {optionTagEditorOpen ? (
        <div className="modal-backdrop stacked">
          <section className="modal-panel option-tag-editor">
            <div className="modal-header">
              <PanelTitle icon={<Pencil size={18} />} title={optionTagEditingId ? "オプションタグ編集" : "オプションタグ追加"} />
              <button className="icon-button" onClick={() => setOptionTagEditorOpen(false)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="two-columns">
              <div className="field">
                <label>日本語</label>
                <input value={optionTagJapanese} onChange={(event) => setOptionTagJapanese(event.target.value)} placeholder="高品質" />
              </div>
              <div className="field">
                <label>英語</label>
                <input value={optionTagEnglish} onChange={(event) => setOptionTagEnglish(event.target.value)} placeholder="masterpiece" />
              </div>
            </div>
            <div className="two-columns">
              <div className="field">
                <label>色</label>
                <input type="color" value={optionTagColor} onChange={(event) => setOptionTagColor(event.target.value)} />
              </div>
              <div className="field">
                <label>カテゴリ</label>
                <input
                  list="option-tag-categories"
                  value={optionTagCategory}
                  onChange={(event) => setOptionTagCategory(event.target.value)}
                  placeholder="品質"
                />
                <datalist id="option-tag-categories">
                  {optionPromptTagEditCategories(optionPromptTags).map((category) => (
                    <option value={category} key={category}>
                      {category}
                    </option>
                  ))}
                </datalist>
              </div>
            </div>
            <div className="button-row">
              <button onClick={() => setOptionTagEditorOpen(false)}>
                <Square size={17} />
                閉じる
              </button>
              {optionTagEditingId ? (
                <button className="danger" onClick={deleteOptionPromptTag}>
                  <Trash2 size={17} />
                  削除
                </button>
              ) : null}
              <button className="primary" disabled={!optionTagJapanese.trim() || !optionTagEnglish.trim()} onClick={saveOptionPromptTag}>
                <Save size={17} />
                更新
              </button>
            </div>
          </section>
        </div>
      ) : null}
      {tagDialogOpen ? (
        <div className="modal-backdrop">
          <section className="modal-panel tag-modal">
            <div className="modal-header">
              <PanelTitle icon={<WandSparkles size={18} />} title={tagEditingId ? "タグ編集" : "タグ追加"} />
              <button className="icon-button" onClick={() => setTagDialogOpen(false)} title="閉じる">
                <Square size={16} />
              </button>
            </div>
            <div className="tag-modal-body">
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
                  <label>色</label>
                  <input type="color" value={tagColor} onChange={(event) => setTagColor(event.target.value)} />
                </div>
                <div className="field">
                  <label>カテゴリ</label>
                  <input list="illustration-tag-categories" value={tagCategory} onChange={(event) => setTagCategory(event.target.value)} />
                  <datalist id="illustration-tag-categories">
                    {uniqueCategories([...illustrationTagEditCategories(illustrationTags), tagCategory]).map((category) => (
                      <option value={category} key={category}>
                        {category}
                      </option>
                    ))}
                  </datalist>
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
            </div>
            <div className="modal-actions">
              <button onClick={() => setTagDialogOpen(false)}>キャンセル</button>
              {tagEditingId ? (
                <button className="danger" onClick={() => void deleteIllustrationTag(tagEditingId)}>
                  <Trash2 size={17} />
                  削除
                </button>
              ) : null}
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
                      <div className="history-tags">
                        <span>
                          {entry.generationSettings?.generationScope === "scene"
                            ? "総合"
                            : entry.generationSettings?.generationScope === "background"
                              ? `背景: ${entry.generationSettings.scene?.background?.title ?? "名称未記録"}`
                            : entry.generationSettings?.generationScope === "character"
                              ? `キャラクタ: ${entry.generationSettings.targetCharacterTitle ?? "名称未記録"}`
                              : "従来生成"}
                        </span>
                        {entry.runtimeRootMatches === false ? <span>別DataPath: {entry.runtimeRootLabel ?? "不明"}</span> : <span>現在DataPath</span>}
                        {entry.restorable === false ? <span className="history-warning">不足あり</span> : null}
                      </div>
                      {entry.japanesePrompt ? <p className="history-japanese">{entry.japanesePrompt}</p> : null}
                      <p>{entry.prompt}</p>
                      {entry.memo ? <p className="history-memo">{entry.memo}</p> : null}
                      {entry.missingAssets?.length ? <small>不足: {entry.missingAssets.join(" / ")}</small> : null}
                      <small>{entry.negativePrompt}</small>
                      <div className="history-card-actions">
                        <button disabled={entry.restorable === false} onClick={() => applyPromptHistory(entry)}>
                          <CheckCircle2 size={17} />
                          {entry.generationSettings?.scene ? "シーンを復元" : "流用"}
                        </button>
                        {entry.generationSettings?.scene?.characters.map((character, index) => (
                          <button key={`${entry.id}-${character.id}`} disabled={entry.restorable === false} onClick={() => restoreCharacterFromHistory(character)}>
                            キャラ{index + 1}を復元
                          </button>
                        ))}
                        {entry.generationSettings?.scene?.background ? (
                          <button disabled={entry.restorable === false} onClick={() => restoreBackgroundFromHistory(entry.generationSettings!.scene!.background!)}>
                            背景を復元
                          </button>
                        ) : null}
                        <button className="icon-button danger" onClick={() => void deletePromptHistory(entry)} title="削除">
                          <Trash2 size={15} />
                        </button>
                      </div>
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
              <div className="trigger-word-list">
                {loraTriggerWords.map((word, index) => (
                  <div className="trigger-word-row" key={index}>
                    <input placeholder="mkchar" value={word} onChange={(event) => updateTriggerWordList("dialog", index, event.target.value)} />
                    <button className="icon-button" type="button" onClick={() => addTriggerWordRow("dialog")} title="追加">
                      <Plus size={15} />
                    </button>
                    <button className="icon-button" type="button" disabled={loraTriggerWords.length <= 1} onClick={() => removeTriggerWordRow("dialog", index)} title="削除">
                      <Trash2 size={15} />
                    </button>
                  </div>
                ))}
              </div>
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
            <button className="primary" disabled={action.busy} onClick={() => void searchLoraCatalog(1)}>
              <RefreshCw size={17} />
              検索
            </button>
            <div className="catalog-list lora-catalog-list" onScroll={handleLoraCatalogScroll}>
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
              {loraCatalogLoadingMore ? <p className="empty">more...</p> : null}
              {loraCatalogModels.length > 0 && !loraCatalogHasMore ? <p className="empty">これ以上ありません。</p> : null}
            </div>
          </section>
        </div>
      ) : null}
      {editingDataPath ? (
        <div className="modal-backdrop">
          <section className="modal-panel data-path-edit-modal">
            <div className="modal-header">
              <PanelTitle icon={<FolderOpen size={18} />} title="DataPathメモ編集" />
              <button className="icon-button" onClick={() => setEditingDataPath(null)} title="閉じる">
                <X size={17} />
              </button>
            </div>
            <div className="field">
              <label>PATH</label>
              <input value={editingDataPath.path} readOnly />
            </div>
            <div className="field">
              <label>メモ</label>
              <textarea value={dataPathMemo} onChange={(event) => setDataPathMemo(event.target.value)} />
            </div>
            <div className="modal-actions">
              <button onClick={() => setEditingDataPath(null)}>キャンセル</button>
              <button className="primary" disabled={action.busy} onClick={() => void saveDataPathMemo()}>
                <Save size={17} />
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
  optionPromptTags,
  japanesePromptTags,
  positivePromptTags,
  positiveOptionTags,
  negativeOptionTags,
  positiveLoraTriggers,
  canEnhancePrompt,
  disabledReason,
  onEnhance,
  onOpenOptionTags,
  onRemoveOptionTag,
  onOpenTagBrowser,
  onReset,
  onOpenHistory,
  onOpenLoraTriggers,
  onRemoveIllustrationTag,
  onRemoveLoraTrigger
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
  optionPromptTags: OptionPromptTag[];
  japanesePromptTags: IllustrationTag[];
  positivePromptTags: IllustrationTag[];
  positiveOptionTags: OptionPromptTag[];
  negativeOptionTags: OptionPromptTag[];
  positiveLoraTriggers: AppliedLoraTrigger[];
  canEnhancePrompt: boolean;
  disabledReason: string;
  onEnhance: () => void;
  onOpenOptionTags: (target: PromptKind) => void;
  onRemoveOptionTag: (target: PromptKind, tag: OptionPromptTag) => void;
  onOpenTagBrowser: (target: TagPromptTarget) => void;
  onReset: () => void;
  onOpenHistory: () => void;
  onOpenLoraTriggers: () => void;
  onRemoveIllustrationTag: (target: TagPromptTarget, tag: IllustrationTag) => void;
  onRemoveLoraTrigger: (trigger: AppliedLoraTrigger) => void;
}) {
  return (
    <div className="prompt-transform">
      <div className="field">
        <div className="prompt-field-header">
          <label>Japanese Prompt</label>
          <button onClick={onReset}>
            <X size={15} />
            リセット
          </button>
          <button onClick={onOpenHistory}>
            <RefreshCw size={15} />
            履歴
          </button>
          <button onClick={() => onOpenTagBrowser("japanese")}>
            <WandSparkles size={15} />
            タグ
          </button>
        </div>
        <textarea
          placeholder="例: 夕方の教室で、青緑のツインテールの女子高生。少し照れている感じ。"
          value={japanesePrompt}
          onChange={(event) => setJapanesePrompt(event.target.value)}
        />
        <AppliedIllustrationTags target="japanese" tags={japanesePromptTags} onRemove={onRemoveIllustrationTag} />
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
        <div className="prompt-field-header">
          <label>Positive Prompt</label>
          <button onClick={() => onOpenTagBrowser("positive")}>
            <WandSparkles size={15} />
            タグ
          </button>
          <button onClick={() => onOpenOptionTags("positive")}>
            <WandSparkles size={15} />
            オプションタグ
          </button>
          <button onClick={onOpenLoraTriggers}>
            <Boxes size={15} />
            LoRA
          </button>
        </div>
        <textarea
          value={prompt}
          onChange={(event) => {
            setPrompt(event.target.value);
          }}
        />
        <AppliedOptionTags
          target="positive"
          tags={positiveOptionTags}
          onRemove={onRemoveOptionTag}
        />
        <AppliedIllustrationTags target="positive" tags={positivePromptTags} onRemove={onRemoveIllustrationTag} />
        <AppliedLoraTriggers triggers={positiveLoraTriggers} onRemove={onRemoveLoraTrigger} />
      </div>
      <div className="field">
        <div className="prompt-field-header">
          <label>Negative Prompt</label>
          <button onClick={() => onOpenOptionTags("negative")}>
            <WandSparkles size={15} />
            オプションタグ
          </button>
        </div>
        <textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} />
        <AppliedOptionTags
          target="negative"
          tags={negativeOptionTags}
          onRemove={onRemoveOptionTag}
        />
      </div>
    </div>
  );
}

function AppliedOptionTags({
  target,
  tags,
  onRemove
}: {
  target: PromptKind;
  tags: OptionPromptTag[];
  onRemove: (target: PromptKind, tag: OptionPromptTag) => void;
}) {
  if (!tags.length) return null;
  return (
    <div className="applied-option-tags">
      {tags.map((tag) => (
        <span className="option-tag-chip" style={{ borderColor: tag.color, backgroundColor: hexToTagBackground(tag.color) }} key={tag.id}>
          {tag.japanese}
          <button onClick={() => onRemove(target, tag)} title="Promptから削除">
            <Trash2 size={13} />
          </button>
        </span>
      ))}
    </div>
  );
}

function AppliedIllustrationTags({
  target,
  tags,
  onRemove
}: {
  target: TagPromptTarget;
  tags: IllustrationTag[];
  onRemove: (target: TagPromptTarget, tag: IllustrationTag) => void;
}) {
  if (!tags.length) return null;
  return (
    <div className="applied-option-tags">
      {tags.map((tag) => (
        <span className="option-tag-chip" style={{ borderColor: tag.color, backgroundColor: hexToTagBackground(tag.color) }} key={tag.id}>
          {tag.name}
          <button onClick={() => onRemove(target, tag)} title="Promptから削除">
            <Trash2 size={13} />
          </button>
        </span>
      ))}
    </div>
  );
}

function AppliedLoraTriggers({
  triggers,
  onRemove
}: {
  triggers: AppliedLoraTrigger[];
  onRemove: (trigger: AppliedLoraTrigger) => void;
}) {
  if (!triggers.length) return null;
  return (
    <div className="applied-option-tags">
      {triggers.map((trigger) => (
        <span className="option-tag-chip" key={trigger.loraName}>
          {trigger.title}
          <button onClick={() => onRemove(trigger)} title="Promptから削除">
            <Trash2 size={13} />
          </button>
        </span>
      ))}
    </div>
  );
}

function OptionPromptTagBrowser({
  tags,
  onApply,
  onEdit
}: {
  tags: OptionPromptTag[];
  onApply: (tag: OptionPromptTag) => void;
  onEdit: (tag: OptionPromptTag) => void;
}) {
  const groups = groupOptionPromptTags(tags);
  if (!groups.length) return <p className="empty">まだオプションタグがありません。</p>;
  return (
    <div className="option-tag-browser">
      {groups.map(({ category, items }) => (
        <section className="option-tag-group" key={category}>
          <div className="option-tag-group-title">
            <span>{category}</span>
          </div>
          <div className="option-tag-list">
            {items.map((tag) => (
              <span className="option-tag-chip" style={{ borderColor: tag.color, backgroundColor: hexToTagBackground(tag.color) }} key={tag.id}>
                <button className="option-tag-name" onClick={() => onApply(tag)}>{tag.japanese}</button>
                <button className="option-tag-edit" onClick={() => onEdit(tag)} title="編集">
                  <Pencil size={13} />
                </button>
              </span>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function IllustrationTagBrowser({
  tags,
  onApply,
  onEdit,
  onAddCategory
}: {
  tags: IllustrationTag[];
  onApply: (tag: IllustrationTag) => void;
  onEdit: (tag: IllustrationTag) => void;
  onAddCategory: (category: string) => void;
}) {
  const groups = groupIllustrationTags(tags);
  if (!groups.length) {
    return (
      <div className="option-tag-browser">
        <p className="empty">まだタグがありません。</p>
        <button className="option-tag-add-category" onClick={() => onAddCategory("未分類")}>
          <Plus size={15} />
          未分類に追加
        </button>
      </div>
    );
  }
  return (
    <div className="option-tag-browser">
      {groups.map(({ category, items }) => (
        <section className="option-tag-group" key={category}>
          <div className="option-tag-group-title">
            <span>{category}</span>
          </div>
          <div className="option-tag-list">
            {items.map((tag) => (
              <span className="option-tag-chip" style={{ borderColor: tag.color, backgroundColor: hexToTagBackground(tag.color) }} key={tag.id}>
                <button className="option-tag-name" onClick={() => onApply(tag)}>{tag.name}</button>
                <button className="option-tag-edit" onClick={() => onEdit(tag)} title="編集">
                  <Pencil size={13} />
                </button>
              </span>
            ))}
            <button className="option-tag-add-category" onClick={() => onAddCategory(category)}>
              <Plus size={15} />
              {category}に追加
            </button>
          </div>
        </section>
      ))}
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
  projects,
  selectedLoras,
  onToggleLora,
  onUpdateLora,
  onClearLoras
}: {
  loras: RuntimeStatus["loras"];
  projects: LoraTrainingProject[];
  selectedLoras: GenerationLora[];
  onToggleLora: (name: string) => void;
  onUpdateLora: (name: string, field: "modelStrength" | "clipStrength", value: number) => void;
  onClearLoras: () => void;
}) {
  const loraFileByName = useMemo(
    () => new Map(loras.filter((lora) => !isStepLoraName(lora.name)).map((lora) => [lora.name, lora])),
    [loras]
  );
  const displayProjects = projects.filter((project) => loraFileByName.has(project.outputName));
  return (
    <div className="lora-apply-panel">
      <div className="tag-builder-header">
        <h2>
          LoRA適用
          {selectedLoras.length ? <span className="count-badge">{selectedLoras.length}</span> : null}
        </h2>
        <div className="inline-actions">
          <span>{selectedLoras.length}件選択</span>
          <button className="tiny-button" disabled={!selectedLoras.length} onClick={onClearLoras}>選択クリア</button>
        </div>
      </div>
      <div className="lora-apply-list">
        {displayProjects.length ? (
          displayProjects.map((project) => {
            const lora = loraFileByName.get(project.outputName);
            const selected = selectedLoras.find((item) => item.name === project.outputName);
            return (
              <article className={selected ? "lora-apply-card lora-project-card selected" : "lora-apply-card lora-project-card"} key={project.id}>
                <div className="lora-project-thumb">
                  <ImageIcon className="lora-project-placeholder-icon" size={22} />
                  {project?.thumbnailDataUrl || project?.thumbnailPath ? (
                    <img src={project.thumbnailDataUrl || `file://${project.thumbnailPath ?? ""}`} alt="" onError={(event) => { event.currentTarget.remove(); }} />
                  ) : null}
                </div>
                <div className="lora-project-body">
                  <header>
                    <strong>{project.name || project.outputName}</strong>
                    <button className={selected ? "tiny-button active" : "tiny-button"} onClick={() => onToggleLora(project.outputName)}>
                      {selected ? "使用中" : "使用"}
                    </button>
                  </header>
                  <div className="lora-meta-tags">
                    <span className="meta-tag base">{project.baseCheckpoint ? inferCheckpointBaseModel(project.baseCheckpoint) : "Base不明"}</span>
                    {getLoraTriggerWords(project).map((word) => <span className="meta-tag trigger" key={word}>{word}</span>)}
                    {lora?.sizeLabel ? <span className="meta-tag size">{lora.sizeLabel}</span> : null}
                  </div>
                  {project.message ? <p>{project.message}</p> : null}
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
                          onChange={(event) => onUpdateLora(project.outputName, "modelStrength", Number(event.target.value))}
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
                          onChange={(event) => onUpdateLora(project.outputName, "clipStrength", Number(event.target.value))}
                        />
                      </label>
                    </div>
                  ) : null}
                </div>
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
  runtimeTab,
  ollamaModel,
  selectedCheckpoint,
  checkpointBaseModel,
  usageHistory
}: {
  status: RuntimeStatus | null;
  ollamaStatus?: OllamaStatus | null;
  runtimeTab: "macos" | "docker";
  ollamaModel: string;
  selectedCheckpoint: RuntimeStatus["checkpoints"][number] | null;
  checkpointBaseModel: string;
  usageHistory: UsageSample[];
}) {
  const runtimeItems: Array<{ label: string; value: string; tone: StatusTone }> =
    runtimeTab === "docker"
      ? [
          { label: "Docker", value: status?.dockerDesktopInstalled ? "検出" : "未検出", tone: status?.dockerDesktopInstalled ? "ok" : "error" },
          { label: "Docker API", value: status?.dockerDaemonReady ? "接続OK" : "未起動", tone: status?.dockerDaemonReady ? "ok" : "warn" },
          { label: "Compose", value: status?.composeCommand ? "OK" : "未検出", tone: status?.composeCommand ? "ok" : "error" }
        ]
      : [{ label: "ComfyUI", value: status?.comfyReachable ? "接続中" : status?.nativeRunning ? "起動中" : "未接続", tone: status?.comfyReachable ? "ok" : status?.nativeRunning ? "warn" : "error" }];

  const ollamaValue = ollamaStatus?.reachable && ollamaStatus.dataLinked ? "接続OK" : ollamaStatus?.reachable ? "保存先NG" : ollamaStatus?.installed ? "未起動" : "未検出";
  const ollamaTone: StatusTone = ollamaStatus?.reachable && ollamaStatus.dataLinked ? "ok" : ollamaStatus?.installed ? "warn" : "error";
  const checkpointTone: StatusTone = selectedCheckpoint?.simpleWorkflowReady ? "ok" : selectedCheckpoint ? "warn" : "error";
  const baseModelLabel = checkpointBaseModel || inferCheckpointBaseModel(selectedCheckpoint?.name ?? "");
  const items = [
    ...runtimeItems,
    { label: "Ollama", value: ollamaValue, tone: ollamaTone },
    { label: "選択モデル", value: selectedCheckpoint?.name ?? "未選択", tone: checkpointTone, tag: baseModelLabel },
    { label: "LLM", value: ollamaModel || "未選択", tone: ollamaModel ? "ok" : "warn" }
  ];

  return (
    <div className="status-grid">
      {items.map(({ label, value, tone, tag }) => (
        <div className={`status-card ${tone} ${label === "ComfyUI" || label === "Ollama" ? "compact" : ""}`} key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
          {tag ? <em>{tag}</em> : null}
        </div>
      ))}
      <UsageStatusCard label="CPU" value={status?.systemUsage.cpuPercent ?? null} samples={usageHistory.map((sample) => sample.cpu)} />
      <UsageStatusCard
        label="GPU"
        value={status?.systemUsage.gpuPercent ?? null}
        samples={usageHistory.map((sample) => sample.gpu)}
        unavailableText={status?.systemUsage.gpuMessage ?? "取得不可"}
      />
    </div>
  );
}

function UsageStatusCard({
  label,
  value,
  samples,
  unavailableText = "取得中"
}: {
  label: string;
  value: number | null;
  samples: Array<number | null>;
  unavailableText?: string;
}) {
  const points = samples.length
    ? samples
        .map((sample, index) => {
          const x = samples.length <= 1 ? 100 : (index / (samples.length - 1)) * 100;
          const normalized = sample ?? 0;
          const y = 34 - (Math.max(0, Math.min(100, normalized)) / 100) * 30;
          return `${x},${y}`;
        })
        .join(" ")
    : "";
  return (
    <div className={`status-card usage ${value === null ? "warn" : "ok"}`}>
      <span>{label}</span>
      <strong>{value === null ? unavailableText : `${value.toFixed(1)}%`}</strong>
      <svg viewBox="0 0 100 36" preserveAspectRatio="none" aria-hidden="true">
        <polyline points={points} />
      </svg>
    </div>
  );
}

function inferCheckpointBaseModel(fileName: string): string {
  const name = fileName.toLowerCase();
  if (!name) return "";
  if (name.includes("illustrious") || name.includes("illustriousxl")) return "Illustrious";
  if (name.includes("anima")) return "Anima";
  if (name.includes("animagine")) return "Animagine";
  if (name.includes("pony")) return "Pony";
  if (name.includes("krea")) return "Krea";
  if (name.includes("flux")) return "Flux";
  if (name.includes("qwen")) return "Qwen Image";
  if (name.includes("sdxl") || name.includes("xl_") || name.includes("_xl")) return "SDXL";
  if (name.includes("sd15") || name.includes("sd1.5") || name.includes("sd_15")) return "SD 1.5";
  return "Base不明";
}

function getLoraTriggerWords(project?: LoraTrainingProject | null): string[] {
  if (!project) return [];
  return normalizedTriggerWords(project.triggerWords ?? [], project.triggerWord);
}

function normalizedTriggerWords(words: string[], fallback = ""): string[] {
  return Array.from(new Set([...words, fallback].map((word) => word.trim()).filter(Boolean)));
}

function sameStringArray(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameGenerationLoras(left: GenerationLora[], right: GenerationLora[]): boolean {
  return left.length === right.length && left.every((value, index) => {
    const other = right[index];
    return Boolean(other) && value.name === other.name && value.modelStrength === other.modelStrength && value.clipStrength === other.clipStrength;
  });
}

function modelTypeLabel(type: ModelAssetType): string {
  if (type === "checkpoint") return "Checkpoint";
  if (type === "controlnet") return "ControlNet";
  if (type === "vae") return "VAE";
  if (type === "clip") return "CLIP";
  return "その他";
}

function samePoseConditioning(left: PoseConditioning | undefined, right: PoseConditioning): boolean {
  if (!left?.enabled && !right.enabled) return true;
  if (!left || left.enabled !== right.enabled) return false;
  return (
    left.controlNet === right.controlNet &&
    left.strength === right.strength &&
    left.startPercent === right.startPercent &&
    left.endPercent === right.endPercent &&
    (left.depthControlNet ?? "") === (right.depthControlNet ?? "") &&
    (left.depthStrength ?? 0.65) === (right.depthStrength ?? 0.65) &&
    (left.depthStartPercent ?? 0) === (right.depthStartPercent ?? 0) &&
    (left.depthEndPercent ?? 0.8) === (right.depthEndPercent ?? 0.8) &&
    left.imageDataUrl === right.imageDataUrl &&
    (left.depthImageDataUrl ?? "") === (right.depthImageDataUrl ?? "") &&
    (left.cameraFov ?? 38) === (right.cameraFov ?? 38) &&
    left.referenceImageDataUrl === right.referenceImageDataUrl &&
    left.visionModel === right.visionModel &&
    left.positivePrompt === right.positivePrompt &&
    left.negativePrompt === right.negativePrompt &&
    left.preserveCharacterProportions === right.preserveCharacterProportions &&
    (left.bodyModel ?? "male") === (right.bodyModel ?? "male") &&
    JSON.stringify(left.bodyShape ?? {}) === JSON.stringify(right.bodyShape ?? {}) &&
    JSON.stringify(left.handSettings ?? {}) === JSON.stringify(right.handSettings ?? {}) &&
    left.analysisStale === right.analysisStale
  );
}

function normalizePoseConditioning(value?: PoseConditioning): PoseConditioning {
  const defaults = createDefaultPoseConditioning();
  if (!value) return defaults;
  return {
    ...defaults,
    ...structuredClone(value),
    bodyModel: value.bodyModel ?? "male",
    preserveCharacterProportions: value.preserveCharacterProportions ?? true,
    bodyShape: defaults.bodyShape ? { ...defaults.bodyShape, ...value.bodyShape } : value.bodyShape,
    handSettings: {
      left: { ...defaults.handSettings!.left, ...value.handSettings?.left },
      right: { ...defaults.handSettings!.right, ...value.handSettings?.right }
    },
    jointRotations: value.jointRotations ?? defaults.jointRotations,
    cameraPosition: value.cameraPosition ?? defaults.cameraPosition,
    cameraTarget: value.cameraTarget ?? defaults.cameraTarget,
    analysisStale: value.analysisStale ?? !value.positivePrompt?.trim()
  };
}

function formatGenerationDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.round(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes ? `${minutes}分${seconds}秒` : `${seconds}秒`;
}

function normalizeExistingAssetName<T extends { name: string }>(name: string, files: T[]): string {
  return name && files.some((file) => file.name === name) ? name : "";
}

function normalizeOptionPromptTags(value: unknown): OptionPromptTag[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const tag = item as Partial<OptionPromptTag>;
    if (!tag.japanese || !tag.english) return [];
    return [
      {
        id: tag.id || crypto.randomUUID(),
        japanese: String(tag.japanese),
        english: String(tag.english),
        color: /^#[0-9a-f]{6}$/i.test(String(tag.color ?? "")) ? String(tag.color) : "#3388ff",
        category: String(tag.category || "未分類")
      }
    ];
  });
}

function optionPromptTagCategories(tags: OptionPromptTag[]): string[] {
  const defaults = ["品質", "人物", "構図", "視線", "服装", "背景", "画風", "Negative"];
  return uniqueCategories([...defaults, ...tags.map((tag) => tag.category)]);
}

function optionPromptTagEditCategories(tags: OptionPromptTag[]): string[] {
  return uniqueCategories(groupOptionPromptTags(tags).map((group) => group.category));
}

function illustrationTagCategories(tags: IllustrationTag[]): string[] {
  const defaults = ["衣装", "キャラ", "構図", "背景", "画風", "品質", "その他"];
  return uniqueCategories([...defaults, ...tags.map((tag) => tag.category)]);
}

function illustrationTagEditCategories(tags: IllustrationTag[]): string[] {
  return uniqueCategories(tags.map((tag) => tag.category));
}

function uniqueCategories(categories: Array<string | undefined | null>): string[] {
  return Array.from(new Set(categories.map((category) => category?.trim()).filter((category): category is string => Boolean(category))));
}

function groupIllustrationTags(tags: IllustrationTag[]): Array<{ category: string; items: IllustrationTag[] }> {
  const groups = new Map<string, IllustrationTag[]>();
  tags.forEach((tag) => {
    const category = tag.category || "未分類";
    groups.set(category, [...(groups.get(category) ?? []), tag]);
  });
  return Array.from(groups.entries()).map(([category, items]) => ({ category, items }));
}

function groupOptionPromptTags(tags: OptionPromptTag[]): Array<{ category: string; items: OptionPromptTag[] }> {
  const groups = new Map<string, OptionPromptTag[]>();
  tags.forEach((tag) => {
    const category = tag.category || "未分類";
    groups.set(category, [...(groups.get(category) ?? []), tag]);
  });
  return Array.from(groups.entries()).map(([category, items]) => ({ category, items }));
}

function splitPromptTags(promptText: string): string[] {
  return promptText.split(",").map((part) => part.trim()).filter(Boolean);
}

function promptContainsTag(promptText: string, tag: string): boolean {
  const normalized = tag.trim().toLowerCase();
  return splitPromptTags(promptText).some((part) => part.toLowerCase() === normalized);
}

function appendPromptTag(promptText: string, tag: string): string {
  const normalized = tag.trim();
  if (!normalized || promptContainsTag(promptText, normalized)) return promptText;
  const parts = splitPromptTags(promptText);
  return [...parts, normalized].join(", ");
}

function appendPromptFragment(promptText: string, fragment: string): string {
  const normalized = fragment.trim();
  if (!normalized) return promptText;
  const current = promptText.trim();
  if (!current) return normalized;
  return `${current}\n${normalized}`;
}

function composeGenerationPrompt(promptText: string, fragments: string[]): string {
  return fragments.reduce((current, fragment) => appendPromptFragment(current, fragment), promptText);
}

function removePromptTag(promptText: string, tag: string): string {
  const normalized = tag.trim().toLowerCase();
  if (!normalized) return promptText;
  return splitPromptTags(promptText).filter((part) => part.toLowerCase() !== normalized).join(", ");
}

function isStepLoraName(name: string): boolean {
  const fileName = name.split(/[\\/]/).pop() ?? name;
  return /(?:^|[-_])step\d+/i.test(fileName);
}

function hexToTagBackground(hex: string): string {
  const safeHex = /^#[0-9a-f]{6}$/i.test(hex) ? hex : "#3388ff";
  return `${safeHex}22`;
}

function readLocalArray<T>(key: string): T[] {
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? "[]");
    return Array.isArray(value) ? value as T[] : [];
  } catch {
    return [];
  }
}

createRoot(document.getElementById("root") as HTMLElement).render(<App />);
