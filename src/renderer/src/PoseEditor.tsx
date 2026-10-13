import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { Camera, Download, ExternalLink, FolderOpen, Images, RotateCcw, Save, Search, Trash2, X } from "lucide-react";
import type { CivitaiModelCard, OpenPoseAsset, OpenPoseCatalogItem, PoseBodyShape, PoseConditioning, PoseExtractionResult, PoseJointRotation, PosePreprocessorStatus, SavedPose } from "../../main/runtime";

type PoseEditorProps = {
  value: PoseConditioning;
  controlNetModels: string[];
  depthControlNetModels: string[];
  onChange: (value: PoseConditioning) => void;
  savedPoses: SavedPose[];
  selectedSavedPoseId: string;
  poseTitle: string;
  onPoseTitleChange: (title: string) => void;
  onSelectSavedPose: (id: string) => void;
  onSavePose: () => void;
  onDeletePose: () => void;
  comfyUrl: string;
  onInstallPreprocessor: () => Promise<void>;
  outputWidth: number;
  outputHeight: number;
  disabled?: boolean;
};

type Rig = {
  root: THREE.Group;
  joints: Map<string, THREE.Object3D>;
  selectable: THREE.Object3D[];
  helpers: THREE.Object3D[];
  restQuaternions: Map<string, THREE.Quaternion>;
  restScales: Map<string, THREE.Vector3>;
  restPositions: Map<string, THREE.Vector3>;
};

type DerivedPosePrompts = {
  positivePrompt: string;
  negativePrompt: string;
};

type JointLimit = {
  min: PoseJointRotation;
  max: PoseJointRotation;
};

const jointLabels: Record<string, string> = {
  hips: "腰",
  spine: "背中",
  chest: "胸",
  neck: "首",
  head: "頭",
  leftShoulder: "左肩",
  leftElbow: "左ひじ",
  leftWrist: "左手首",
  rightShoulder: "右肩",
  rightElbow: "右ひじ",
  rightWrist: "右手首",
  leftHip: "左脚付け根",
  leftKnee: "左ひざ",
  leftAnkle: "左足首",
  rightHip: "右脚付け根",
  rightKnee: "右ひざ",
  rightAnkle: "右足首"
};

const defaultBodyShape: PoseBodyShape = {
  height: 1,
  headSize: 1,
  build: 1,
  shoulderWidth: 1,
  hipWidth: 1,
  torsoLength: 1,
  armLength: 1,
  legLength: 1
};

const bodyShapeControls: Array<{ key: keyof PoseBodyShape; label: string; min: number; max: number }> = [
  { key: "height", label: "身長", min: 0.78, max: 1.22 },
  { key: "headSize", label: "頭サイズ", min: 0.82, max: 1.28 },
  { key: "build", label: "体格", min: 0.72, max: 1.3 },
  { key: "shoulderWidth", label: "肩幅", min: 0.72, max: 1.3 },
  { key: "hipWidth", label: "腰幅", min: 0.75, max: 1.28 },
  { key: "torsoLength", label: "胴", min: 0.78, max: 1.22 },
  { key: "armLength", label: "腕", min: 0.78, max: 1.22 },
  { key: "legLength", label: "脚", min: 0.75, max: 1.3 }
];

// Xinsir OpenPose SDXLの学習ラベルに合わせたBODY_18の接続順と線色。
// 線は公式draw_bodyposeと同じく、各BODY_18色を60%輝度で描画する。
const openPoseLinks: Array<[string, string, string]> = [
  ["neck", "rightShoulder", "#990000"],
  ["neck", "leftShoulder", "#993300"],
  ["rightShoulder", "rightElbow", "#996600"],
  ["rightElbow", "rightWrist", "#999900"],
  ["leftShoulder", "leftElbow", "#669900"],
  ["leftElbow", "leftWrist", "#339900"],
  ["neck", "rightHip", "#009900"],
  ["rightHip", "rightKnee", "#009933"],
  ["rightKnee", "rightAnkle", "#009966"],
  ["neck", "leftHip", "#009999"],
  ["leftHip", "leftKnee", "#006699"],
  ["leftKnee", "leftAnkle", "#003399"],
  ["neck", "head", "#000099"],
  ["head", "rightEye", "#330099"],
  ["rightEye", "rightEar", "#660099"],
  ["head", "leftEye", "#990099"],
  ["leftEye", "leftEar", "#990066"]
];

const openPosePointColors: Record<string, string> = {
  head: "#ff0000",
  neck: "#ff5500",
  rightShoulder: "#ffaa00",
  rightElbow: "#ffff00",
  rightWrist: "#aaff00",
  leftShoulder: "#55ff00",
  leftElbow: "#00ff00",
  leftWrist: "#00ff55",
  rightHip: "#00ffaa",
  rightKnee: "#00ffff",
  rightAnkle: "#00aaff",
  leftHip: "#0055ff",
  leftKnee: "#0000ff",
  leftAnkle: "#5500ff",
  rightEye: "#aa00ff",
  rightEar: "#ff00ff",
  leftEye: "#ff00aa",
  leftEar: "#ff0055"
};

const defaultRotations: Record<string, PoseJointRotation> = {
  hips: [0, 0, 0],
  spine: [0, 0, 0],
  chest: [0, 0, 0],
  neck: [0, 0, 0],
  head: [0, 0, 0],
  leftShoulder: [0, 0, -0.22],
  leftElbow: [0, 0, -0.08],
  leftWrist: [0, 0, 0],
  rightShoulder: [0, 0, 0.22],
  rightElbow: [0, 0, 0.08],
  rightWrist: [0, 0, 0],
  leftHip: [0, 0, -0.04],
  leftKnee: [0, 0, 0],
  leftAnkle: [0, 0, 0],
  rightHip: [0, 0, 0.04],
  rightKnee: [0, 0, 0],
  rightAnkle: [0, 0, 0]
};

const jointLimits: Record<string, JointLimit> = {
  hips: { min: [-0.55, -0.65, -0.45], max: [0.55, 0.65, 0.45] },
  spine: { min: [-0.7, -0.55, -0.5], max: [0.55, 0.55, 0.5] },
  chest: { min: [-0.65, -0.75, -0.6], max: [0.55, 0.75, 0.6] },
  neck: { min: [-0.6, -0.85, -0.55], max: [0.6, 0.85, 0.55] },
  head: { min: [-0.65, -1.05, -0.65], max: [0.65, 1.05, 0.65] },
  leftShoulder: { min: [-2.75, -1.4, -1.85], max: [2.75, 1.4, 1.2] },
  rightShoulder: { min: [-2.75, -1.4, -1.2], max: [2.75, 1.4, 1.85] },
  leftElbow: { min: [-2.65, -0.4, -0.75], max: [0.15, 0.4, 0.3] },
  rightElbow: { min: [-2.65, -0.4, -0.3], max: [0.15, 0.4, 0.75] },
  leftWrist: { min: [-0.9, -0.8, -0.8], max: [0.9, 0.8, 0.8] },
  rightWrist: { min: [-0.9, -0.8, -0.8], max: [0.9, 0.8, 0.8] },
  leftHip: { min: [-1.9, -0.8, -0.75], max: [1.15, 0.8, 0.55] },
  rightHip: { min: [-1.9, -0.8, -0.55], max: [1.15, 0.8, 0.75] },
  leftKnee: { min: [-0.08, -0.18, -0.7], max: [2.75, 0.18, 0.7] },
  rightKnee: { min: [-0.08, -0.18, -0.7], max: [2.75, 0.18, 0.7] },
  leftAnkle: { min: [-0.85, -0.4, -0.4], max: [0.65, 0.4, 0.4] },
  rightAnkle: { min: [-0.85, -0.4, -0.4], max: [0.65, 0.4, 0.4] }
};

function clampJointRotation(name: string, rotation: PoseJointRotation): PoseJointRotation {
  const limit = jointLimits[name];
  if (!limit) return rotation.map((value) => Number.isFinite(value) ? value : 0) as PoseJointRotation;
  return rotation.map((value, axis) => Math.max(limit.min[axis], Math.min(limit.max[axis], Number.isFinite(value) ? value : 0))) as PoseJointRotation;
}

function clampJointRotations(rotations: Record<string, PoseJointRotation>): Record<string, PoseJointRotation> {
  return Object.fromEntries(Object.entries(defaultRotations).map(([name, fallback]) => [
    name,
    clampJointRotation(name, rotations[name] ?? fallback)
  ]));
}

const builtInPoseTemplates: Array<{ id: string; label: string; rotations: Record<string, PoseJointRotation> }> = [
  { id: "standing", label: "直立", rotations: templateRotations({}) },
  { id: "running", label: "走る", rotations: templateRotations({
    hips: [-0.08, 0.08, 0.04], spine: [-0.22, 0, 0], chest: [-0.16, 0, 0], neck: [0.18, 0, 0], head: [0.08, 0, 0],
    leftShoulder: [-0.82, 0, -0.16], leftElbow: [-1.18, 0, -0.08], rightShoulder: [0.72, 0, 0.16], rightElbow: [-1.05, 0, 0.08],
    leftHip: [-0.92, 0, -0.05], leftKnee: [1.72, 0, 0], leftAnkle: [-0.38, 0, 0], rightHip: [0.68, 0, 0.05], rightKnee: [0.48, 0, 0], rightAnkle: [-0.18, 0, 0]
  }) },
  { id: "seiza", label: "正座", rotations: templateRotations({
    hips: [0.08, 0, 0], spine: [-0.08, 0, 0], chest: [0.04, 0, 0],
    leftHip: [-1.25, 0, -0.12], leftKnee: [2.3, 0, 0], leftAnkle: [-0.72, 0, 0],
    rightHip: [-1.25, 0, 0.12], rightKnee: [2.3, 0, 0], rightAnkle: [-0.72, 0, 0]
  }) },
  { id: "gym-sitting", label: "体育座り", rotations: templateRotations({
    hips: [0.18, 0, 0], spine: [-0.3, 0, 0], chest: [-0.22, 0, 0], neck: [0.18, 0, 0], head: [0.18, 0, 0],
    leftShoulder: [-0.42, -0.08, 0.42], leftElbow: [-1.12, 0.12, -0.28], rightShoulder: [-0.42, 0.08, -0.42], rightElbow: [-1.12, -0.12, 0.28],
    leftHip: [-1.58, 0, -0.1], leftKnee: [2.48, 0, 0.06], leftAnkle: [-0.72, 0, 0], rightHip: [-1.58, 0, 0.1], rightKnee: [2.48, 0, -0.06], rightAnkle: [-0.72, 0, 0]
  }) },
  { id: "forward-bend", label: "前屈運動", rotations: templateRotations({
    hips: [-0.22, 0, 0], spine: [-0.55, 0, 0], chest: [-0.45, 0, 0], neck: [0.35, 0, 0], head: [0.25, 0, 0],
    leftShoulder: [-1.25, 0, -0.1], leftElbow: [-0.12, 0, 0], rightShoulder: [-1.25, 0, 0.1], rightElbow: [-0.12, 0, 0]
  }) },
  { id: "back-bend", label: "体後屈", rotations: templateRotations({
    hips: [0.18, 0, 0], spine: [0.38, 0, 0], chest: [0.38, 0, 0], neck: [-0.28, 0, 0], head: [-0.22, 0, 0],
    leftShoulder: [2.35, 0, -0.18], leftElbow: [-0.18, 0, 0], rightShoulder: [2.35, 0, 0.18], rightElbow: [-0.18, 0, 0]
  }) }
];

function templateRotations(overrides: Record<string, PoseJointRotation>): Record<string, PoseJointRotation> {
  return clampJointRotations(Object.fromEntries(Object.entries({ ...defaultRotations, ...overrides }).map(([name, rotation]) => [name, [...rotation] as PoseJointRotation])));
}

export function createDefaultPoseConditioning(): PoseConditioning {
  return {
    enabled: false,
    preserveCharacterProportions: true,
    bodyShape: { ...defaultBodyShape },
    imageDataUrl: "",
    inputMode: "skeleton",
    sourceName: "3D人形を直接編集",
    controlNet: "",
    strength: 0.9,
    startPercent: 0,
    endPercent: 0.9,
    depthImageDataUrl: "",
    depthControlNet: "",
    depthStrength: 0.65,
    depthStartPercent: 0,
    depthEndPercent: 0.8,
    jointRotations: structuredClone(defaultRotations),
    cameraPosition: [0, 2.8, 8],
    cameraTarget: [0, 2.25, 0],
    cameraFov: 38,
    referenceImageDataUrl: "",
    visionModel: "",
    positivePrompt: "",
    negativePrompt: "",
    analysisStale: true
  };
}

export function PoseEditor({ value, controlNetModels, depthControlNetModels, onChange, savedPoses, selectedSavedPoseId, poseTitle, onPoseTitleChange, onSelectSavedPose, onSavePose, onDeletePose, comfyUrl, onInstallPreprocessor, outputWidth, outputHeight, disabled = false }: PoseEditorProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const valueRef = useRef(value);
  const onChangeRef = useRef(onChange);
  const applyValueRef = useRef<((next: PoseConditioning) => void) | null>(null);
  const capturePoseRef = useRef<(() => string) | null>(null);
  const captureDepthRef = useRef<(() => string) | null>(null);
  const captureReferenceRef = useRef<(() => string) | null>(null);
  const capturePromptRef = useRef<(() => DerivedPosePrompts) | null>(null);
  const [selectedJoint, setSelectedJoint] = useState("hips");
  const [selectedTemplateId, setSelectedTemplateId] = useState("");
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [catalogSource, setCatalogSource] = useState<"openposes" | "civitai">("openposes");
  const [catalogQuery, setCatalogQuery] = useState("");
  const [openPoseResults, setOpenPoseResults] = useState<OpenPoseCatalogItem[]>([]);
  const [civitaiResults, setCivitaiResults] = useState<CivitaiModelCard[]>([]);
  const [catalogBusy, setCatalogBusy] = useState(false);
  const [catalogError, setCatalogError] = useState("");
  const [sourceBusy, setSourceBusy] = useState(false);
  const [sourceError, setSourceError] = useState("");
  const [modelError, setModelError] = useState("");
  const [preprocessorStatus, setPreprocessorStatus] = useState<PosePreprocessorStatus | null>(null);

  valueRef.current = value;
  onChangeRef.current = onChange;

  useEffect(() => {
    void window.illustMK.poses.preprocessorStatus(comfyUrl).then(setPreprocessorStatus).catch(() => undefined);
  }, [comfyUrl]);

  useEffect(() => {
    applyValueRef.current?.(value);
    const frame = requestAnimationFrame(() => {
      const imageDataUrl = capturePoseRef.current?.();
      const depthImageDataUrl = captureDepthRef.current?.();
      const referenceImageDataUrl = captureReferenceRef.current?.();
      const prompts = capturePromptRef.current?.();
      if (
        (imageDataUrl && imageDataUrl !== valueRef.current.imageDataUrl) ||
        (depthImageDataUrl && depthImageDataUrl !== valueRef.current.depthImageDataUrl) ||
        (referenceImageDataUrl && referenceImageDataUrl !== valueRef.current.referenceImageDataUrl) ||
        (prompts?.positivePrompt && prompts.positivePrompt !== valueRef.current.positivePrompt) ||
        (prompts?.negativePrompt && prompts.negativePrompt !== valueRef.current.negativePrompt)
      ) {
        onChangeRef.current({
          ...valueRef.current,
          imageDataUrl: imageDataUrl || valueRef.current.imageDataUrl,
          depthImageDataUrl: depthImageDataUrl || valueRef.current.depthImageDataUrl,
          referenceImageDataUrl: referenceImageDataUrl || valueRef.current.referenceImageDataUrl,
          positivePrompt: prompts?.positivePrompt || valueRef.current.positivePrompt,
          negativePrompt: prompts?.negativePrompt || valueRef.current.negativePrompt,
          analysisStale: false
        });
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [value.jointRotations, value.cameraPosition, value.cameraTarget, value.cameraFov, value.preserveCharacterProportions, value.bodyShape]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050914);
    const camera = new THREE.PerspectiveCamera(valueRef.current.cameraFov ?? 38, 1, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xddeeff, 0x18203a, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(4, 8, 6);
    key.castShadow = true;
    scene.add(key);
    const floor = new THREE.GridHelper(16, 32, 0x28558d, 0x14213e);
    scene.add(floor);

    const rig = createRig();
    scene.add(rig.root);
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.minDistance = 5;
    orbit.maxDistance = 18;
    orbit.target.fromArray(valueRef.current.cameraTarget);
    camera.position.fromArray(valueRef.current.cameraPosition);
    orbit.update();

    const transform = new TransformControls(camera, renderer.domElement);
    transform.setMode("rotate");
    transform.setSize(0.72);
    const transformHelper = transform.getHelper();
    scene.add(transformHelper);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let applying = false;
    let rigReady = false;
    let disposed = false;
    let animationFrame = 0;

    const applyValue = (next: PoseConditioning) => {
      applying = true;
      applyBodyShape(rig, next.bodyShape ?? defaultBodyShape);
      for (const [name, joint] of rig.joints) {
        const rotation = clampJointRotation(name, next.jointRotations[name] ?? defaultRotations[name] ?? [0, 0, 0]);
        applyJointRotation(rig, name, joint, rotation);
      }
      camera.position.fromArray(next.cameraPosition);
      camera.fov = next.cameraFov ?? 38;
      camera.updateProjectionMatrix();
      orbit.target.fromArray(next.cameraTarget);
      orbit.update();
      applying = false;
      renderer.render(scene, camera);
    };
    applyValueRef.current = applyValue;

    const emitPose = () => {
      if (applying || !rigReady) return;
      scene.updateMatrixWorld(true);
      const rotations: Record<string, PoseJointRotation> = {};
      for (const [name, joint] of rig.joints) {
        const rotation = clampJointRotation(name, readJointRotation(rig, name, joint));
        applyJointRotation(rig, name, joint, rotation);
        rotations[name] = rotation;
      }
      const current = valueRef.current;
      onChangeRef.current({
        ...current,
        imageDataUrl: renderOpenPoseImage(rig, camera, current, outputWidth, outputHeight),
        depthImageDataUrl: renderDepthImage(renderer, scene, rig, floor, camera, transformHelper, outputWidth, outputHeight),
        referenceImageDataUrl: renderMannequinImage(renderer, scene, rig, camera, transformHelper, outputWidth, outputHeight),
        ...derivePosePrompts(rig, camera, rotations, current.preserveCharacterProportions ?? true, current.bodyShape ?? defaultBodyShape),
        analysisStale: false,
        jointRotations: rotations,
        cameraPosition: camera.position.toArray() as [number, number, number],
        cameraTarget: orbit.target.toArray() as [number, number, number],
        cameraFov: camera.fov
      });
    };
    capturePoseRef.current = () => {
      if (!rigReady) return valueRef.current.imageDataUrl;
      scene.updateMatrixWorld(true);
      return renderOpenPoseImage(rig, camera, valueRef.current, outputWidth, outputHeight);
    };
    captureDepthRef.current = () => {
      if (!rigReady) return valueRef.current.depthImageDataUrl ?? "";
      scene.updateMatrixWorld(true);
      return renderDepthImage(renderer, scene, rig, floor, camera, transformHelper, outputWidth, outputHeight);
    };
    captureReferenceRef.current = () => rigReady ? renderMannequinImage(renderer, scene, rig, camera, transformHelper, outputWidth, outputHeight) : valueRef.current.referenceImageDataUrl ?? "";
    capturePromptRef.current = () => derivePosePrompts(rig, camera, valueRef.current.jointRotations, valueRef.current.preserveCharacterProportions ?? true, valueRef.current.bodyShape ?? defaultBodyShape);

    const selectAtPointer = (event: PointerEvent) => {
      if (transform.dragging) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(rig.selectable, false)[0];
      const jointName = hit?.object.userData.jointName as string | undefined;
      const joint = jointName ? rig.joints.get(jointName) : null;
      if (!joint || !jointName) return;
      transform.attach(joint);
      setSelectedJoint(jointName);
    };

    const resize = () => {
      const width = Math.max(host.clientWidth, 320);
      const height = Math.max(host.clientHeight, 220);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
      emitPose();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    renderer.domElement.addEventListener("pointerdown", selectAtPointer);
    transform.addEventListener("dragging-changed", (event) => {
      orbit.enabled = !event.value;
      if (!event.value) emitPose();
    });
    transform.addEventListener("objectChange", () => {
      const jointName = transform.object?.userData.logicalJointName as string | undefined;
      if (jointName && transform.object) {
        const rotation = clampJointRotation(jointName, readJointRotation(rig, jointName, transform.object));
        applyJointRotation(rig, jointName, transform.object, rotation);
      }
      emitPose();
    });
    orbit.addEventListener("change", emitPose);
    resize();
    void loadMannequinRig(rig).then(() => {
      if (disposed) return;
      rigReady = true;
      setModelError("");
      applyValue(valueRef.current);
      const hips = rig.joints.get("hips");
      if (hips) transform.attach(hips);
      scene.updateMatrixWorld(true);
      emitPose();
    }).catch((error) => {
      if (!disposed) setModelError(`3Dデッサン人形を読み込めませんでした: ${error instanceof Error ? error.message : String(error)}`);
    });

    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      orbit.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      disposed = true;
      cancelAnimationFrame(animationFrame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", selectAtPointer);
      transform.detach();
      transform.dispose();
      orbit.dispose();
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      renderer.domElement.remove();
      applyValueRef.current = null;
      capturePoseRef.current = null;
      captureDepthRef.current = null;
      captureReferenceRef.current = null;
      capturePromptRef.current = null;
    };
  }, [outputHeight, outputWidth]);

  const resetPose = () => {
    const reset = {
      ...createDefaultPoseConditioning(),
      enabled: value.enabled,
      controlNet: value.controlNet,
      depthControlNet: value.depthControlNet,
      visionModel: value.visionModel
    };
    onChange(reset);
  };

  const resetCamera = () => {
    const defaults = createDefaultPoseConditioning();
    onChange({ ...value, cameraPosition: defaults.cameraPosition, cameraTarget: defaults.cameraTarget, cameraFov: defaults.cameraFov, analysisStale: true });
  };

  const applyBuiltInTemplate = (templateId: string) => {
    setSelectedTemplateId(templateId);
    const template = builtInPoseTemplates.find((item) => item.id === templateId);
    if (!template) return;
    onChange({
      ...value,
      jointRotations: structuredClone(template.rotations),
      inputMode: "skeleton",
      sourceName: `基本テンプレート: ${template.label}`,
      sourcePageUrl: undefined,
      analysisStale: true
    });
  };

  const refreshPreprocessorStatus = async () => {
    setPreprocessorStatus(await window.illustMK.poses.preprocessorStatus(comfyUrl));
  };

  const openCatalog = async () => {
    setCatalogOpen(true);
    setCatalogError("");
    await Promise.all([searchCatalog("openposes"), refreshPreprocessorStatus()]);
  };

  const searchCatalog = async (source = catalogSource) => {
    setCatalogBusy(true);
    setCatalogError("");
    try {
      if (source === "openposes") {
        setOpenPoseResults(await window.illustMK.poses.searchOpenPoses(catalogQuery));
      } else {
        const result = await window.illustMK.runtime.searchCivitaiPage({
          query: catalogQuery.trim() || "OpenPose",
          sort: "Most Downloaded",
          period: "AllTime",
          limit: 40,
          nsfw: false,
          type: "Poses"
        });
        setCivitaiResults(result.models);
      }
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : String(error));
    } finally {
      setCatalogBusy(false);
    }
  };

  const applyOpenPoseAsset = async (item: OpenPoseCatalogItem) => {
    setCatalogBusy(true);
    setCatalogError("");
    try {
      const asset = await window.illustMK.poses.getOpenPose(item.id);
      applyExtractedPose(asset, `OpenPoses: ${item.name}`, item.name, "https://openposes.com/");
      setCatalogOpen(false);
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : String(error));
    } finally {
      setCatalogBusy(false);
    }
  };

  const applyExtractedPose = (asset: OpenPoseAsset | PoseExtractionResult, sourceName: string, title: string, sourcePageUrl?: string) => {
    const defaults = createDefaultPoseConditioning();
    onChange({
      ...value,
      imageDataUrl: renderCatalogOpenPose(asset, outputWidth, outputHeight),
      inputMode: "skeleton",
      sourceName,
      sourcePageUrl,
      jointRotations: clampJointRotations(openPoseKeypointsToRotations(asset)),
      cameraPosition: defaults.cameraPosition,
      cameraTarget: defaults.cameraTarget,
      cameraFov: defaults.cameraFov,
      depthImageDataUrl: "",
      depthControlNet: "",
      positivePrompt: "",
      negativePrompt: "",
      analysisStale: true
    });
    onPoseTitleChange(title.replace(/\.[^.]+$/, ""));
  };

  const importLocalImage = async () => {
    setSourceBusy(true);
    setSourceError("");
    try {
      const imported = await window.illustMK.poses.importImage();
      if (!imported) return;
      const extracted = await window.illustMK.poses.extractImage({ comfyUrl, imageDataUrl: imported.imageDataUrl });
      applyExtractedPose(extracted, `画像: ${imported.name}`, imported.name);
    } catch (error) {
      setSourceError(error instanceof Error ? error.message : String(error));
    } finally {
      setSourceBusy(false);
      await refreshPreprocessorStatus();
    }
  };

  const applyCivitaiImage = async (model: CivitaiModelCard) => {
    if (!model.imageUrl) return;
    setCatalogBusy(true);
    setCatalogError("");
    try {
      const imported = await window.illustMK.poses.importRemoteImage({ url: model.imageUrl, sourcePageUrl: model.pageUrl });
      const extracted = await window.illustMK.poses.extractImage({ comfyUrl, imageDataUrl: imported.imageDataUrl });
      applyExtractedPose(extracted, `Civitai: ${model.name}`, model.name, model.pageUrl);
      setCatalogOpen(false);
    } catch (error) {
      setCatalogError(error instanceof Error ? error.message : String(error));
    } finally {
      setCatalogBusy(false);
    }
  };

  return (
    <div className="pose-editor">
      <div className="pose-library-bar">
        <div className="field">
          <label>基本テンプレート</label>
          <select value={selectedTemplateId} onChange={(event) => applyBuiltInTemplate(event.target.value)}>
            <option value="">選択してください</option>
            {builtInPoseTemplates.map((template) => <option key={template.id} value={template.id}>{template.label}</option>)}
          </select>
        </div>
        <div className="field">
          <label>タイトル</label>
          <input value={poseTitle} onChange={(event) => onPoseTitleChange(event.target.value)} placeholder="立ちポーズ、振り向きなど" />
        </div>
        <div className="field">
          <label>保存済みポーズ</label>
          <select value={selectedSavedPoseId} onChange={(event) => onSelectSavedPose(event.target.value)}>
            <option value="">新規</option>
            {savedPoses.map((pose) => <option key={pose.id} value={pose.id}>{pose.title} / {new Date(pose.updatedAt).toLocaleString("ja-JP", { hour12: false })}</option>)}
          </select>
        </div>
        <button type="button" disabled={disabled || !poseTitle.trim()} onClick={onSavePose}><Save size={16} />保存</button>
        <button type="button" className="danger icon-button" disabled={disabled || !selectedSavedPoseId} onClick={onDeletePose} title="保存ポーズを削除"><Trash2 size={16} /></button>
      </div>
      <div className="pose-source-bar">
        <button type="button" onClick={() => void openCatalog()} disabled={disabled}>
          <Images size={16} />ポーズカタログ
        </button>
        <button type="button" onClick={() => void importLocalImage()} disabled={disabled}>
          <FolderOpen size={16} />{sourceBusy ? "関節解析中" : "画像からポーズ"}
        </button>
        <span>{value.sourceName || "3D人形を直接編集"}</span>
        {value.sourcePageUrl ? (
          <button type="button" className="icon-button" onClick={() => void window.illustMK.app.openExternal(value.sourcePageUrl!)} title="素材ページを開く">
            <ExternalLink size={16} />
          </button>
        ) : null}
      </div>
      {sourceError ? <p className="pose-warning">{sourceError}</p> : null}
      {modelError ? <p className="pose-warning">{modelError}</p> : null}
      <div className="pose-body-shape-panel">
        <div className="pose-body-shape-heading">
          <strong>体型</strong>
          <button
            type="button"
            className="icon-button"
            title="体型を標準へ戻す"
            disabled={disabled}
            onClick={() => onChange({ ...value, bodyShape: { ...defaultBodyShape }, analysisStale: true })}
          >
            <RotateCcw size={14} />
          </button>
        </div>
        <div className="pose-body-shape-controls">
          {bodyShapeControls.map((control) => {
            const bodyShape = { ...defaultBodyShape, ...value.bodyShape };
            return (
              <label key={control.key}>
                <span>{control.label}<b>{Math.round(bodyShape[control.key] * 100)}%</b></span>
                <input
                  type="range"
                  min={control.min}
                  max={control.max}
                  step="0.01"
                  value={bodyShape[control.key]}
                  disabled={disabled}
                  onChange={(event) => onChange({
                    ...value,
                    bodyShape: { ...bodyShape, [control.key]: Number(event.target.value) },
                    analysisStale: true
                  })}
                />
              </label>
            );
          })}
        </div>
      </div>
      <div className="pose-viewport-wrap">
        <div className="pose-viewport" ref={hostRef} />
        <div className="pose-selection">選択中: {jointLabels[selectedJoint] ?? selectedJoint}</div>
        {value.imageDataUrl ? <img className="pose-map-preview" src={value.imageDataUrl} alt="OpenPose入力プレビュー" /> : null}
        {value.depthImageDataUrl ? <img className="depth-map-preview" src={value.depthImageDataUrl} alt="Depth入力プレビュー" /> : null}
        <label className="pose-fov-control">
          <span>画角 {Math.round(value.cameraFov ?? 38)}°</span>
          <input
            type="range"
            min="20"
            max="80"
            step="1"
            value={value.cameraFov ?? 38}
            onChange={(event) => onChange({ ...value, cameraFov: Number(event.target.value), analysisStale: true })}
            disabled={disabled}
          />
        </label>
        <button className="pose-camera-reset icon-button" type="button" onClick={resetCamera} title="視点だけリセット">
          <Camera size={16} />
        </button>
        <button className="pose-reset icon-button" type="button" onClick={resetPose} title="ポーズと視点をリセット">
          <RotateCcw size={16} />
        </button>
      </div>
      {preprocessorStatus && !preprocessorStatus.available ? (
        <div className={preprocessorStatus?.available ? "pose-preprocessor-status ready" : "pose-preprocessor-status"}>
          <span>画像からデッサン人形へ取り込むにはDWPoseが必要です。{preprocessorStatus.message}</span>
          {!preprocessorStatus?.available ? (
            <button type="button" disabled={disabled} onClick={async () => { await onInstallPreprocessor(); await refreshPreprocessorStatus(); }}>
              <Download size={16} />DWPoseをインストール
            </button>
          ) : null}
        </div>
      ) : null}
      <div className="pose-footer">
        <div className="pose-controlnet-stack">
          <div className="pose-controlnet-row">
            <div className="pose-controlnet-field">
              <label>OPENPOSE CONTROLNET</label>
              <div className="pose-controlnet-select">
                <select
                  value={value.controlNet}
                  onChange={(event) => onChange({ ...value, controlNet: event.target.value })}
                  disabled={disabled}
                >
                  <option value="">選択してください</option>
                  {controlNetModels.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </div>
            </div>
            <div className="pose-strength-field">
              <label>強度 {value.strength.toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={value.strength}
                onChange={(event) => onChange({ ...value, strength: Number(event.target.value) })}
              />
            </div>
          </div>
          <div className="pose-controlnet-row">
            <div className="pose-controlnet-field">
              <label>DEPTH CONTROLNET</label>
              <div className="pose-controlnet-select">
                <select
                  value={value.depthControlNet ?? ""}
                  onChange={(event) => onChange({ ...value, depthControlNet: event.target.value })}
                  disabled={disabled}
                >
                  <option value="">使用しない</option>
                  {depthControlNetModels.map((name) => <option key={name} value={name}>{name}</option>)}
                </select>
              </div>
            </div>
            <div className="pose-strength-field">
              <label>強度 {(value.depthStrength ?? 0.65).toFixed(2)}</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.05"
                value={value.depthStrength ?? 0.65}
                onChange={(event) => onChange({ ...value, depthStrength: Number(event.target.value) })}
                disabled={disabled || !value.depthControlNet}
              />
            </div>
          </div>
        </div>
        <button
          type="button"
          className={value.preserveCharacterProportions !== false ? "toggle-button active pose-proportion-toggle" : "toggle-button pose-proportion-toggle"}
          aria-pressed={value.preserveCharacterProportions !== false}
          disabled={disabled}
          onClick={() => onChange({ ...value, preserveCharacterProportions: value.preserveCharacterProportions === false })}
        >
          LoRA体型を優先
        </button>
        <button
          type="button"
          className={value.enabled ? "toggle-button active pose-enabled-toggle" : "toggle-button pose-enabled-toggle"}
          aria-pressed={value.enabled}
          disabled={disabled || (!value.enabled && (!value.controlNet || !value.imageDataUrl))}
          onClick={() => onChange({ ...value, enabled: !value.enabled })}
        >
          {value.enabled ? "有効" : "無効"}
        </button>
      </div>
      {!controlNetModels.length ? <p className="pose-warning">「モデル」タブからOpenPose用ControlNetモデルを導入してください。</p> : null}
      {!depthControlNetModels.length ? <p className="pose-warning">視点・画角を固定する場合は「モデル」タブからDepth用ControlNetモデルを導入してください。</p> : null}
      {catalogOpen ? (
        <div className="modal-backdrop stacked" onMouseDown={() => setCatalogOpen(false)}>
          <section className="modal-panel pose-catalog-modal" onMouseDown={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h2>ポーズカタログ</h2>
              <button type="button" className="icon-button" onClick={() => setCatalogOpen(false)} title="閉じる"><X size={18} /></button>
            </div>
            <div className="pose-catalog-tabs">
              <button type="button" className={catalogSource === "openposes" ? "active" : ""} onClick={() => { setCatalogSource("openposes"); void searchCatalog("openposes"); }}>OpenPoses</button>
              <button type="button" className={catalogSource === "civitai" ? "active" : ""} onClick={() => { setCatalogSource("civitai"); void searchCatalog("civitai"); }}>Civitai</button>
            </div>
            <div className="pose-catalog-search">
              <input value={catalogQuery} onChange={(event) => setCatalogQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.nativeEvent.isComposing) void searchCatalog(); }} placeholder={catalogSource === "openposes" ? "standing, sitting, dance..." : "OpenPose、座る、走る..."} />
              <button type="button" onClick={() => void searchCatalog()} disabled={catalogBusy}><Search size={16} />検索</button>
            </div>
            {catalogError ? <p className="pose-warning">{catalogError}</p> : null}
            <div className="pose-catalog-grid">
              {catalogSource === "openposes" ? openPoseResults.map((item) => (
                <article className="pose-catalog-card" key={item.id}>
                  <img src={item.previewUrl} alt={item.name} />
                  <strong>{item.name}</strong>
                  <small>{item.tags.join(" / ")}</small>
                  <button type="button" disabled={catalogBusy} onClick={() => void applyOpenPoseAsset(item)}>適用</button>
                </article>
              )) : civitaiResults.map((model) => (
                <article className="pose-catalog-card" key={`${model.id}-${model.versionId}`}>
                  {model.imageUrl ? <img src={model.imageUrl} alt={model.name} /> : <div className="catalog-no-image">No Image</div>}
                  <strong>{model.name}</strong>
                  <small>{model.creator} / DL {model.downloadCount.toLocaleString()}</small>
                  <div className="pose-catalog-actions">
                    <button type="button" disabled={catalogBusy || !model.imageUrl} onClick={() => void applyCivitaiImage(model)}>取り込んで適用</button>
                    <button type="button" className="icon-button" disabled={!model.pageUrl} onClick={() => void window.illustMK.app.openExternal(model.pageUrl)} title="Civitaiページを開く"><ExternalLink size={16} /></button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

type OpenPoseKeypointSource = Pick<OpenPoseAsset, "canvasWidth" | "canvasHeight" | "keypoints">;

function openPoseKeypointsToRotations(asset: OpenPoseKeypointSource): Record<string, PoseJointRotation> {
  type Point = { x: number; y: number; confidence: number };
  const point = (index: number): Point => ({
    x: Number(asset.keypoints[index * 3] ?? 0),
    y: Number(asset.keypoints[index * 3 + 1] ?? 0),
    confidence: Number(asset.keypoints[index * 3 + 2] ?? 0)
  });
  const valid = (value: Point) => value.confidence > 0.01 && Number.isFinite(value.x) && Number.isFinite(value.y);
  const midpoint = (a: Point, b: Point): Point => ({
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    confidence: Math.min(a.confidence, b.confidence)
  });
  const downAngle = (from: Point, to: Point) => Math.atan2(to.x - from.x, to.y - from.y);
  const upAngle = (from: Point, to: Point) => Math.atan2(-(to.x - from.x), -(to.y - from.y));
  const normalizeAngle = (angle: number) => Math.atan2(Math.sin(angle), Math.cos(angle));
  const rotations = structuredClone(defaultRotations);
  const setZ = (name: string, angle: number) => {
    const current = rotations[name] ?? [0, 0, 0];
    rotations[name] = [current[0], current[1], normalizeAngle(angle)];
  };

  const neck = point(1);
  const nose = point(0);
  const rightHip = point(8);
  const leftHip = point(11);
  let torsoAngle = 0;
  if (valid(neck) && valid(rightHip) && valid(leftHip)) {
    torsoAngle = upAngle(midpoint(rightHip, leftHip), neck);
    setZ("spine", torsoAngle * 0.58);
    setZ("chest", torsoAngle * 0.42);
  }
  if (valid(neck) && valid(nose)) {
    const headAngle = upAngle(neck, nose);
    const localHeadAngle = normalizeAngle(headAngle - torsoAngle);
    setZ("neck", localHeadAngle * 0.45);
    setZ("head", localHeadAngle * 0.55);
  }

  const applyChain = (jointNames: [string, string], indices: [number, number, number], parentAngle: number) => {
    const [start, middle, end] = indices.map(point);
    if (!valid(start) || !valid(middle)) return;
    const firstAngle = downAngle(start, middle);
    setZ(jointNames[0], firstAngle - parentAngle);
    if (valid(end)) setZ(jointNames[1], downAngle(middle, end) - firstAngle);
  };

  const armChains: Array<[Point, [number, number, number]]> = [
    [point(2), [2, 3, 4]],
    [point(5), [5, 6, 7]]
  ];
  armChains.sort((a, b) => a[0].x - b[0].x);
  applyChain(["leftShoulder", "leftElbow"], armChains[0][1], torsoAngle);
  applyChain(["rightShoulder", "rightElbow"], armChains[1][1], torsoAngle);

  const legChains: Array<[Point, [number, number, number]]> = [
    [point(8), [8, 9, 10]],
    [point(11), [11, 12, 13]]
  ];
  legChains.sort((a, b) => a[0].x - b[0].x);
  applyChain(["leftHip", "leftKnee"], legChains[0][1], 0);
  applyChain(["rightHip", "rightKnee"], legChains[1][1], 0);
  return rotations;
}

function renderCatalogOpenPose(asset: OpenPoseKeypointSource, outputWidth: number, outputHeight: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(256, outputWidth);
  canvas.height = Math.max(256, outputHeight);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("OpenPose画像を描画できませんでした。");
  context.fillStyle = "#000";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const scale = Math.min(canvas.width / asset.canvasWidth, canvas.height / asset.canvasHeight);
  const offsetX = (canvas.width - asset.canvasWidth * scale) / 2;
  const offsetY = (canvas.height - asset.canvasHeight * scale) / 2;
  const points = Array.from({ length: 18 }, (_, index) => ({
    x: offsetX + Number(asset.keypoints[index * 3] ?? 0) * scale,
    y: offsetY + Number(asset.keypoints[index * 3 + 1] ?? 0) * scale,
    confidence: Number(asset.keypoints[index * 3 + 2] ?? 0)
  }));
  const links: Array<[number, number]> = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]];
  const colors = ["#ff0000", "#ff5500", "#ffaa00", "#ffff00", "#aaff00", "#55ff00", "#00ff00", "#00ff55", "#00ffaa", "#00ffff", "#00aaff", "#0055ff", "#0000ff", "#5500ff", "#aa00ff", "#ff00ff", "#ff00aa", "#ff0055"];
  context.lineCap = "round";
  context.lineWidth = Math.max(4, 9 * scale);
  links.forEach(([from, to], index) => {
    if (points[from].confidence <= 0 || points[to].confidence <= 0) return;
    context.strokeStyle = colors[index % colors.length];
    context.beginPath();
    context.moveTo(points[from].x, points[from].y);
    context.lineTo(points[to].x, points[to].y);
    context.stroke();
  });
  points.forEach((point, index) => {
    if (point.confidence <= 0) return;
    context.fillStyle = colors[index % colors.length];
    context.beginPath();
    context.arc(point.x, point.y, Math.max(3, 7 * scale), 0, Math.PI * 2);
    context.fill();
  });
  return canvas.toDataURL("image/png");
}

function derivePosePrompts(
  rig: Rig,
  camera: THREE.Camera,
  rotations: Record<string, PoseJointRotation>,
  preserveCharacterProportions: boolean,
  bodyShape: PoseBodyShape
): DerivedPosePrompts {
  rig.root.updateWorldMatrix(true, true);
  camera.updateWorldMatrix(true, false);
  const position = (name: string) => rig.joints.get(name)?.getWorldPosition(new THREE.Vector3()) ?? new THREE.Vector3();
  const leftHip = rotations.leftHip ?? [0, 0, 0];
  const rightHip = rotations.rightHip ?? [0, 0, 0];
  const leftKnee = rotations.leftKnee ?? [0, 0, 0];
  const rightKnee = rotations.rightKnee ?? [0, 0, 0];
  const leftElbow = rotations.leftElbow ?? [0, 0, 0];
  const rightElbow = rotations.rightElbow ?? [0, 0, 0];
  const spinePitch = (rotations.hips?.[0] ?? 0) + (rotations.spine?.[0] ?? 0) + (rotations.chest?.[0] ?? 0);
  const hipStride = Math.abs(leftHip[0] - rightHip[0]);
  const kneeAsymmetry = Math.abs(leftKnee[0] - rightKnee[0]);
  const bothKneesBent = leftKnee[0] > 1.25 && rightKnee[0] > 1.25;
  const elbowsBent = Math.abs(leftElbow[0]) > 0.55 && Math.abs(rightElbow[0]) > 0.55;
  const leftAnkle = position("leftAnkle");
  const rightAnkle = position("rightAnkle");
  const ankleHeightDifference = Math.abs(leftAnkle.y - rightAnkle.y);

  let pose = "standing pose";
  let conflict = "sitting, lying down";
  if (bothKneesBent && leftHip[0] < -0.7 && rightHip[0] < -0.7) {
    if (spinePitch > -0.18) {
      pose = "formal seiza kneeling pose, sitting upright on both heels, knees together";
      conflict = "standing upright, legs extended, cross-legged sitting";
    } else {
      pose = "compact seated stretch, both knees drawn up toward the chest";
      conflict = "standing upright, straight legs";
    }
  } else if (spinePitch < -0.55 && leftKnee[0] < 0.55 && rightKnee[0] < 0.55) {
    pose = "standing forward bend stretch, waist folded forward, both legs straight";
    conflict = "torso leaning backward, bent knees";
  } else if (spinePitch > 0.45 && leftKnee[0] < 0.65 && rightKnee[0] < 0.65) {
    pose = "standing back bend stretch, chest lifted, torso arched backward";
    conflict = "torso bending forward, sitting";
  } else if (hipStride > 1.05 && kneeAsymmetry > 0.55) {
    pose = "dynamic running pose, one knee raised forward, opposite leg extended backward";
    conflict = "standing still, sitting, both legs straight together";
  } else if (hipStride > 0.45) {
    pose = "walking pose, legs striding in opposite directions";
    conflict = "standing still, feet together";
  } else if (bothKneesBent) {
    pose = "kneeling or compact seated pose, both knees deeply bent";
    conflict = "standing upright, straight legs";
  } else if (ankleHeightDifference > 0.38) {
    pose = "one-legged pose, one foot raised off the ground";
    conflict = "both feet planted evenly";
  }

  const details = [pose];
  if (elbowsBent) details.push("both elbows bent");
  else if (Math.abs(leftElbow[0]) > 0.55 || Math.abs(rightElbow[0]) > 0.55) details.push("one elbow bent");
  if (hipStride > 0.45) details.push("opposing arm and leg motion");
  if (spinePitch < -0.22) details.push("torso leaning forward");
  else if (spinePitch > 0.22) details.push("torso leaning backward");
  details.push(cameraViewPrompt(camera));
  details.push(faceDirectionPrompt(rig, camera));
  if (preserveCharacterProportions) {
    details.push("preserve the character's established age, stature, head-to-body ratio, and original body proportions from the character LoRA");
  } else {
    if (bodyShape.height < 0.9) details.push("short stature");
    else if (bodyShape.height > 1.1) details.push("tall stature");
    if (bodyShape.headSize > 1.1) details.push("large head-to-body ratio");
    else if (bodyShape.headSize < 0.9) details.push("small head-to-body ratio");
    if (bodyShape.build < 0.88) details.push("slender build");
    else if (bodyShape.build > 1.12) details.push("broad build");
    if (bodyShape.legLength < 0.9) details.push("short legs");
    else if (bodyShape.legLength > 1.1) details.push("long legs");
  }
  details.push("full body, anatomically coherent limb placement");

  return {
    positivePrompt: details.join(", "),
    negativePrompt: [
      conflict,
      "contradictory limb positions, disconnected limbs, twisted neck, face pointing in the wrong direction",
      preserveCharacterProportions ? "altered body proportions, elongated body, excessively tall stature, excessively long legs, stretched torso" : ""
    ].filter(Boolean).join(", ")
  };
}

function faceDirectionPrompt(rig: Rig, camera: THREE.Camera): string {
  const head = rig.joints.get("head");
  if (!head) return "face orientation consistent with head direction";
  const headPosition = head.getWorldPosition(new THREE.Vector3());
  const headRotation = head.getWorldQuaternion(new THREE.Quaternion());
  const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(headRotation).normalize();
  const toCamera = camera.position.clone().sub(headPosition).normalize();
  const facing = forward.dot(toCamera);
  if (facing > 0.48) return "face looking toward viewer, facial features on the visible front of the head";
  if (facing < -0.48) return "face looking away from viewer, back of head visible";
  return "face in side profile, facial features aligned with the head direction";
}

function cameraViewPrompt(camera: THREE.Camera): string {
  const direction = camera.position.clone();
  const horizontal = Math.hypot(direction.x, direction.z) || 1;
  const elevation = Math.atan2(direction.y - 2.25, horizontal);
  const azimuth = Math.atan2(direction.x, direction.z);
  const horizontalView = Math.abs(Math.sin(azimuth)) > 0.82
    ? "side view"
    : Math.cos(azimuth) < -0.35
      ? "back view"
      : Math.abs(Math.sin(azimuth)) > 0.35
        ? "three-quarter view"
        : "front view";
  if (elevation > 0.42) return `${horizontalView}, viewed from above`;
  if (elevation < -0.28) return `${horizontalView}, viewed from below`;
  return horizontalView;
}

function renderMannequinImage(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  rig: Rig,
  camera: THREE.Camera,
  transformHelper: THREE.Object3D,
  outputWidth: number,
  outputHeight: number
): string {
  if (!(camera instanceof THREE.PerspectiveCamera)) return "";
  const aspect = Math.max(0.35, Math.min(2.5, outputWidth / Math.max(1, outputHeight)));
  const longSide = 768;
  const width = Math.max(256, aspect >= 1 ? longSide : Math.round(longSide * aspect));
  const height = Math.max(256, aspect >= 1 ? Math.round(longSide / aspect) : longSide);
  const projectionCamera = camera.clone();
  projectionCamera.aspect = aspect;
  projectionCamera.updateProjectionMatrix();
  projectionCamera.updateMatrixWorld(true);
  const target = new THREE.WebGLRenderTarget(width, height, { format: THREE.RGBAFormat, type: THREE.UnsignedByteType, depthBuffer: true });
  target.texture.colorSpace = THREE.SRGBColorSpace;
  const wasVisible = transformHelper.visible;
  const helperVisibility = rig.helpers.map((helper) => helper.visible);
  const previousTarget = renderer.getRenderTarget();
  transformHelper.visible = false;
  rig.helpers.forEach((helper) => { helper.visible = false; });
  renderer.setRenderTarget(target);
  renderer.clear();
  renderer.render(scene, projectionCamera);
  const pixels = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
  renderer.setRenderTarget(previousTarget);
  transformHelper.visible = wasVisible;
  rig.helpers.forEach((helper, index) => { helper.visible = helperVisibility[index]; });
  renderer.render(scene, camera);
  target.dispose();
  return rgbaPixelsToDataUrl(pixels, width, height);
}

function rgbaPixelsToDataUrl(pixels: Uint8Array, width: number, height: number): string {
  const flipped = new Uint8ClampedArray(pixels.length);
  const rowSize = width * 4;
  for (let y = 0; y < height; y += 1) {
    flipped.set(pixels.subarray(y * rowSize, (y + 1) * rowSize), (height - y - 1) * rowSize);
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")?.putImageData(new ImageData(flipped, width, height), 0, 0);
  return canvas.toDataURL("image/png");
}

function renderDepthImage(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  rig: Rig,
  floor: THREE.Object3D,
  camera: THREE.Camera,
  transformHelper: THREE.Object3D,
  outputWidth: number,
  outputHeight: number
): string {
  if (!(camera instanceof THREE.PerspectiveCamera)) return "";
  const aspect = Math.max(0.35, Math.min(2.5, outputWidth / Math.max(1, outputHeight)));
  const longSide = 768;
  const width = Math.max(256, aspect >= 1 ? longSide : Math.round(longSide * aspect));
  const height = Math.max(256, aspect >= 1 ? Math.round(longSide / aspect) : longSide);
  const projectionCamera = camera.clone();
  projectionCamera.aspect = aspect;
  projectionCamera.updateProjectionMatrix();
  projectionCamera.updateMatrixWorld(true);

  rig.root.updateWorldMatrix(true, true);
  const helperVisibility = rig.helpers.map((helper) => helper.visible);
  rig.helpers.forEach((helper) => { helper.visible = false; });
  const box = new THREE.Box3().setFromObject(rig.root);
  const corners = [
    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
    new THREE.Vector3(box.min.x, box.min.y, box.max.z),
    new THREE.Vector3(box.min.x, box.max.y, box.min.z),
    new THREE.Vector3(box.min.x, box.max.y, box.max.z),
    new THREE.Vector3(box.max.x, box.min.y, box.min.z),
    new THREE.Vector3(box.max.x, box.min.y, box.max.z),
    new THREE.Vector3(box.max.x, box.max.y, box.min.z),
    new THREE.Vector3(box.max.x, box.max.y, box.max.z)
  ];
  const depths = corners.map((point) => -point.applyMatrix4(projectionCamera.matrixWorldInverse).z);
  const nearDepth = Math.max(0.01, Math.min(...depths) - 0.35);
  const farDepth = Math.max(nearDepth + 0.1, Math.max(...depths) + 0.35);
  const depthMaterial = new THREE.ShaderMaterial({
    uniforms: {
      nearDepth: { value: nearDepth },
      farDepth: { value: farDepth }
    },
    vertexShader: `
      #include <common>
      #include <batching_pars_vertex>
      #include <morphtarget_pars_vertex>
      #include <skinning_pars_vertex>
      varying float viewDepth;
      void main() {
        #include <batching_vertex>
        #include <skinbase_vertex>
        #include <begin_vertex>
        #include <morphtarget_vertex>
        #include <skinning_vertex>
        vec4 viewPosition = modelViewMatrix * vec4(transformed, 1.0);
        viewDepth = -viewPosition.z;
        gl_Position = projectionMatrix * viewPosition;
      }
    `,
    fragmentShader: `
      uniform float nearDepth;
      uniform float farDepth;
      varying float viewDepth;
      void main() {
        float depthValue = 1.0 - clamp((viewDepth - nearDepth) / (farDepth - nearDepth), 0.0, 1.0);
        gl_FragColor = vec4(vec3(depthValue), 1.0);
      }
    `
  });
  const target = new THREE.WebGLRenderTarget(width, height, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: true
  });
  const previousTarget = renderer.getRenderTarget();
  const previousOverride = scene.overrideMaterial;
  const previousBackground = scene.background;
  const floorVisible = floor.visible;
  const helperVisible = transformHelper.visible;
  floor.visible = false;
  transformHelper.visible = false;
  scene.overrideMaterial = depthMaterial;
  scene.background = new THREE.Color(0x000000);
  renderer.setRenderTarget(target);
  renderer.clear();
  renderer.render(scene, projectionCamera);
  const pixels = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
  renderer.setRenderTarget(previousTarget);
  scene.overrideMaterial = previousOverride;
  scene.background = previousBackground;
  floor.visible = floorVisible;
  transformHelper.visible = helperVisible;
  rig.helpers.forEach((helper, index) => { helper.visible = helperVisibility[index]; });
  renderer.render(scene, camera);

  target.dispose();
  depthMaterial.dispose();
  return rgbaPixelsToDataUrl(pixels, width, height);
}

function createRig(): Rig {
  return {
    root: new THREE.Group(),
    joints: new Map(),
    selectable: [],
    helpers: [],
    restQuaternions: new Map(),
    restScales: new Map(),
    restPositions: new Map()
  };
}

async function loadMannequinRig(rig: Rig): Promise<void> {
  const modelUrl = new URL("./models/human-base-rigged.glb", window.location.href).href;
  const gltf = await new GLTFLoader().loadAsync(modelUrl);
  const model = gltf.scene;
  model.name = "drawing-mannequin";
  model.traverse((object) => {
    object.castShadow = object instanceof THREE.Mesh;
    object.receiveShadow = object instanceof THREE.Mesh;
    if (object instanceof THREE.Mesh) {
      object.material = new THREE.MeshStandardMaterial({
        color: 0xb8bdc7,
        roughness: 0.72,
        metalness: 0.02,
        side: THREE.DoubleSide
      });
    }
  });
  rig.root.add(model);

  const boneMap: Record<string, string> = {
    hips: "pelvis",
    spine: "spine_01",
    chest: "spine_03",
    neck: "neck_01",
    head: "head",
    leftShoulder: "upperarm_l",
    leftElbow: "lowerarm_l",
    leftWrist: "hand_l",
    rightShoulder: "upperarm_r",
    rightElbow: "lowerarm_r",
    rightWrist: "hand_r",
    leftHip: "thigh_l",
    leftKnee: "calf_l",
    leftAnkle: "foot_l",
    rightHip: "thigh_r",
    rightKnee: "calf_r",
    rightAnkle: "foot_r"
  };
  const helperMaterial = new THREE.MeshBasicMaterial({
    color: 0x64c8ff,
    transparent: true,
    opacity: 0.12,
    depthWrite: false
  });
  for (const [logicalName, boneName] of Object.entries(boneMap)) {
    const bone = model.getObjectByName(boneName);
    if (!bone) throw new Error(`モデル内に必要な骨がありません: ${boneName}`);
    bone.userData.logicalJointName = logicalName;
    rig.joints.set(logicalName, bone);
    rig.restQuaternions.set(logicalName, bone.quaternion.clone());
    rig.restScales.set(logicalName, bone.scale.clone());
    rig.restPositions.set(logicalName, bone.position.clone());
    const helper = new THREE.Mesh(new THREE.SphereGeometry(logicalName === "hips" ? 0.075 : 0.055, 16, 12), helperMaterial);
    helper.userData.jointName = logicalName;
    helper.renderOrder = 20;
    bone.add(helper);
    rig.helpers.push(helper);
    rig.selectable.push(helper);
  }

  for (const boneName of ["clavicle_l", "clavicle_r", "spine_01", "spine_02", "spine_03", "upperarm_l", "lowerarm_l", "upperarm_r", "lowerarm_r", "thigh_l", "calf_l", "thigh_r", "calf_r", "head"]) {
    const bone = model.getObjectByName(boneName);
    if (!bone) continue;
    bone.userData.restScale = bone.scale.clone();
    bone.userData.restPosition = bone.position.clone();
  }
  rig.root.updateMatrixWorld(true);
}

function applyJointRotation(rig: Rig, name: string, joint: THREE.Object3D, rotation: PoseJointRotation): void {
  const rest = rig.restQuaternions.get(name);
  if (!rest) return;
  const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(rotation[0], rotation[1], rotation[2], "XYZ"));
  joint.quaternion.copy(rest).multiply(delta);
}

function readJointRotation(rig: Rig, name: string, joint: THREE.Object3D): PoseJointRotation {
  const rest = rig.restQuaternions.get(name);
  if (!rest) return [0, 0, 0];
  const delta = rest.clone().invert().multiply(joint.quaternion);
  const euler = new THREE.Euler().setFromQuaternion(delta, "XYZ");
  return [euler.x, euler.y, euler.z];
}

function applyBodyShape(rig: Rig, rawShape: PoseBodyShape): void {
  const shape = { ...defaultBodyShape, ...rawShape };
  const baseScale = 2.45;
  rig.root.scale.set(baseScale * shape.build, baseScale * shape.height, baseScale * shape.build);
  const model = rig.root.getObjectByName("drawing-mannequin");
  if (!model) return;
  const scaleBone = (name: string, x: number, y: number, z: number) => {
    const bone = model.getObjectByName(name);
    const rest = bone?.userData.restScale as THREE.Vector3 | undefined;
    if (bone && rest) bone.scale.set(rest.x * x, rest.y * y, rest.z * z);
  };
  const moveBone = (name: string, x: number, y: number, z: number) => {
    const bone = model.getObjectByName(name);
    const rest = bone?.userData.restPosition as THREE.Vector3 | undefined;
    if (bone && rest) bone.position.set(rest.x * x, rest.y * y, rest.z * z);
  };

  scaleBone("head", shape.headSize / shape.build, shape.headSize / shape.height, shape.headSize / shape.build);
  for (const name of ["spine_01", "spine_02", "spine_03"]) scaleBone(name, 1, shape.torsoLength, 1);
  for (const name of ["clavicle_l", "clavicle_r"]) scaleBone(name, 1, shape.shoulderWidth, 1);
  for (const name of ["upperarm_l", "lowerarm_l", "upperarm_r", "lowerarm_r"]) scaleBone(name, 1, shape.armLength, 1);
  for (const name of ["thigh_l", "calf_l", "thigh_r", "calf_r"]) scaleBone(name, 1, shape.legLength, 1);
  moveBone("thigh_l", shape.hipWidth, 1, 1);
  moveBone("thigh_r", shape.hipWidth, 1, 1);
  rig.root.updateMatrixWorld(true);
}

function createDepthProxy(rig: Rig): THREE.Group {
  const proxy = new THREE.Group();
  proxy.name = "depth-proxy";
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const position = (name: string) => rig.joints.get(name)?.getWorldPosition(new THREE.Vector3()) ?? new THREE.Vector3();
  const orientation = (name: string) => rig.joints.get(name)?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion();

  const addCapsule = (from: THREE.Vector3, to: THREE.Vector3, radius: number, overlap = radius * 0.7) => {
    const direction = new THREE.Vector3().subVectors(to, from);
    const distance = direction.length();
    if (distance < 0.001) return;
    const totalLength = distance + overlap * 2;
    const mesh = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius, Math.max(0.01, totalLength - radius * 2), 10, 20),
      material
    );
    mesh.position.copy(from).add(to).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
    proxy.add(mesh);
  };
  const addEllipsoid = (center: THREE.Vector3, scale: [number, number, number], quaternion?: THREE.Quaternion) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 28, 20), material);
    mesh.position.copy(center);
    mesh.scale.set(...scale);
    if (quaternion) mesh.quaternion.copy(quaternion);
    proxy.add(mesh);
  };

  const hips = position("hips");
  const spine = position("spine");
  const chest = position("chest");
  const neck = position("neck");
  const head = position("head");
  addCapsule(hips, spine, 0.29, 0.18);
  addCapsule(spine, chest, 0.3, 0.2);
  addEllipsoid(hips, [0.42, 0.32, 0.29], orientation("hips"));
  addEllipsoid(chest, [0.42, 0.43, 0.27], orientation("chest"));
  addCapsule(chest, neck, 0.18, 0.12);
  addCapsule(neck, head, 0.145, 0.1);
  const headOrientation = orientation("head");
  addEllipsoid(head, [0.29, 0.4, 0.3], headOrientation);
  const faceCenter = new THREE.Vector3(0, 0, 0.25).applyQuaternion(headOrientation).add(head);
  addEllipsoid(faceCenter, [0.215, 0.285, 0.06], headOrientation);
  const noseEnd = new THREE.Vector3(0, -0.01, 0.4).applyQuaternion(headOrientation).add(head);
  addCapsule(faceCenter, noseEnd, 0.045, 0.02);

  for (const side of ["left", "right"] as const) {
    const shoulder = position(`${side}Shoulder`);
    const elbow = position(`${side}Elbow`);
    const wrist = position(`${side}Wrist`);
    const hip = position(`${side}Hip`);
    const knee = position(`${side}Knee`);
    const ankle = position(`${side}Ankle`);
    addCapsule(chest, shoulder, 0.16, 0.14);
    addCapsule(shoulder, elbow, 0.125, 0.1);
    addCapsule(elbow, wrist, 0.105, 0.09);
    addEllipsoid(wrist, [0.1, 0.2, 0.08], orientation(`${side}Wrist`));
    addCapsule(hip, knee, 0.165, 0.12);
    addCapsule(knee, ankle, 0.135, 0.11);
    const footEnd = new THREE.Vector3(0, -0.08, 0.34)
      .applyQuaternion(orientation(`${side}Ankle`))
      .add(ankle);
    addCapsule(ankle, footEnd, 0.14, 0.08);
  }

  return proxy;
}

function disposeDepthProxy(proxy: THREE.Object3D): void {
  const materials = new Set<THREE.Material>();
  proxy.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.geometry.dispose();
    const objectMaterials = Array.isArray(object.material) ? object.material : [object.material];
    objectMaterials.forEach((material) => materials.add(material));
  });
  materials.forEach((material) => material.dispose());
}

function createProceduralRig(): Rig {
  const root = new THREE.Group();
  root.position.y = 0;
  const joints = new Map<string, THREE.Group>();
  const selectable: THREE.Object3D[] = [];
  const surface = new THREE.MeshStandardMaterial({ color: 0xb9bdc6, roughness: 0.64, metalness: 0.03 });
  const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xd2d4da, roughness: 0.6 });
  const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0x343944, roughness: 0.5 });

  const joint = (name: string, parent: THREE.Object3D, position: [number, number, number]) => {
    const group = new THREE.Group();
    group.name = name;
    group.position.set(...position);
    parent.add(group);
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.115, 22, 16), jointMaterial);
    marker.userData.jointName = name;
    marker.castShadow = true;
    group.add(marker);
    joints.set(name, group);
    selectable.push(marker);
    return group;
  };
  const segment = (name: string, parent: THREE.Group, length: number, radius: number) => {
    const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(radius, Math.max(0.04, length - radius * 2), 12, 24), surface);
    mesh.position.y = -length / 2;
    mesh.userData.jointName = name;
    mesh.castShadow = true;
    parent.add(mesh);
    selectable.push(mesh);
    return mesh;
  };
  const ellipsoid = (
    name: string,
    parent: THREE.Object3D,
    position: [number, number, number],
    scale: [number, number, number],
    material: THREE.Material = surface
  ) => {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), material);
    mesh.position.set(...position);
    mesh.scale.set(...scale);
    mesh.userData.jointName = name;
    mesh.castShadow = true;
    parent.add(mesh);
    selectable.push(mesh);
    return mesh;
  };

  const hips = joint("hips", root, [0, 2.15, 0]);
  ellipsoid("hips", hips, [0, 0.02, 0], [0.42, 0.31, 0.29]);
  ellipsoid("hips", hips, [0, 0.27, 0], [0.3, 0.31, 0.23]);

  const spine = joint("spine", hips, [0, 0.38, 0]);
  ellipsoid("spine", spine, [0, 0.25, 0], [0.3, 0.48, 0.23]);
  const chest = joint("chest", spine, [0, 0.65, 0]);
  ellipsoid("chest", chest, [0, 0.06, 0], [0.42, 0.43, 0.26]);
  ellipsoid("chest", chest, [0, -0.2, 0], [0.34, 0.3, 0.23]);
  const neck = joint("neck", chest, [0, 0.4, 0]);
  ellipsoid("neck", neck, [0, 0.1, 0], [0.14, 0.24, 0.14]);
  const head = joint("head", neck, [0, 0.38, 0]);
  ellipsoid("head", head, [0, 0.02, 0], [0.29, 0.4, 0.3]);
  const facePlane = ellipsoid("head", head, [0, -0.01, 0.245], [0.215, 0.285, 0.055]);
  facePlane.scale.z = 0.055;
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.13, 16), surface);
  nose.rotation.x = Math.PI / 2;
  nose.position.set(0, -0.015, 0.355);
  nose.userData.jointName = "head";
  nose.castShadow = true;
  head.add(nose);
  selectable.push(nose);
  for (const side of [-1, 1] as const) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.02, 14, 10), eyeMaterial);
    eye.position.set(side * 0.095, 0.075, 0.304);
    eye.userData.jointName = "head";
    head.add(eye);
    selectable.push(eye);
    ellipsoid("head", head, [side * 0.285, 0.025, 0], [0.045, 0.09, 0.06]);
  }

  for (const side of [-1, 1] as const) {
    const prefix = side < 0 ? "left" : "right";
    const shoulder = joint(`${prefix}Shoulder`, chest, [side * 0.43, 0.14, 0]);
    ellipsoid(`${prefix}Shoulder`, shoulder, [0, -0.05, 0], [0.15, 0.18, 0.15]);
    segment(`${prefix}Shoulder`, shoulder, 0.74, 0.12);
    const elbow = joint(`${prefix}Elbow`, shoulder, [0, -0.72, 0]);
    segment(`${prefix}Elbow`, elbow, 0.66, 0.1);
    const wrist = joint(`${prefix}Wrist`, elbow, [0, -0.64, 0]);
    ellipsoid(`${prefix}Wrist`, wrist, [0, -0.16, 0.025], [0.095, 0.2, 0.075]);

    const hip = joint(`${prefix}Hip`, hips, [side * 0.25, -0.15, 0]);
    segment(`${prefix}Hip`, hip, 0.9, 0.155);
    const knee = joint(`${prefix}Knee`, hip, [0, -0.88, 0]);
    segment(`${prefix}Knee`, knee, 0.84, 0.125);
    const ankle = joint(`${prefix}Ankle`, knee, [0, -0.82, 0]);
    const foot = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.32, 12, 20), surface);
    foot.rotation.x = Math.PI / 2;
    foot.position.set(0, -0.08, 0.22);
    foot.userData.jointName = `${prefix}Ankle`;
    foot.castShadow = true;
    ankle.add(foot);
    selectable.push(foot);
  }

  return {
    root,
    joints,
    selectable,
    helpers: [],
    restQuaternions: new Map(Array.from(joints, ([name, object]) => [name, object.quaternion.clone()])),
    restScales: new Map(Array.from(joints, ([name, object]) => [name, object.scale.clone()])),
    restPositions: new Map(Array.from(joints, ([name, object]) => [name, object.position.clone()]))
  };
}

function renderOpenPoseImage(
  rig: Rig,
  camera: THREE.Camera,
  value: PoseConditioning,
  outputWidth: number,
  outputHeight: number
): string {
  const aspect = Math.max(0.35, Math.min(2.5, outputWidth / Math.max(1, outputHeight)));
  const projectionCamera = camera instanceof THREE.PerspectiveCamera ? camera.clone() : camera;
  if (projectionCamera instanceof THREE.PerspectiveCamera) {
    projectionCamera.aspect = aspect;
    projectionCamera.updateProjectionMatrix();
    projectionCamera.updateMatrixWorld(true);
  }
  const longSide = 768;
  const width = aspect >= 1 ? longSide : Math.round(longSide * aspect);
  const height = aspect >= 1 ? Math.round(longSide / aspect) : longSide;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(256, width);
  canvas.height = Math.max(256, height);
  const context = canvas.getContext("2d");
  if (!context) return value.imageDataUrl;
  context.fillStyle = "#000000";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const points = new Map<string, { x: number; y: number; visible: boolean }>();
  const projectPoint = (name: string, worldPosition: THREE.Vector3) => {
    const projected = worldPosition.project(projectionCamera);
    points.set(name, {
      x: (projected.x * 0.5 + 0.5) * canvas.width,
      y: (-projected.y * 0.5 + 0.5) * canvas.height,
      visible: projected.z >= -1 && projected.z <= 1
    });
  };
  for (const [name, object] of rig.joints) {
    projectPoint(name, object.getWorldPosition(new THREE.Vector3()));
  }
  const headJoint = rig.joints.get("head");
  if (headJoint) {
    const headPosition = headJoint.getWorldPosition(new THREE.Vector3());
    const headOrientation = headJoint.getWorldQuaternion(new THREE.Quaternion());
    const faceForward = new THREE.Vector3(0, 0, 1).applyQuaternion(headOrientation).normalize();
    const toCamera = projectionCamera.position.clone().sub(headPosition).normalize();
    const facing = faceForward.dot(toCamera);
    const facialPoint = (name: string, local: [number, number, number]) => {
      projectPoint(name, new THREE.Vector3(...local).applyQuaternion(headOrientation).add(headPosition));
    };
    points.delete("head");
    if (facing > -0.25) {
      facialPoint("head", [0, 0.02, 0.39]);
      if (facing > 0.28) {
        facialPoint("rightEye", [-0.1, 0.08, 0.32]);
        facialPoint("leftEye", [0.1, 0.08, 0.32]);
        facialPoint("rightEar", [-0.29, 0.04, 0]);
        facialPoint("leftEar", [0.29, 0.04, 0]);
      } else {
        const headRight = new THREE.Vector3(1, 0, 0).applyQuaternion(headOrientation).normalize();
        const rightSideVisible = headRight.dot(toCamera) > 0;
        facialPoint(rightSideVisible ? "rightEye" : "leftEye", [rightSideVisible ? -0.1 : 0.1, 0.08, 0.32]);
        facialPoint(rightSideVisible ? "rightEar" : "leftEar", [rightSideVisible ? -0.29 : 0.29, 0.04, 0]);
      }
    }
  }
  context.lineCap = "round";
  context.lineJoin = "round";
  const maxSide = Math.max(canvas.width, canvas.height);
  const xinsirScale = maxSide < 500 ? 1 : maxSide < 1000 ? 2 : Math.min(7, Math.floor(maxSide / 1000) + 2);
  // 公式実装のellipse半径(stickwidth * scale)をCanvasの直径へ換算する。
  context.lineWidth = 8 * xinsirScale;
  for (const [from, to, color] of openPoseLinks) {
    const a = points.get(from);
    const b = points.get(to);
    if (!a?.visible || !b?.visible) continue;
    context.strokeStyle = color;
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.stroke();
  }
  const radius = 4 * xinsirScale;
  for (const [name, color] of Object.entries(openPosePointColors)) {
    const point = points.get(name);
    if (!point?.visible) continue;
    context.fillStyle = color;
    context.beginPath();
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
  }
  return canvas.toDataURL("image/png");
}
