import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { Bot, Camera, RotateCcw, Save, Send, Trash2 } from "lucide-react";
import type { PoseConditioning, PoseJointRotation, SavedPose } from "../../main/runtime";

type PoseEditorProps = {
  value: PoseConditioning;
  controlNetModels: string[];
  onChange: (value: PoseConditioning) => void;
  savedPoses: SavedPose[];
  selectedSavedPoseId: string;
  poseTitle: string;
  onPoseTitleChange: (title: string) => void;
  onSelectSavedPose: (id: string) => void;
  onSavePose: () => void;
  onDeletePose: () => void;
  onAssistPose: (instruction: string) => Promise<string>;
  outputWidth: number;
  outputHeight: number;
  disabled?: boolean;
};

type Rig = {
  root: THREE.Group;
  joints: Map<string, THREE.Group>;
  selectable: THREE.Object3D[];
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

// OpenPose BODY_18の接続順と配色。headは顔の代表点（nose）として扱う。
const openPoseLinks: Array<[string, string, string]> = [
  ["neck", "rightShoulder", "#ff0000"],
  ["rightShoulder", "rightElbow", "#ff5500"],
  ["rightElbow", "rightWrist", "#ffaa00"],
  ["neck", "leftShoulder", "#ffff00"],
  ["leftShoulder", "leftElbow", "#aaff00"],
  ["leftElbow", "leftWrist", "#55ff00"],
  ["neck", "rightHip", "#00ff55"],
  ["rightHip", "rightKnee", "#00ffaa"],
  ["rightKnee", "rightAnkle", "#00ffff"],
  ["neck", "leftHip", "#00aaff"],
  ["leftHip", "leftKnee", "#0055ff"],
  ["leftKnee", "leftAnkle", "#0000ff"],
  ["neck", "head", "#aa00ff"]
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
  leftAnkle: "#5500ff"
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

export function createDefaultPoseConditioning(): PoseConditioning {
  return {
    enabled: false,
    imageDataUrl: "",
    controlNet: "",
    strength: 0.9,
    startPercent: 0,
    endPercent: 0.9,
    jointRotations: structuredClone(defaultRotations),
    cameraPosition: [0, 4.4, 10],
    cameraTarget: [0, 3.8, 0]
  };
}

export function PoseEditor({ value, controlNetModels, onChange, savedPoses, selectedSavedPoseId, poseTitle, onPoseTitleChange, onSelectSavedPose, onSavePose, onDeletePose, onAssistPose, outputWidth, outputHeight, disabled = false }: PoseEditorProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const valueRef = useRef(value);
  const onChangeRef = useRef(onChange);
  const applyValueRef = useRef<((next: PoseConditioning) => void) | null>(null);
  const capturePoseRef = useRef<(() => string) | null>(null);
  const [selectedJoint, setSelectedJoint] = useState("hips");
  const [assistantInstruction, setAssistantInstruction] = useState("");
  const [assistantMessage, setAssistantMessage] = useState("");
  const [assistantBusy, setAssistantBusy] = useState(false);

  valueRef.current = value;
  onChangeRef.current = onChange;

  useEffect(() => {
    applyValueRef.current?.(value);
    const frame = requestAnimationFrame(() => {
      const imageDataUrl = capturePoseRef.current?.();
      if (imageDataUrl && imageDataUrl !== valueRef.current.imageDataUrl) {
        onChangeRef.current({ ...valueRef.current, imageDataUrl });
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [value.jointRotations, value.cameraPosition, value.cameraTarget]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050914);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
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
    scene.add(transform.getHelper());
    transform.attach(rig.joints.get("hips")!);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let applying = false;
    let animationFrame = 0;

    const applyValue = (next: PoseConditioning) => {
      applying = true;
      for (const [name, joint] of rig.joints) {
        const rotation = next.jointRotations[name] ?? defaultRotations[name] ?? [0, 0, 0];
        joint.rotation.set(rotation[0], rotation[1], rotation[2]);
      }
      camera.position.fromArray(next.cameraPosition);
      orbit.target.fromArray(next.cameraTarget);
      orbit.update();
      applying = false;
      renderer.render(scene, camera);
    };
    applyValueRef.current = applyValue;

    const emitPose = () => {
      if (applying) return;
      scene.updateMatrixWorld(true);
      const rotations: Record<string, PoseJointRotation> = {};
      for (const [name, joint] of rig.joints) {
        rotations[name] = [joint.rotation.x, joint.rotation.y, joint.rotation.z];
      }
      const current = valueRef.current;
      onChangeRef.current({
        ...current,
        imageDataUrl: renderOpenPoseImage(rig, camera, current, outputWidth, outputHeight),
        jointRotations: rotations,
        cameraPosition: camera.position.toArray() as [number, number, number],
        cameraTarget: orbit.target.toArray() as [number, number, number]
      });
    };
    capturePoseRef.current = () => {
      scene.updateMatrixWorld(true);
      return renderOpenPoseImage(rig, camera, valueRef.current, outputWidth, outputHeight);
    };

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
    transform.addEventListener("objectChange", emitPose);
    orbit.addEventListener("change", emitPose);
    applyValue(valueRef.current);
    resize();

    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      orbit.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
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
    };
  }, [outputHeight, outputWidth]);

  const resetPose = () => {
    const reset = { ...createDefaultPoseConditioning(), enabled: value.enabled, controlNet: value.controlNet };
    onChange(reset);
  };

  const resetCamera = () => {
    const defaults = createDefaultPoseConditioning();
    onChange({ ...value, cameraPosition: defaults.cameraPosition, cameraTarget: defaults.cameraTarget });
  };

  const runAssistant = async () => {
    const instruction = assistantInstruction.trim();
    if (!instruction || assistantBusy || disabled) return;
    setAssistantBusy(true);
    setAssistantMessage("");
    try {
      setAssistantMessage(await onAssistPose(instruction));
    } catch (error) {
      setAssistantMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setAssistantBusy(false);
    }
  };

  return (
    <div className="pose-editor">
      <div className="pose-library-bar">
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
      <div className="pose-assistant-bar">
        <Bot size={17} />
        <input
          value={assistantInstruction}
          onChange={(event) => setAssistantInstruction(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.nativeEvent.isComposing) void runAssistant();
          }}
          placeholder="体育座り、走っているポーズなど"
          disabled={disabled || assistantBusy}
        />
        <button type="button" className="icon-button" onClick={() => void runAssistant()} disabled={disabled || assistantBusy || !assistantInstruction.trim()} title="LLMでポーズを設定">
          <Send size={16} />
        </button>
        {assistantMessage ? <span title={assistantMessage}>{assistantMessage}</span> : null}
      </div>
      <div className="pose-viewport-wrap">
        <div className="pose-viewport" ref={hostRef} />
        <div className="pose-selection">選択中: {jointLabels[selectedJoint] ?? selectedJoint}</div>
        {value.imageDataUrl ? <img className="pose-map-preview" src={value.imageDataUrl} alt="OpenPose入力プレビュー" /> : null}
        <button className="pose-camera-reset icon-button" type="button" onClick={resetCamera} title="視点だけリセット">
          <Camera size={16} />
        </button>
        <button className="pose-reset icon-button" type="button" onClick={resetPose} title="ポーズと視点をリセット">
          <RotateCcw size={16} />
        </button>
      </div>
      <div className="pose-footer">
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
    </div>
  );
}

function createRig(): Rig {
  const root = new THREE.Group();
  root.position.y = 0;
  const joints = new Map<string, THREE.Group>();
  const selectable: THREE.Object3D[] = [];
  const skin = new THREE.MeshStandardMaterial({ color: 0xd9b69c, roughness: 0.72 });
  const body = new THREE.MeshStandardMaterial({ color: 0x78a8d8, roughness: 0.62 });
  const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xf3d6bf, roughness: 0.65 });

  const joint = (name: string, parent: THREE.Object3D, position: [number, number, number]) => {
    const group = new THREE.Group();
    group.name = name;
    group.position.set(...position);
    parent.add(group);
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.12, 18, 14), jointMaterial);
    marker.userData.jointName = name;
    marker.castShadow = true;
    group.add(marker);
    joints.set(name, group);
    selectable.push(marker);
    return group;
  };
  const segment = (name: string, parent: THREE.Group, length: number, radius: number, material = skin) => {
    const mesh = new THREE.Mesh(new THREE.CapsuleGeometry(radius, Math.max(0.04, length - radius * 2), 8, 16), material);
    mesh.position.y = -length / 2;
    mesh.userData.jointName = name;
    mesh.castShadow = true;
    parent.add(mesh);
    selectable.push(mesh);
  };

  const hips = joint("hips", root, [0, 4.25, 0]);
  const pelvis = new THREE.Mesh(new THREE.CapsuleGeometry(0.34, 0.42, 8, 18), body);
  pelvis.rotation.z = Math.PI / 2;
  pelvis.userData.jointName = "hips";
  hips.add(pelvis);
  selectable.push(pelvis);

  const spine = joint("spine", hips, [0, 0.52, 0]);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.36, 0.72, 8, 18), body);
  torso.position.y = 0.38;
  torso.userData.jointName = "spine";
  torso.castShadow = true;
  spine.add(torso);
  selectable.push(torso);
  const chest = joint("chest", spine, [0, 0.86, 0]);
  const neck = joint("neck", chest, [0, 0.55, 0]);
  segment("neck", neck, 0.24, 0.11, skin);
  const head = joint("head", neck, [0, 0.48, 0]);
  const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.34, 24, 18), skin);
  headMesh.scale.set(0.86, 1.12, 0.92);
  headMesh.userData.jointName = "head";
  headMesh.castShadow = true;
  head.add(headMesh);
  selectable.push(headMesh);

  for (const side of [-1, 1] as const) {
    const prefix = side < 0 ? "left" : "right";
    const shoulder = joint(`${prefix}Shoulder`, chest, [side * 0.56, 0.34, 0]);
    segment(`${prefix}Shoulder`, shoulder, 0.9, 0.13);
    const elbow = joint(`${prefix}Elbow`, shoulder, [0, -0.9, 0]);
    segment(`${prefix}Elbow`, elbow, 0.82, 0.11);
    const wrist = joint(`${prefix}Wrist`, elbow, [0, -0.82, 0]);
    segment(`${prefix}Wrist`, wrist, 0.28, 0.1);

    const hip = joint(`${prefix}Hip`, hips, [side * 0.28, -0.22, 0]);
    segment(`${prefix}Hip`, hip, 1.25, 0.18);
    const knee = joint(`${prefix}Knee`, hip, [0, -1.25, 0]);
    segment(`${prefix}Knee`, knee, 1.22, 0.15);
    const ankle = joint(`${prefix}Ankle`, knee, [0, -1.22, 0]);
    const foot = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.34, 8, 14), skin);
    foot.rotation.x = Math.PI / 2;
    foot.position.set(0, -0.08, 0.2);
    foot.userData.jointName = `${prefix}Ankle`;
    foot.castShadow = true;
    ankle.add(foot);
    selectable.push(foot);
  }

  return { root, joints, selectable };
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
  for (const [name, object] of rig.joints) {
    const projected = object.getWorldPosition(new THREE.Vector3()).project(projectionCamera);
    points.set(name, {
      x: (projected.x * 0.5 + 0.5) * canvas.width,
      y: (-projected.y * 0.5 + 0.5) * canvas.height,
      visible: projected.z >= -1 && projected.z <= 1
    });
  }
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = Math.max(4, Math.round(Math.min(canvas.width, canvas.height) * 0.012));
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
  const radius = Math.max(4, Math.round(Math.min(canvas.width, canvas.height) * 0.01));
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
