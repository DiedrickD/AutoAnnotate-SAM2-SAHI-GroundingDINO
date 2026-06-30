"use client";

import { useMemo, useRef, useState } from "react";
import {
  FiActivity,
  FiAperture,
  FiDownload,
  FiFilm,
  FiGrid,
  FiImage,
  FiLoader,
  FiMoon,
  FiPlus,
  FiSquare,
  FiSun,
  FiTag,
  FiTarget,
  FiTrash2,
  FiUploadCloud,
  FiVideo,
  FiZap,
} from "react-icons/fi";

const models = [
  {
    id: "facebook/sam2-hiera-tiny",
    name: "SAM 2 Tiny",
    detail: "Best fit for 6 GB GPU",
  },
  {
    id: "facebook/sam2.1-hiera-base-plus",
    name: "SAM 2.1 Base+",
    detail: "Sharper masks, more memory",
  },
];

const detectorModels = [
  {
    id: "dino-tiny",
    detector: "dino",
    dinoModel: "IDEA-Research/grounding-dino-tiny",
    name: "DINO Tiny",
    detail: "Open-vocab, lower VRAM",
  },
  {
    id: "dino-base",
    detector: "dino",
    dinoModel: "IDEA-Research/grounding-dino-base",
    name: "DINO Base",
    detail: "Stronger open-vocab detector",
  },
  {
    id: "yolo11x-vehicle",
    detector: "yolo",
    dinoModel: "IDEA-Research/grounding-dino-tiny",
    name: "YOLO11x Vehicle",
    detail: "Strong COCO vehicle boxes",
  },
];

const outputModes = [
  {
    id: "mask-box-label",
    name: "Mask + label",
    detail: "SAM mask, box, text",
    useSam: true,
    showBoxes: true,
    showLabels: true,
  },
  {
    id: "mask-box",
    name: "Mask + box",
    detail: "SAM mask, clean box",
    useSam: true,
    showBoxes: true,
    showLabels: false,
  },
  {
    id: "mask",
    name: "Mask only",
    detail: "SAM mask, no box",
    useSam: true,
    showBoxes: false,
    showLabels: false,
  },
  {
    id: "box-label",
    name: "Box + label",
    detail: "DINO box, text",
    useSam: false,
    showBoxes: true,
    showLabels: true,
  },
  {
    id: "box",
    name: "Box only",
    detail: "DINO box, no text",
    useSam: false,
    showBoxes: true,
    showLabels: false,
  },
];

export default function Home() {
  const [theme, setTheme] = useState("dark");
  const [file, setFile] = useState(null);
  const [mediaType, setMediaType] = useState("");
  const [preview, setPreview] = useState("");
  const [prompt, setPrompt] = useState("I want to segment the pink flower");
  const [classPrompts, setClassPrompts] = useState(["car", "motorcycle"]);
  const [multipleEnabled, setMultipleEnabled] = useState(false);
  const [sahiEnabled, setSahiEnabled] = useState(false);
  const [everythingEnabled, setEverythingEnabled] = useState(false);
  const [outputMode, setOutputMode] = useState(outputModes[0].id);
  const [samModel, setSamModel] = useState(models[0].id);
  const [detectorModel, setDetectorModel] = useState(detectorModels[0].id);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingMode, setLoadingMode] = useState("");
  const [progress, setProgress] = useState(null);
  const [activeJobId, setActiveJobId] = useState("");
  const [stopping, setStopping] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const activeModel = useMemo(() => models.find((model) => model.id === samModel), [samModel]);
  const activeDetectorModel = useMemo(
    () => detectorModels.find((model) => model.id === detectorModel) || detectorModels[0],
    [detectorModel],
  );
  const activeOutputMode = useMemo(
    () => outputModes.find((mode) => mode.id === outputMode) || outputModes[0],
    [outputMode],
  );
  const useSam = activeOutputMode.useSam;
  const showBoxes = activeOutputMode.showBoxes;
  const showLabels = activeOutputMode.showLabels;
  const isVideoResult = result?.mediaType === "video" || result?.metadata?.media_type === "video";
  const boxOnlyResult = Boolean(result?.metadata?.box_only || result?.metadata?.sam_enabled === false);
  const cleanBoxResult = boxOnlyResult && result?.metadata?.show_labels === false;
  const resultLabel = result?.metadata?.detection_label || result?.metadata?.detector_prompt || "Detected target";
  const dinoScore = Number(result?.metadata?.detection_score);
  const samScore = Number(result?.metadata?.sam2_mask_score);
  const detectorScoreLabel = result?.metadata?.detector === "yolo" ? "YOLO" : "DINO";
  const isEverythingLoading = loading && loadingMode === "everything";
  const isSahiLoading = loading && (loadingMode === "sahi" || loadingMode === "multiSahi");
  const isMultiClassLoading = loading && (loadingMode === "multiClass" || loadingMode === "multiSahi");
  const isMultiSahiLoading = loading && loadingMode === "multiSahi";
  const detectionMode = multipleEnabled ? "multiple" : everythingEnabled ? "everything" : "single";
  const fileKindLabel = mediaType ? mediaType.charAt(0).toUpperCase() + mediaType.slice(1) : "Media";
  const fileSizeLabel = file ? formatFileSize(file.size) : "";
  const fileSummary = file ? `${fileKindLabel} | ${fileSizeLabel}` : "PNG, JPG, WEBP, MP4, MOV, WEBM, AVI, MKV";

  function getMediaType(nextFile) {
    if (nextFile?.type?.startsWith("video/")) return "video";
    if (nextFile?.type?.startsWith("image/")) return "image";
    return /\.(mp4|mov|webm|avi|mkv)$/i.test(nextFile?.name || "") ? "video" : "image";
  }

  function scoreText(score) {
    return Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : "N/A";
  }

  function formatFileSize(size = 0) {
    if (!Number.isFinite(size) || size <= 0) return "Size unknown";

    const units = ["B", "KB", "MB", "GB"];
    let value = size;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex += 1;
    }

    const precision = value >= 10 || unitIndex === 0 ? 0 : 1;
    return `${value.toFixed(precision)} ${units[unitIndex]}`;
  }

  function outputModeIcon(mode) {
    if (!mode.useSam) return mode.showLabels ? <FiTag /> : <FiSquare />;
    if (!mode.showBoxes) return <FiAperture />;
    return mode.showLabels ? <FiTag /> : <FiSquare />;
  }

  function cleanClassPrompts(values = classPrompts) {
    const seen = new Set();
    const cleaned = [];
    for (const value of values) {
      const prompt = String(value || "").trim();
      const key = prompt.toLowerCase();
      if (!prompt || seen.has(key)) continue;
      seen.add(key);
      cleaned.push(prompt);
      if (cleaned.length >= 12) break;
    }
    return cleaned;
  }

  function updateClassPrompt(index, value) {
    setClassPrompts((current) => current.map((item, itemIndex) => (itemIndex === index ? value : item)));
  }

  function addClassPrompt() {
    setClassPrompts((current) => (current.length >= 12 ? current : [...current, ""]));
  }

  function removeClassPrompt(index) {
    setClassPrompts((current) => (current.length <= 1 ? [""] : current.filter((_, itemIndex) => itemIndex !== index)));
  }

  function setDetectionMode(mode) {
    setMultipleEnabled(mode === "multiple");
    setEverythingEnabled(mode === "everything");
  }

  function handleDropzoneKeyDown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      inputRef.current?.click();
    }
  }

  function selectedModeLabel() {
    const outputName = activeOutputMode.name;
    if (mediaType === "video") return `Video ${outputName}`;
    if (multipleEnabled && sahiEnabled) return `Multiple + SAHI ${outputName}`;
    if (multipleEnabled) return `Multiple ${outputName}`;
    if (sahiEnabled) return `SAHI ${outputName}`;
    if (everythingEnabled) return `Everything ${outputName}`;
    return outputName;
  }

  function stageText(stage) {
    if (stage === "loading-dino") return "Loading GroundingDINO";
    if (stage === "loading-sam2") return "Loading SAM 2";
    if (stage === "segmenting") return "Segmenting frames";
    if (stage === "stopping") return "Stopping";
    if (stage === "finalizing") return "Finalizing video";
    if (stage === "done") return "Complete";
    return "Preparing video";
  }

  async function readStreamingResult(response) {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("The server did not provide a progress stream.");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let finalPayload = null;

    function handleEventLine(line) {
      if (!line.trim()) return;

      const event = JSON.parse(line);
      if (event.type === "started") {
        setActiveJobId(event.jobId || "");
        setProgress({
          stage: "starting",
          processed: 0,
          total: null,
          percent: null,
          message: "Preparing video job",
        });
        return;
      }

      if (event.type === "progress") {
        const percent = Number(event.percent);
        setProgress({
          stage: event.stage,
          processed: event.processed ?? 0,
          total: event.total ?? null,
          percent: Number.isFinite(percent) ? percent : null,
          message: event.message || stageText(event.stage),
        });
        return;
      }

      if (event.type === "done") {
        finalPayload = event.payload;
        setStopping(false);
        setActiveJobId("");
        setProgress({
          stage: "done",
          processed: event.payload?.metadata?.processed_frames ?? 0,
          total: event.payload?.metadata?.processed_frames ?? null,
          percent: 100,
          message: event.payload?.metadata?.stopped_early ? "Partial video ready" : "Video segmentation complete",
        });
        return;
      }

      if (event.type === "error") {
        throw new Error(event.detail || event.error || "Segmentation failed.");
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        handleEventLine(line);
      }
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      handleEventLine(buffer);
    }

    if (!finalPayload) {
      throw new Error("Segmentation finished without a result.");
    }

    return finalPayload;
  }

  function pickFile(nextFile) {
    if (!nextFile) return;
    setFile(nextFile);
    setMediaType(getMediaType(nextFile));
    setResult(null);
    setError("");
    setProgress(null);
    setActiveJobId("");
    setStopping(false);
    setPreview(URL.createObjectURL(nextFile));
  }

  async function stopSegmentation() {
    if (!activeJobId || stopping) return;

    setStopping(true);
    setProgress((current) => ({
      ...(current || { processed: 0, total: null, percent: null }),
      stage: "stopping",
      message: "Stop requested. Finishing the current frame.",
    }));

    try {
      const response = await fetch("/api/segment/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId: activeJobId }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not stop segmentation.");
      }
    } catch (caught) {
      setStopping(false);
      setError(caught.message);
    }
  }

  async function runSegmentation() {
    const multiClass = mediaType === "image" && multipleEnabled;
    const sahi = mediaType === "image" && sahiEnabled;
    const everything = mediaType === "image" && everythingEnabled && !multiClass;
    const nextClassPrompts = cleanClassPrompts();

    if (!file) {
      setError("Drop an image or video first.");
      return;
    }

    if (!multiClass && !prompt.trim()) {
      setError("Write a prompt first.");
      return;
    }

    if (multiClass && nextClassPrompts.length === 0) {
      setError("Add at least one class for Multiple mode.");
      return;
    }

    if ((everything || sahi || multiClass) && mediaType !== "image") {
      setError("Everything, SAHI, and Multiple work with images only.");
      return;
    }

    if (multiClass) {
      setClassPrompts(nextClassPrompts);
    }
    setLoading(true);
    setLoadingMode(multiClass && sahi ? "multiSahi" : multiClass ? "multiClass" : sahi ? "sahi" : everything ? "everything" : "segment");
    setError("");
    setResult(null);
    setActiveJobId("");
    setStopping(false);
    setProgress(mediaType === "video" ? { stage: "starting", processed: 0, total: null, percent: null, message: "Preparing video job" } : null);

    const formData = new FormData();
    formData.append("media", file);
    formData.append("prompt", prompt);
    formData.append("samModel", samModel);
    formData.append("dinoModel", activeDetectorModel.dinoModel);
    formData.append("detector", activeDetectorModel.detector);
    formData.append("showBoxes", showBoxes ? "1" : "0");
    formData.append("showLabels", showLabels ? "1" : "0");
    formData.append("boxOnly", useSam ? "0" : "1");
    if (everything) {
      formData.append("everything", "1");
    }
    if (sahi) {
      formData.append("sahi", "1");
    }
    if (multiClass) {
      formData.append("multiClass", "1");
      formData.append("classPrompts", JSON.stringify(nextClassPrompts));
    }

    try {
      const response = await fetch("/api/segment", {
        method: "POST",
        body: formData,
      });
      const contentType = response.headers.get("content-type") || "";
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || payload.error || "Segmentation failed.");
      }
      if (contentType.includes("application/x-ndjson")) {
        setResult(await readStreamingResult(response));
        return;
      }
      const payload = await response.json();
      setResult(payload);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setLoading(false);
      setLoadingMode("");
      setActiveJobId("");
      setStopping(false);
    }
  }

  return (
    <main className={`shell ${theme}`}>
      <a className="skipLink" href="#workspace">
        Skip to workspace
      </a>

      <section className="topbar" aria-label="Studio header">
        <div className="brand">
          <span className="brandMark">
            <FiAperture />
          </span>
          <div className="brandCopy">
            <span className="eyebrow">Vision console</span>
            <h1>Grounded SAM 2 Studio</h1>
            <p>Prompt an object, get a box and optional mask.</p>
          </div>
        </div>
        <div className="topActions">
          <div className="statusPills" aria-label="Current configuration">
            <span>{file ? fileKindLabel : "No media"}</span>
            <span>{activeDetectorModel?.name}</span>
            <span>{selectedModeLabel()}</span>
          </div>
          <button
            className="iconButton"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            title="Toggle theme"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            type="button"
          >
            {theme === "dark" ? <FiSun /> : <FiMoon />}
          </button>
        </div>
      </section>

      <section className="workspace" id="workspace">
        <aside className="panel controls" aria-label="Segmentation controls">
          <div className="panelHeader">
            <div>
              <span className="eyebrow">Input</span>
              <h2>Media and prompt</h2>
            </div>
            <span className={`runState ${loading ? "running" : file ? "ready" : ""}`}>
              {loading ? "Running" : file ? "Ready" : "Idle"}
            </span>
          </div>

          <div
            className={`dropzone ${dragging ? "dragging" : ""} ${preview ? "hasPreview" : ""}`}
            onClick={() => {
              if (!preview) inputRef.current?.click();
            }}
            onKeyDown={preview ? undefined : handleDropzoneKeyDown}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              pickFile(event.dataTransfer.files?.[0]);
            }}
            role={preview ? undefined : "button"}
            tabIndex={preview ? undefined : 0}
            aria-label={preview ? undefined : "Upload image or video"}
          >
            <input
              id="media-upload"
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,video/mp4,video/webm,video/quicktime,video/x-msvideo,video/x-matroska"
              onChange={(event) => pickFile(event.target.files?.[0])}
              aria-label="Choose image or video"
            />
            {preview ? (
              <div className="previewFrame">
                {mediaType === "video" ? (
                  <video src={preview} controls muted playsInline aria-label={`Preview of ${file?.name || "selected video"}`} />
                ) : (
                  <img src={preview} alt={file?.name ? `Preview of ${file.name}` : "Selected media preview"} />
                )}
                <div className="previewMeta">
                  <div className="previewMetaText">
                    <strong>{file?.name || "Selected media"}</strong>
                    <span>{fileSummary}</span>
                  </div>
                  <button className="replaceButton" type="button" onClick={() => inputRef.current?.click()}>
                    <FiUploadCloud />
                    Replace
                  </button>
                </div>
              </div>
            ) : (
              <div className="dropContent">
                <FiUploadCloud />
                <strong>Drop an image or video</strong>
                <span>{fileSummary}</span>
              </div>
            )}
          </div>

          <div className="field">
            <span>Output</span>
            <div className="modelGrid outputGrid">
              {outputModes.map((mode) => (
                <button
                  key={mode.id}
                  className={`modelCard compact ${outputMode === mode.id ? "selected" : ""}`}
                  onClick={() => setOutputMode(mode.id)}
                  disabled={loading}
                  type="button"
                  title={mode.name}
                  aria-pressed={outputMode === mode.id}
                >
                  {outputModeIcon(mode)}
                  <strong>{mode.name}</strong>
                  <small>{mode.detail}</small>
                </button>
              ))}
            </div>
          </div>

          {mediaType !== "video" ? (
            <div className="field">
              <span>Detection Mode</span>
              <div className="modelGrid modeGrid">
                <button
                  className={`modelCard compact ${detectionMode === "single" ? "selected" : ""}`}
                  onClick={() => setDetectionMode("single")}
                  disabled={loading}
                  type="button"
                  title="Single prompt"
                  aria-pressed={detectionMode === "single"}
                >
                  <FiTarget />
                  <strong>Single</strong>
                  <small>One prompt</small>
                </button>
                <button
                  className={`modelCard compact ${detectionMode === "multiple" ? "selected" : ""}`}
                  onClick={() => setDetectionMode("multiple")}
                  disabled={loading}
                  type="button"
                  title="Multiple classes"
                  aria-pressed={detectionMode === "multiple"}
                >
                  <FiPlus />
                  <strong>Multiple</strong>
                  <small>Class list</small>
                </button>
                <button
                  className={`modelCard compact ${detectionMode === "everything" ? "selected" : ""}`}
                  onClick={() => setDetectionMode("everything")}
                  disabled={loading}
                  type="button"
                  title="Everything matching"
                  aria-pressed={detectionMode === "everything"}
                >
                  <FiActivity />
                  <strong>Everything</strong>
                  <small>All matches</small>
                </button>
              </div>
              <button
                className={`featureToggle wideToggle ${sahiEnabled ? "active" : ""}`}
                onClick={() => setSahiEnabled((enabled) => !enabled)}
                disabled={loading}
                type="button"
                title="Toggle SAHI"
                aria-pressed={sahiEnabled}
              >
                <FiGrid />
                SAHI Tiling
              </button>
            </div>
          ) : null}

          {mediaType !== "video" && multipleEnabled ? (
            <div className="multiClassPanel">
              <div className="multiClassHeader">
                <span>Classes</span>
                <button
                  className="miniIconButton"
                  onClick={addClassPrompt}
                  disabled={loading || classPrompts.length >= 12}
                  type="button"
                  title="Add class"
                  aria-label="Add class prompt"
                >
                  <FiPlus />
                </button>
              </div>
              <div className="classList">
                {classPrompts.map((classPrompt, index) => (
                  <div className="classRow" key={index}>
                    <input
                      value={classPrompt}
                      onChange={(event) => updateClassPrompt(index, event.target.value)}
                      placeholder={index === 0 ? "car" : "motorcycle"}
                      aria-label={`Class prompt ${index + 1}`}
                    />
                    <button
                      className="miniIconButton danger"
                      onClick={() => removeClassPrompt(index)}
                      disabled={loading || classPrompts.length <= 1}
                      type="button"
                      title="Remove class"
                      aria-label={`Remove class prompt ${index + 1}`}
                    >
                      <FiTrash2 />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <label className="field">
              <span>Prompt</span>
              <input
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="I want to segment the pink flower"
                aria-describedby={error ? "form-error" : undefined}
              />
            </label>
          )}

          {useSam ? (
            <div className="field">
              <span>SAM Model</span>
              <div className="modelGrid">
                {models.map((model) => (
                  <button
                    key={model.id}
                    className={`modelCard ${samModel === model.id ? "selected" : ""}`}
                    onClick={() => setSamModel(model.id)}
                    disabled={loading}
                    type="button"
                    aria-pressed={samModel === model.id}
                  >
                    <FiZap />
                    <strong>{model.name}</strong>
                    <small>{model.detail}</small>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <div className="field">
            <span>Detector</span>
            <div className="modelGrid">
              {detectorModels.map((model) => (
                <button
                  key={model.id}
                  className={`modelCard ${detectorModel === model.id ? "selected" : ""}`}
                  onClick={() => setDetectorModel(model.id)}
                  disabled={loading}
                  type="button"
                  aria-pressed={detectorModel === model.id}
                >
                  {model.detector === "yolo" ? <FiActivity /> : <FiTarget />}
                  <strong>{model.name}</strong>
                  <small>{model.detail}</small>
                </button>
              ))}
            </div>
          </div>

          <div className="controlButtons">
            <button className="runButton" onClick={runSegmentation} disabled={loading} type="button" aria-busy={loading}>
              {loading ? <FiLoader className="spin" /> : <FiActivity />}
              {loading ? `Running ${selectedModeLabel()}...` : `Run ${selectedModeLabel()}`}
            </button>
            {loading && mediaType === "video" ? (
              <button className="stopButton" onClick={stopSegmentation} disabled={!activeJobId || stopping} type="button">
                {stopping ? <FiLoader className="spin" /> : <FiSquare />}
                {stopping ? "Stopping..." : "Stop"}
              </button>
            ) : null}
          </div>

          {loading && progress ? (
            <div className="progressCard" role="status" aria-live="polite">
              <div className="progressMeta">
                <span>{progress.message || stageText(progress.stage)}</span>
                <strong>
                  {progress.percent !== null && progress.percent !== undefined
                    ? `${Math.round(progress.percent)}%`
                    : progress.processed
                      ? `${progress.processed} frames`
                      : "Starting"}
                </strong>
              </div>
              <div className={`progressTrack ${progress.percent === null || progress.percent === undefined ? "indeterminate" : ""}`}>
                <span style={{ width: `${progress.percent ?? 38}%` }} />
              </div>
              <small>
                {progress.total
                  ? `${progress.processed || 0} of ${progress.total} frames`
                  : stageText(progress.stage)}
              </small>
            </div>
          ) : null}

          {error ? (
            <div className="notice error" id="form-error" role="alert">
              {error}
            </div>
          ) : null}
        </aside>

        <section className="panel resultPanel" aria-label="Segmentation result" aria-live="polite">
          {result ? (
            <>
              <div className="resultHeader">
                <div>
                  <span className="eyebrow">Result</span>
                  <h2>{resultLabel}</h2>
                </div>
                <div className="scores">
                  <span>{(result.metadata.device || "cpu").toUpperCase()}</span>
                  {result.metadata.stopped_early ? <span>Partial</span> : null}
                  {isVideoResult ? <span>{result.metadata.processed_frames || 0} frames</span> : null}
                  {result.metadata.combined_mode === "multi_class_sahi" ? <span>Multiple + SAHI {result.metadata.detection_count || 0}</span> : null}
                  {result.metadata.multi_class_mode && result.metadata.combined_mode !== "multi_class_sahi" ? <span>Multiple {result.metadata.detection_count || 0}</span> : null}
                  {result.metadata.everything_mode ? <span>All {result.metadata.detection_count || 0}</span> : null}
                  {result.metadata.sahi_mode && result.metadata.combined_mode !== "multi_class_sahi" ? <span>SAHI {result.metadata.detection_count || 0}</span> : null}
                  {boxOnlyResult ? <span>{cleanBoxResult ? "Box only" : "Box + label"}</span> : null}
                  <span>{result.metadata.show_boxes === false ? "Boxes off" : "Boxes on"}</span>
                  <span>{result.metadata.show_labels === false ? "Labels off" : "Labels on"}</span>
                  <span>{detectorScoreLabel} {scoreText(dinoScore)}</span>
                  <span>{boxOnlyResult ? "SAM skipped" : `SAM ${scoreText(samScore)}`}</span>
                </div>
              </div>

              {isVideoResult ? (
                <div className={`imageCompare videoCompare ${boxOnlyResult ? "singleResult" : ""}`}>
                  <figure>
                    <video
                      src={result.annotatedVideoUrl}
                      poster={result.previewUrl}
                      controls
                      muted
                      playsInline
                      aria-label="Annotated video result"
                    />
                    <figcaption>
                      <FiVideo /> {cleanBoxResult ? "Box Video" : boxOnlyResult ? "Box + Label Video" : "Overlay Video"}
                    </figcaption>
                  </figure>
                  {boxOnlyResult ? null : (
                    <figure>
                      <video
                        src={result.maskVideoUrl}
                        poster={result.maskPreviewUrl}
                        controls
                        muted
                        playsInline
                        aria-label="Mask video result"
                      />
                      <figcaption>
                        <FiFilm /> Mask Video
                      </figcaption>
                    </figure>
                  )}
                </div>
              ) : (
                <div className={`imageCompare ${boxOnlyResult ? "singleResult" : ""}`}>
                  <figure>
                    <img src={result.annotatedUrl} alt="Annotated segmentation result" />
                    <figcaption>
                      <FiImage /> {cleanBoxResult ? "Box" : boxOnlyResult ? "Box + Label" : "Overlay"}
                    </figcaption>
                  </figure>
                  {boxOnlyResult ? null : (
                    <figure>
                      <img src={result.maskUrl} alt="Mask result" />
                      <figcaption>
                        <FiAperture /> Mask
                      </figcaption>
                    </figure>
                  )}
                </div>
              )}

              <div className="actions">
                <a
                  href={isVideoResult ? result.annotatedVideoUrl : result.annotatedUrl}
                  download
                  aria-label={`Download ${cleanBoxResult ? "box" : boxOnlyResult ? "box and label" : "overlay"} result`}
                >
                  <FiDownload /> {cleanBoxResult ? "Box" : boxOnlyResult ? "Box + Label" : "Overlay"}
                </a>
                {boxOnlyResult ? null : (
                  <a href={isVideoResult ? result.maskVideoUrl : result.maskUrl} download aria-label="Download mask result">
                    <FiDownload /> Mask
                  </a>
                )}
              </div>
            </>
          ) : (
            <div className="emptyState" aria-live="polite">
              <FiAperture />
              <h2>{isMultiSahiLoading ? "Running Multiple + SAHI..." : isMultiClassLoading ? "Running classes..." : isSahiLoading ? "Running SAHI..." : isEverythingLoading ? "Finding everything..." : loading ? "Models are thinking..." : "Your result will land here"}</h2>
              <p>{isMultiSahiLoading ? `${cleanClassPrompts().length} classes over SAHI slices with ${activeDetectorModel?.name}.` : isMultiClassLoading ? `${cleanClassPrompts().length} class prompts through ${activeDetectorModel?.name}.` : isSahiLoading ? `Overlapping slices through ${activeDetectorModel?.name}.` : isEverythingLoading ? `All kept boxes through ${activeDetectorModel?.name}.` : loading ? (useSam ? `Running ${activeDetectorModel?.name} and ${activeModel?.name}.` : `Running ${activeDetectorModel?.name} for ${activeOutputMode.name.toLowerCase()}.`) : "Drop media, tune the prompt, and run segmentation."}</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
