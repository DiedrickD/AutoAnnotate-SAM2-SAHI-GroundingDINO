import { execFile, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);

const SAM_MODELS = new Set([
  "facebook/sam2-hiera-tiny",
  "facebook/sam2.1-hiera-base-plus",
]);

const DINO_MODELS = new Set([
  "IDEA-Research/grounding-dino-tiny",
  "IDEA-Research/grounding-dino-base",
]);

const DETECTORS = new Set(["dino", "yolo"]);
const YOLO_MODEL_PATH = "yolo11x.pt";
const YOLO_IMAGE_SIZE = "960";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".webm", ".avi", ".mkv"]);

function extensionFromType(type) {
  if (type?.startsWith("image/")) {
    if (type === "image/jpeg") return ".jpg";
    if (type === "image/webp") return ".webp";
    return ".png";
  }
  if (type?.startsWith("video/")) {
    if (type === "video/webm") return ".webm";
    if (type === "video/quicktime") return ".mov";
    return ".mp4";
  }
  return ".png";
}

function safeName(name, type) {
  const parsed = path.parse(name || `upload${extensionFromType(type)}`);
  const stem = parsed.name.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[._]+|[._]+$/g, "") || "image";
  const parsedExt = parsed.ext.toLowerCase();
  const ext =
    IMAGE_EXTENSIONS.has(parsedExt) || VIDEO_EXTENSIONS.has(parsedExt)
      ? parsedExt
      : extensionFromType(type);
  return `${stem}${ext}`;
}

function mediaKind(fileName, type) {
  const ext = path.extname(fileName).toLowerCase();
  if (IMAGE_EXTENSIONS.has(ext) || type?.startsWith("image/")) return "image";
  if (VIDEO_EXTENSIONS.has(ext) || type?.startsWith("video/")) return "video";
  return "unknown";
}

function resultUrl(jobId, fileName) {
  return `/results/${jobId}/${fileName}`;
}

function safeStem(fileName) {
  return path.parse(fileName).name.replace(/[^A-Za-z0-9_.-]+/g, "_").replace(/^[._]+|[._]+$/g, "") || "image";
}

function parseClassPrompts(value) {
  if (!value) return [];
  let parsed;
  try {
    parsed = JSON.parse(String(value));
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];

  const seen = new Set();
  const prompts = [];
  for (const item of parsed) {
    const prompt = String(item || "").trim();
    const key = prompt.toLowerCase();
    if (!prompt || seen.has(key)) continue;
    seen.add(key);
    prompts.push(prompt);
    if (prompts.length >= 12) break;
  }
  return prompts;
}

function videoPayload({ jobId, stem, stdout, stderr, metadata }) {
  return {
    jobId,
    mediaType: "video",
    stdout,
    stderr,
    metadata,
    annotatedVideoUrl: resultUrl(jobId, `${stem}_grounded_sam2_annotated.mp4`),
    maskVideoUrl: resultUrl(jobId, `${stem}_grounded_sam2_mask.mp4`),
    previewUrl: resultUrl(jobId, `${stem}_grounded_sam2_annotated_preview.png`),
    maskPreviewUrl: resultUrl(jobId, `${stem}_grounded_sam2_mask_preview.png`),
  };
}

function streamVideoRun({ pythonPath, runnerArgs, root, outputDir, jobId, stem }) {
  const encoder = new TextEncoder();

  return new Response(
    new ReadableStream({
      start(controller) {
        let stdout = "";
        let stderr = "";
        let stdoutBuffer = "";
        let stderrBuffer = "";
        let closed = false;

        function send(event) {
          if (!closed) {
            controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
          }
        }

        function close() {
          if (!closed) {
            closed = true;
            controller.close();
          }
        }

        function handleStdoutLine(line) {
          stdout += `${line}\n`;
          if (!line.startsWith("@@progress ")) return;
          try {
            send({ type: "progress", ...JSON.parse(line.slice("@@progress ".length)) });
          } catch {
            send({ type: "log", message: line });
          }
        }

        function handleStderrLine(line) {
          stderr += `${line}\n`;
        }

        send({ type: "started", jobId, mediaType: "video" });

        const child = spawn(pythonPath, runnerArgs, {
          cwd: root,
          windowsHide: true,
          env: {
            ...process.env,
            HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "0",
          },
        });

        child.stdout?.setEncoding("utf8");
        child.stderr?.setEncoding("utf8");

        child.stdout?.on("data", (chunk) => {
          stdoutBuffer += chunk;
          const lines = stdoutBuffer.split(/\r?\n/);
          stdoutBuffer = lines.pop() || "";
          for (const line of lines) {
            handleStdoutLine(line);
          }
        });

        child.stderr?.on("data", (chunk) => {
          stderrBuffer += chunk;
          const lines = stderrBuffer.split(/\r?\n/);
          stderrBuffer = lines.pop() || "";
          for (const line of lines) {
            handleStderrLine(line);
          }
        });

        child.on("error", (error) => {
          send({
            type: "error",
            error: "Segmentation failed.",
            detail: error.message,
          });
          close();
        });

        child.on("close", async (code) => {
          if (stdoutBuffer) {
            handleStdoutLine(stdoutBuffer);
            stdoutBuffer = "";
          }
          if (stderrBuffer) {
            handleStderrLine(stderrBuffer);
            stderrBuffer = "";
          }

          if (closed) return;

          if (code !== 0) {
            send({
              type: "error",
              error: "Segmentation failed.",
              detail: stderr || stdout || `Python exited with code ${code}.`,
            });
            close();
            return;
          }

          try {
            const metadataPath = path.join(outputDir, `${stem}_grounded_sam2.json`);
            const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
            send({
              type: "done",
              payload: videoPayload({ jobId, stem, stdout, stderr, metadata }),
            });
          } catch (error) {
            send({
              type: "error",
              error: "Segmentation finished, but result metadata could not be read.",
              detail: error.message,
            });
          } finally {
            close();
          }
        });
      },
    }),
    {
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-store",
      },
    },
  );
}

export async function POST(request) {
  const formData = await request.formData();
  const media = formData.get("media") || formData.get("image");
  const prompt = String(formData.get("prompt") || "").trim();
  const samModel = String(formData.get("samModel") || "facebook/sam2-hiera-tiny");
  const dinoModel = String(formData.get("dinoModel") || "IDEA-Research/grounding-dino-tiny");
  const detector = String(formData.get("detector") || "dino");
  const boxOnly = String(formData.get("boxOnly") || "") === "1";
  const showBoxes = boxOnly || String(formData.get("showBoxes") || "1") !== "0";
  const showLabels = showBoxes && String(formData.get("showLabels") || "1") !== "0";
  const everything = String(formData.get("everything") || "") === "1";
  const sahi = String(formData.get("sahi") || "") === "1";
  const multiClass = String(formData.get("multiClass") || "") === "1";
  const classPrompts = parseClassPrompts(formData.get("classPrompts"));
  const effectivePrompt = multiClass && classPrompts.length ? classPrompts.join(", ") : prompt;

  if (!media || typeof media.arrayBuffer !== "function") {
    return Response.json({ error: "Drop an image or video first." }, { status: 400 });
  }

  if (!effectivePrompt) {
    return Response.json({ error: "Type a prompt before running segmentation." }, { status: 400 });
  }
  if (multiClass && classPrompts.length === 0) {
    return Response.json({ error: "Add at least one class for Multiple mode." }, { status: 400 });
  }

  if (!boxOnly && !SAM_MODELS.has(samModel)) {
    return Response.json({ error: "Unsupported SAM model." }, { status: 400 });
  }
  if (!DINO_MODELS.has(dinoModel)) {
    return Response.json({ error: "Unsupported DINO model." }, { status: 400 });
  }
  if (!DETECTORS.has(detector)) {
    return Response.json({ error: "Unsupported detector." }, { status: 400 });
  }

  const root = process.cwd();
  const jobId = randomUUID();
  const uploadDir = path.join(root, "webapp-data", "uploads", jobId);
  const outputDir = path.join(root, "public", "results", jobId);
  await mkdir(uploadDir, { recursive: true });
  await mkdir(outputDir, { recursive: true });

  const fileName = safeName(media.name, media.type);
  const kind = mediaKind(fileName, media.type);
  if (kind === "unknown") {
    return Response.json({ error: "Unsupported file type. Use an image or video." }, { status: 400 });
  }
  if ((everything || sahi || multiClass) && kind !== "image") {
    return Response.json({ error: "Everything, SAHI, and Multiple work with images only." }, { status: 400 });
  }

  const mediaPath = path.join(uploadDir, fileName);
  const bytes = Buffer.from(await media.arrayBuffer());
  await writeFile(mediaPath, bytes);

  const pythonPath = process.env.PYTHON_PATH || process.env.PYTHON || path.join(root, ".venv", "Scripts", "python.exe");
  const scriptPath = path.join(root, "run_grounded_sam2.py");
  const stopFile = path.join(uploadDir, "stop.requested");
  const runnerArgs = [
    scriptPath,
    kind === "video" ? "--video" : "--image",
    mediaPath,
    "--prompt",
    effectivePrompt,
    "--sam-model",
    samModel,
    "--dino-model",
    dinoModel,
    "--detector",
    detector,
    "--output-dir",
    outputDir,
  ];
  if (detector === "yolo") {
    runnerArgs.push("--yolo-model", YOLO_MODEL_PATH, "--yolo-imgsz", YOLO_IMAGE_SIZE);
  }
  if (!showBoxes) {
    runnerArgs.push("--hide-boxes");
  }
  if (!showLabels) {
    runnerArgs.push("--hide-labels");
  }
  if (boxOnly) {
    runnerArgs.push("--box-only");
  }
  if (multiClass && kind === "image") {
    runnerArgs.push(
      "--multi-class",
      "--class-prompts",
      JSON.stringify(classPrompts),
      "--max-detections",
      "24",
      "--max-detections-per-class",
      "24",
    );
    if (sahi) {
      runnerArgs.push("--sahi");
    }
  } else if (sahi && kind === "image") {
    runnerArgs.push("--sahi", "--max-detections", "24");
  } else if (everything && kind === "image") {
    runnerArgs.push("--all-detections", "--max-detections", "24");
  }
  const stem = safeStem(fileName);

  if (kind === "video") {
    runnerArgs.push("--stop-file", stopFile);
    return streamVideoRun({ pythonPath, runnerArgs, root, outputDir, jobId, stem });
  }

  try {
    const { stdout, stderr } = await execFileAsync(
      pythonPath,
      runnerArgs,
      {
        cwd: root,
        timeout: 1000 * 60 * 60,
        maxBuffer: 1024 * 1024 * 32,
        env: {
          ...process.env,
          HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE || "0",
        },
      },
    );

    const metadataPath = path.join(outputDir, `${stem}_grounded_sam2.json`);
    const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
    const mediaType = metadata.media_type || kind;

    if (mediaType === "video") {
      return Response.json(videoPayload({ jobId, stem, stdout, stderr, metadata }));
    }

    return Response.json({
      jobId,
      mediaType,
      stdout,
      stderr,
      metadata,
      annotatedUrl: resultUrl(jobId, `${stem}_grounded_sam2_annotated.png`),
      maskUrl: resultUrl(jobId, `${stem}_grounded_sam2_mask.png`),
    });
  } catch (error) {
    return Response.json(
      {
        error: "Segmentation failed.",
        detail: error.stderr || error.stdout || error.message,
      },
      { status: 500 },
    );
  }
}
