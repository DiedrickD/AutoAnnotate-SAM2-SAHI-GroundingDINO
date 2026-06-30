import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


OVERLAY_COLOR = np.array([30, 220, 70], dtype=np.uint8)
PROGRESS_PREFIX = "@@progress "
YOLO_VEHICLE_ALIASES = {
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
    "car": "car",
    "cars": "car",
    "auto": "car",
    "automobile": "car",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "scooter": "motorcycle",
    "scooters": "motorcycle",
    "bus": "bus",
    "buses": "bus",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
}
YOLO_VEHICLE_CLASSES = ["bicycle", "car", "motorcycle", "bus", "truck"]


class ModelLoadError(RuntimeError):
    pass


def load_pretrained_cached_first(loader, model_name: str, description: str, **kwargs):
    try:
        loaded = loader.from_pretrained(model_name, local_files_only=True, **kwargs)
        print(f"Loaded {description} from the local Hugging Face cache.", flush=True)
        return loaded
    except Exception:
        print(
            f"{description} is not fully cached. Trying to download {model_name} from Hugging Face...",
            flush=True,
        )

    try:
        return loader.from_pretrained(model_name, **kwargs)
    except Exception as error:
        raise ModelLoadError(
            f"Could not load {description} '{model_name}'. The local Hugging Face cache is incomplete "
            "and huggingface.co could not be reached. Connect to the internet once to download the model, "
            "or choose a model that is already cached."
        ) from error


def ensure_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    return prompt if prompt.endswith(".") else f"{prompt}."


def detector_prompt_from_user_prompt(prompt: str) -> str:
    """Turn natural commands into a compact GroundingDINO object prompt."""
    text = re.sub(r"\s+", " ", prompt.strip())
    text = text.strip(" .,!?:;\"'")

    command = (
        r"^(?:i\s+want\s+to\s+|i\s+would\s+like\s+to\s+|please\s+|"
        r"can\s+you\s+|could\s+you\s+)?"
        r"(?:segment|detect|mask|track|find|select|isolate)\s+"
        r"(?:the\s+|a\s+|an\s+)?(.+)$"
    )
    match = re.match(command, text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip(" .,!?:;\"'")

    return ensure_prompt(text or prompt)


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "image"


def emit_progress(stage: str, processed=None, total=None, message=None) -> None:
    event = {"stage": stage}
    if processed is not None:
        event["processed"] = int(processed)
    if total is not None:
        event["total"] = int(total)
    if processed is not None and total:
        event["percent"] = round(min(100.0, (processed / total) * 100.0), 2)
    if message:
        event["message"] = message
    print(f"{PROGRESS_PREFIX}{json.dumps(event, separators=(',', ':'))}", flush=True)


def stop_requested(stop_file: Path | None) -> bool:
    return bool(stop_file and stop_file.exists())


def draw_box_and_mask(
    image: np.ndarray,
    box,
    label: str,
    score: float,
    mask: np.ndarray,
    show_boxes: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    return draw_boxes_and_mask(
        image,
        [{"box": box, "label": label, "score": score}],
        mask,
        show_boxes=show_boxes,
        show_labels=show_labels,
    )


def draw_boxes_and_mask(
    image: np.ndarray,
    detections,
    mask: np.ndarray,
    show_boxes: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    output = image.copy()
    color = OVERLAY_COLOR

    overlay = output.copy()
    if mask is not None and np.any(mask):
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)
        output = cv2.addWeighted(overlay, 0.85, output, 0.15, 0)

    height, width = output.shape[:2]
    if not show_boxes:
        return output

    label_rects = []
    for detection in detections:
        box = detection["box"]
        label = str(detection.get("label") or "target")
        score = float(detection.get("score") or 0.0)

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(output, (x1, y1), (x2, y2), (30, 220, 70), 3)
        if not show_labels:
            continue

        caption = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        caption_x = max(0, min(width - tw - 10, x1))
        label_h = th + baseline + 8
        label_w = tw + 10
        preferred_top = max(0, y1 - label_h)
        candidate_tops = [preferred_top]
        for offset in range(1, 8):
            candidate_tops.append(max(0, preferred_top - offset * label_h))
            candidate_tops.append(min(height - label_h, preferred_top + offset * label_h))

        top = preferred_top
        for candidate_top in candidate_tops:
            candidate_rect = (caption_x, candidate_top, caption_x + label_w, candidate_top + label_h)
            overlaps = any(
                candidate_rect[0] < rect[2]
                and candidate_rect[2] > rect[0]
                and candidate_rect[1] < rect[3]
                and candidate_rect[3] > rect[1]
                for rect in label_rects
            )
            if not overlaps:
                top = candidate_top
                label_rects.append(candidate_rect)
                break
        else:
            label_rects.append((caption_x, top, caption_x + label_w, top + label_h))

        cv2.rectangle(
            output,
            (caption_x, top),
            (caption_x + label_w, top + label_h),
            (30, 220, 70),
            -1,
        )
        cv2.putText(
            output,
            caption,
            (caption_x + 5, top + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def select_device_and_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def load_grounding_dino(model_name: str, device: str, dtype):
    print("Loading GroundingDINO...")
    processor = load_pretrained_cached_first(
        AutoProcessor,
        model_name,
        "GroundingDINO processor",
    )
    # GroundingDINO's deformable attention path is not fully fp16-safe here.
    dino_dtype = torch.float32
    model = load_pretrained_cached_first(
        AutoModelForZeroShotObjectDetection,
        model_name,
        "GroundingDINO model",
        dtype=dino_dtype,
    ).to(device)
    model.eval()
    return processor, model


def load_yolo_detector(model_name: str):
    print(f"Loading YOLO detector: {model_name}...")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed. Run `.venv\\Scripts\\python.exe -m pip install ultralytics` "
            "and download the YOLO model first."
        ) from error

    return YOLO(model_name)


def yolo_names_map(yolo_model):
    names = yolo_model.names
    if isinstance(names, dict):
        return {int(index): str(name).lower() for index, name in names.items()}
    return {index: str(name).lower() for index, name in enumerate(names)}


def yolo_class_ids_for_prompt(prompt: str, names_by_id: dict[int, str]):
    text = re.sub(r"[^a-z0-9 ]+", " ", prompt.lower())
    tokens = set(text.split())
    wanted = set()

    if {"vehicle", "vehicles", "traffic", "road"}.intersection(tokens):
        wanted.update(YOLO_VEHICLE_CLASSES)

    for alias, class_name in YOLO_VEHICLE_ALIASES.items():
        if alias in tokens or alias in text:
            wanted.add(class_name)

    if not wanted:
        compact = text.strip()
        if compact in YOLO_VEHICLE_ALIASES:
            wanted.add(YOLO_VEHICLE_ALIASES[compact])

    return [class_id for class_id, name in names_by_id.items() if name in wanted]


def detect_yolo_boxes(
    image: Image.Image,
    prompt: str,
    yolo_model,
    device: str,
    confidence_threshold: float,
    max_detections: int | None = None,
    nms_threshold: float | None = 0.6,
    image_size: int = 960,
):
    names_by_id = yolo_names_map(yolo_model)
    class_ids = yolo_class_ids_for_prompt(prompt, names_by_id)
    if not class_ids:
        print(
            f"YOLO detector skipped prompt '{prompt}'. Supported vehicle prompts: "
            f"{', '.join(YOLO_VEHICLE_CLASSES)}."
        )
        return []

    results = yolo_model.predict(
        source=np.array(image),
        imgsz=image_size,
        conf=confidence_threshold,
        iou=nms_threshold if nms_threshold is not None else 0.7,
        classes=class_ids,
        max_det=max_detections or 300,
        device=0 if device == "cuda" else "cpu",
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    detections = []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    for box, score, class_id in zip(boxes, scores, classes):
        class_name = names_by_id.get(int(class_id), "vehicle")
        detections.append(
            {
                "box": [float(value) for value in box],
                "score": float(score),
                "label": class_name,
            }
        )

    return detections


def load_image_detector(args, device: str, dtype):
    if args.detector == "yolo":
        return {
            "kind": "yolo",
            "model": load_yolo_detector(args.yolo_model),
        }

    dino_processor, dino_model = load_grounding_dino(args.dino_model, device, dtype)
    return {
        "kind": "dino",
        "processor": dino_processor,
        "model": dino_model,
    }


def load_sam2(model_name: str, device: str, dtype):
    print("Loading SAM 2...")
    processor = load_pretrained_cached_first(
        Sam2Processor,
        model_name,
        "SAM 2 processor",
    )
    model = load_pretrained_cached_first(
        Sam2Model,
        model_name,
        "SAM 2 model",
        dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model


def model_dtype(model):
    return next(model.parameters()).dtype


def move_model_inputs(inputs, device: str, dtype):
    inputs = inputs.to(device)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            inputs[key] = value.to(dtype=dtype)
    return inputs


def detect_best_box(
    image: Image.Image,
    prompt: str,
    dino_processor,
    dino_model,
    device: str,
    box_threshold: float,
    text_threshold: float,
):
    detections = detect_boxes(
        image,
        prompt,
        dino_processor,
        dino_model,
        device,
        box_threshold,
        text_threshold,
        max_detections=1,
        nms_threshold=None,
    )
    return detections[0] if detections else None


def detect_with_image_detector(
    image: Image.Image,
    prompt: str,
    detector,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_detections: int | None = None,
    nms_threshold: float | None = 0.6,
    yolo_imgsz: int = 960,
):
    if detector["kind"] == "yolo":
        return detect_yolo_boxes(
            image,
            prompt,
            detector["model"],
            device,
            box_threshold,
            max_detections=max_detections,
            nms_threshold=nms_threshold,
            image_size=yolo_imgsz,
        )

    return detect_boxes(
        image,
        prompt,
        detector["processor"],
        detector["model"],
        device,
        box_threshold,
        text_threshold,
        max_detections=max_detections,
        nms_threshold=nms_threshold,
    )


def detect_best_with_image_detector(
    image: Image.Image,
    prompt: str,
    detector,
    device: str,
    box_threshold: float,
    text_threshold: float,
    yolo_imgsz: int = 960,
):
    detections = detect_with_image_detector(
        image,
        prompt,
        detector,
        device,
        box_threshold,
        text_threshold,
        max_detections=1,
        nms_threshold=None,
        yolo_imgsz=yolo_imgsz,
    )
    return detections[0] if detections else None


def box_iou_xyxy(first, second) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_intersection_over_smaller_xyxy(first, second) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def boxes_are_duplicates(first, second, iou_threshold: float, ios_threshold: float = 0.72) -> bool:
    return (
        box_iou_xyxy(first, second) > iou_threshold
        or box_intersection_over_smaller_xyxy(first, second) > ios_threshold
    )


def detect_boxes(
    image: Image.Image,
    prompt: str,
    dino_processor,
    dino_model,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_detections: int | None = None,
    nms_threshold: float | None = 0.6,
):
    dino_inputs = move_model_inputs(
        dino_processor(images=image, text=prompt, return_tensors="pt"),
        device,
        model_dtype(dino_model),
    )
    with torch.no_grad():
        dino_outputs = dino_model(**dino_inputs)

    detection = dino_processor.post_process_grounded_object_detection(
        dino_outputs,
        dino_inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    if len(detection["boxes"]) == 0:
        return []

    boxes = detection["boxes"].detach().cpu().tolist()
    scores = detection["scores"].detach().cpu().tolist()
    labels = detection["labels"]
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected = []

    for index in order:
        box = [float(value) for value in boxes[index]]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if nms_threshold is not None and any(boxes_are_duplicates(box, item["box"], nms_threshold) for item in selected):
            continue

        selected.append(
            {
                "box": box,
                "score": float(scores[index]),
                "label": str(labels[index]),
            }
        )
        if max_detections and len(selected) >= max_detections:
            break

    return selected


def segment_box(image: Image.Image, box, sam_processor, sam_model, device: str):
    sam_inputs = move_model_inputs(
        sam_processor(images=image, input_boxes=[[box]], return_tensors="pt"),
        device,
        model_dtype(sam_model),
    )
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.detach().cpu(),
        sam_inputs["original_sizes"].detach().cpu(),
    )[0]

    mask_scores = sam_outputs.iou_scores.detach().cpu().reshape(-1)
    mask_idx = int(torch.argmax(mask_scores).item()) if mask_scores.numel() else 0
    mask = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[mask_idx].numpy() > 0
    mask_score = float(mask_scores[mask_idx].item()) if mask_scores.numel() else None
    return mask, mask_score


def should_show_boxes(args) -> bool:
    return bool(args.box_only or not args.hide_boxes)


def should_show_labels(args) -> bool:
    return bool(should_show_boxes(args) and not args.hide_labels)


def sam_metadata(args) -> dict:
    return {
        "sam_model": None if args.box_only else args.sam_model,
        "sam_enabled": not args.box_only,
        "box_only": bool(args.box_only),
        "show_boxes": should_show_boxes(args),
        "show_labels": should_show_labels(args),
    }


def run_image(args, device: str, dtype, detector_prompt: str) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    print(f"Device: {device}")
    detector = load_image_detector(args, device, dtype)

    print(f"Detecting: {detector_prompt}")
    detection = detect_best_with_image_detector(
        image,
        detector_prompt,
        detector,
        device,
        args.box_threshold,
        args.text_threshold,
        args.yolo_imgsz,
    )

    if detection is None:
        raise SystemExit(
            "No detection found. Try a shorter object prompt or lower --box-threshold and "
            "--text-threshold, e.g. 0.15."
        )

    box = detection["box"]
    score = detection["score"]
    label = detection["label"]
    print(f"Best detection: {label} score={score:.3f} box={box}")

    if args.box_only:
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask_score = None
    else:
        sam_processor, sam_model = load_sam2(args.sam_model, device, dtype)
        mask, mask_score = segment_box(image, box, sam_processor, sam_model, device)

    image_np = np.array(image)
    annotated = draw_box_and_mask(
        image_np,
        box,
        str(label),
        score,
        mask,
        show_boxes=should_show_boxes(args),
        show_labels=should_show_labels(args),
    )
    mask_png = mask.astype(np.uint8) * 255

    output_stem = safe_stem(image_path)
    annotated_path = output_dir / f"{output_stem}_grounded_sam2_annotated.png"
    mask_path = output_dir / f"{output_stem}_grounded_sam2_mask.png"
    metadata_path = output_dir / f"{output_stem}_grounded_sam2.json"

    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_png)
    metadata_path.write_text(
        json.dumps(
            {
                "media_type": "image",
                "image": str(image_path),
                "prompt": args.prompt,
                "detector_prompt": detector_prompt,
                "detector": args.detector,
                "dino_model": args.dino_model,
                "yolo_model": args.yolo_model if args.detector == "yolo" else None,
                **sam_metadata(args),
                "device": device,
                "detection_label": label,
                "detection_score": score,
                "sam2_mask_score": mask_score,
                "box_xyxy": box,
                "annotated_output": str(annotated_path),
                "mask_output": str(mask_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved annotated image: {annotated_path}")
    print(f"Saved mask image: {mask_path}")
    print(f"Saved metadata: {metadata_path}")


def run_everything_image(args, device: str, dtype, detector_prompt: str) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    print(f"Device: {device}")
    print(f"Everything mode: max_detections={args.max_detections}, nms_threshold={args.nms_threshold}")
    detector = load_image_detector(args, device, dtype)

    print(f"Detecting everything matching: {detector_prompt}")
    detections = detect_with_image_detector(
        image,
        detector_prompt,
        detector,
        device,
        args.box_threshold,
        args.text_threshold,
        max_detections=args.max_detections,
        nms_threshold=args.nms_threshold,
        yolo_imgsz=args.yolo_imgsz,
    )

    if not detections:
        raise SystemExit(
            "No detections found. Try a shorter object prompt or lower --box-threshold and "
            "--text-threshold, e.g. 0.15."
        )

    sam_processor = sam_model = None
    if not args.box_only:
        sam_processor, sam_model = load_sam2(args.sam_model, device, dtype)
    combined_mask = np.zeros((height, width), dtype=bool)
    detection_records = []
    mask_score_total = 0.0
    mask_score_count = 0

    for index, detection in enumerate(detections):
        box = detection["box"]
        if args.box_only:
            mask_score = None
        else:
            mask, mask_score = segment_box(image, box, sam_processor, sam_model, device)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            combined_mask |= mask

            if mask_score is not None:
                mask_score_total += mask_score
                mask_score_count += 1

        detection_record = {
            "index": index,
            "label": detection["label"],
            "score": detection["score"],
            "mask_score": mask_score,
            "box": box,
        }
        detection_records.append(detection_record)
        print(
            f"Everything detection {index + 1}/{len(detections)}: "
            f"{detection['label']} score={detection['score']:.3f} box={box}"
        )

    output_stem = safe_stem(image_path)
    annotated_path = output_dir / f"{output_stem}_grounded_sam2_annotated.png"
    mask_path = output_dir / f"{output_stem}_grounded_sam2_mask.png"
    metadata_path = output_dir / f"{output_stem}_grounded_sam2.json"

    image_np = np.array(image)
    annotated = draw_boxes_and_mask(
        image_np,
        detection_records,
        combined_mask,
        show_boxes=should_show_boxes(args),
        show_labels=should_show_labels(args),
    )
    mask_png = combined_mask.astype(np.uint8) * 255

    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_png)

    top_detection = max(detection_records, key=lambda item: item["score"])
    average_mask_score = mask_score_total / mask_score_count if mask_score_count else None
    metadata_path.write_text(
        json.dumps(
            {
                "media_type": "image",
                "image": str(image_path),
                "prompt": args.prompt,
                "detector_prompt": detector_prompt,
                "detector": args.detector,
                "dino_model": args.dino_model,
                "yolo_model": args.yolo_model if args.detector == "yolo" else None,
                **sam_metadata(args),
                "device": device,
                "everything_mode": True,
                "detection_count": len(detection_records),
                "max_detections": args.max_detections,
                "nms_threshold": args.nms_threshold,
                "detection_label": f"All {len(detection_records)} detections",
                "detection_score": top_detection["score"],
                "sam2_mask_score": average_mask_score,
                "box_xyxy": top_detection["box"],
                "top_detection": top_detection,
                "detections": detection_records,
                "annotated_output": str(annotated_path),
                "mask_output": str(mask_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved everything annotated image: {annotated_path}")
    print(f"Saved everything mask image: {mask_path}")
    print(f"Saved everything metadata: {metadata_path}")


def make_axis_starts(length: int, window: int, step: int):
    if length <= window:
        return [0]

    starts = []
    current = 0
    while True:
        starts.append(current)
        if current + window >= length:
            break
        next_start = min(length - window, current + step)
        if next_start == current:
            break
        current = next_start
    return starts


def make_sahi_slices(width: int, height: int, slice_size: int, overlap_ratio: float):
    slice_size = max(64, int(slice_size))
    overlap_ratio = max(0.0, min(0.8, float(overlap_ratio)))
    slice_w = min(width, slice_size)
    slice_h = min(height, slice_size)
    step_x = max(1, int(round(slice_w * (1.0 - overlap_ratio))))
    step_y = max(1, int(round(slice_h * (1.0 - overlap_ratio))))
    x_starts = make_axis_starts(width, slice_w, step_x)
    y_starts = make_axis_starts(height, slice_h, step_y)

    slices = []
    for y in y_starts:
        for x in x_starts:
            slices.append(
                {
                    "index": len(slices),
                    "x": x,
                    "y": y,
                    "width": slice_w,
                    "height": slice_h,
                }
            )
    return slices


def merge_overlapping_detections(detections, max_detections: int | None, nms_threshold: float | None, class_aware: bool = False):
    order = sorted(range(len(detections)), key=lambda index: detections[index]["score"], reverse=True)
    selected = []

    for index in order:
        detection = detections[index]
        detection_class = detection.get("class_prompt") or detection.get("label")
        has_duplicate = False
        if nms_threshold is not None:
            for item in selected:
                item_class = item.get("class_prompt") or item.get("label")
                if class_aware and detection_class != item_class:
                    continue
                if boxes_are_duplicates(detection["box"], item["box"], nms_threshold):
                    has_duplicate = True
                    break
        if has_duplicate:
            continue
        selected.append(detection)
        if max_detections and len(selected) >= max_detections:
            break

    return selected


def merge_detections_per_class(detections, max_detections_per_class: int | None, nms_threshold: float | None):
    class_order = []
    grouped = {}
    for detection in detections:
        class_key = detection.get("class_prompt") or detection.get("label") or ""
        if class_key not in grouped:
            grouped[class_key] = []
            class_order.append(class_key)
        grouped[class_key].append(detection)

    merged = []
    for class_key in class_order:
        merged.extend(
            merge_overlapping_detections(
                grouped[class_key],
                max_detections_per_class,
                nms_threshold,
                class_aware=False,
            )
        )

    return merged


def run_multi_class_image(args, device: str, dtype, class_prompts) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    print(f"Device: {device}")
    print(
        f"Multiple mode: classes={len(class_prompts)}, "
        f"max_detections_per_class={args.max_detections_per_class}, nms_threshold={args.nms_threshold}"
    )
    detector = load_image_detector(args, device, dtype)

    raw_detections = []
    class_metadata = []
    for class_index, class_prompt in enumerate(class_prompts):
        print(f"Detecting class {class_index + 1}/{len(class_prompts)}: {class_prompt}")
        class_detections = detect_with_image_detector(
            image,
            class_prompt,
            detector,
            device,
            args.box_threshold,
            args.text_threshold,
            max_detections=args.max_detections_per_class,
            nms_threshold=args.nms_threshold,
            yolo_imgsz=args.yolo_imgsz,
        )
        class_label = class_prompt.rstrip(".")
        class_metadata.append(
            {
                "index": class_index,
                "prompt": class_prompt,
                "detection_count": len(class_detections),
            }
        )
        for detection in class_detections:
            raw_detections.append(
                {
                    "class_index": class_index,
                    "class_prompt": class_prompt,
                    "label": class_label,
                    "detector_label": detection["label"],
                    "score": detection["score"],
                    "box": detection["box"],
                }
            )
        print(f"Multiple class {class_index + 1}/{len(class_prompts)}: {len(class_detections)} detections")

    merged_detections = merge_detections_per_class(
        raw_detections,
        args.max_detections_per_class,
        args.nms_threshold,
    )

    if not merged_detections:
        raise SystemExit(
            "No detections found for any class. Try shorter class names or lower "
            "--box-threshold and --text-threshold, e.g. 0.15."
        )

    sam_processor = sam_model = None
    if not args.box_only:
        sam_processor, sam_model = load_sam2(args.sam_model, device, dtype)
    combined_mask = np.zeros((height, width), dtype=bool)
    detection_records = []
    mask_score_total = 0.0
    mask_score_count = 0

    for index, detection in enumerate(merged_detections):
        box = detection["box"]
        if args.box_only:
            mask_score = None
        else:
            mask, mask_score = segment_box(image, box, sam_processor, sam_model, device)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            combined_mask |= mask

            if mask_score is not None:
                mask_score_total += mask_score
                mask_score_count += 1

        detection_record = {
            "index": index,
            "class_index": detection["class_index"],
            "class_prompt": detection["class_prompt"],
            "label": detection["label"],
            "detector_label": detection["detector_label"],
            "score": detection["score"],
            "mask_score": mask_score,
            "box": box,
        }
        detection_records.append(detection_record)
        print(
            f"Multiple detection {index + 1}/{len(merged_detections)}: "
            f"{detection['label']} score={detection['score']:.3f} box={box}"
        )

    output_stem = safe_stem(image_path)
    annotated_path = output_dir / f"{output_stem}_grounded_sam2_annotated.png"
    mask_path = output_dir / f"{output_stem}_grounded_sam2_mask.png"
    metadata_path = output_dir / f"{output_stem}_grounded_sam2.json"

    image_np = np.array(image)
    annotated = draw_boxes_and_mask(
        image_np,
        detection_records,
        combined_mask,
        show_boxes=should_show_boxes(args),
        show_labels=should_show_labels(args),
    )
    mask_png = combined_mask.astype(np.uint8) * 255

    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_png)

    top_detection = max(detection_records, key=lambda item: item["score"])
    average_mask_score = mask_score_total / mask_score_count if mask_score_count else None
    metadata_path.write_text(
        json.dumps(
            {
                "media_type": "image",
                "image": str(image_path),
                "prompt": args.prompt,
                "detector_prompt": ", ".join(class_prompts),
                "class_prompts": class_prompts,
                "class_results": class_metadata,
                "detector": args.detector,
                "dino_model": args.dino_model,
                "yolo_model": args.yolo_model if args.detector == "yolo" else None,
                **sam_metadata(args),
                "device": device,
                "multi_class_mode": True,
                "raw_detection_count": len(raw_detections),
                "detection_count": len(detection_records),
                "final_detection_limit": "per_class",
                "max_detections_per_class": args.max_detections_per_class,
                "nms_threshold": args.nms_threshold,
                "detection_label": f"Multiple {len(detection_records)} detections",
                "detection_score": top_detection["score"],
                "sam2_mask_score": average_mask_score,
                "box_xyxy": top_detection["box"],
                "top_detection": top_detection,
                "detections": detection_records,
                "annotated_output": str(annotated_path),
                "mask_output": str(mask_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved multiple annotated image: {annotated_path}")
    print(f"Saved multiple mask image: {mask_path}")
    print(f"Saved multiple metadata: {metadata_path}")


def run_sahi_multi_class_image(args, device: str, dtype, class_prompts) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    slices = make_sahi_slices(width, height, args.sahi_slice_size, args.sahi_overlap)

    print(f"Device: {device}")
    print(
        f"Multiple + SAHI mode: classes={len(class_prompts)}, slices={len(slices)}, "
        f"slice_size={args.sahi_slice_size}, overlap={args.sahi_overlap}"
    )
    detector = load_image_detector(args, device, dtype)

    raw_detections = []
    class_metadata = []
    for class_index, class_prompt in enumerate(class_prompts):
        class_label = class_prompt.rstrip(".")
        class_raw_count = 0
        slice_results = []
        print(f"SAHI detecting class {class_index + 1}/{len(class_prompts)}: {class_prompt}")

        for slice_info in slices:
            x = slice_info["x"]
            y = slice_info["y"]
            slice_w = slice_info["width"]
            slice_h = slice_info["height"]
            crop = image.crop((x, y, x + slice_w, y + slice_h))
            slice_detections = detect_with_image_detector(
                crop,
                class_prompt,
                detector,
                device,
                args.box_threshold,
                args.text_threshold,
                max_detections=args.max_detections_per_class,
                nms_threshold=args.nms_threshold,
                yolo_imgsz=args.yolo_imgsz,
            )

            class_raw_count += len(slice_detections)
            slice_results.append(
                {
                    "slice_index": slice_info["index"],
                    "detection_count": len(slice_detections),
                }
            )

            for detection in slice_detections:
                local_box = detection["box"]
                global_box = [
                    local_box[0] + x,
                    local_box[1] + y,
                    local_box[2] + x,
                    local_box[3] + y,
                ]
                raw_detections.append(
                    {
                        "class_index": class_index,
                        "class_prompt": class_prompt,
                        "slice_index": slice_info["index"],
                        "label": class_label,
                        "detector_label": detection["label"],
                        "score": detection["score"],
                        "box": global_box,
                        "local_box_xyxy": local_box,
                        "slice_xywh": [x, y, slice_w, slice_h],
                    }
                )

        class_metadata.append(
            {
                "index": class_index,
                "prompt": class_prompt,
                "raw_detection_count": class_raw_count,
                "slices": slice_results,
            }
        )
        print(f"Multiple + SAHI class {class_index + 1}/{len(class_prompts)}: {class_raw_count} raw detections")

    merged_detections = merge_detections_per_class(
        raw_detections,
        args.max_detections_per_class,
        args.nms_threshold,
    )

    if not merged_detections:
        raise SystemExit(
            "No detections found for any class in SAHI slices. Try shorter class names "
            "or lower --box-threshold and --text-threshold, e.g. 0.15."
        )

    sam_processor = sam_model = None
    if not args.box_only:
        sam_processor, sam_model = load_sam2(args.sam_model, device, dtype)
    combined_mask = np.zeros((height, width), dtype=bool)
    detection_records = []
    mask_score_total = 0.0
    mask_score_count = 0

    for index, detection in enumerate(merged_detections):
        box = detection["box"]
        if args.box_only:
            mask_score = None
        else:
            mask, mask_score = segment_box(image, box, sam_processor, sam_model, device)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            combined_mask |= mask

            if mask_score is not None:
                mask_score_total += mask_score
                mask_score_count += 1

        detection_record = {
            "index": index,
            "class_index": detection["class_index"],
            "class_prompt": detection["class_prompt"],
            "slice_index": detection["slice_index"],
            "label": detection["label"],
            "detector_label": detection["detector_label"],
            "score": detection["score"],
            "mask_score": mask_score,
            "box": box,
            "local_box_xyxy": detection["local_box_xyxy"],
            "slice_xywh": detection["slice_xywh"],
        }
        detection_records.append(detection_record)
        print(
            f"Multiple + SAHI detection {index + 1}/{len(merged_detections)}: "
            f"{detection['label']} score={detection['score']:.3f} box={box}"
        )

    output_stem = safe_stem(image_path)
    annotated_path = output_dir / f"{output_stem}_grounded_sam2_annotated.png"
    mask_path = output_dir / f"{output_stem}_grounded_sam2_mask.png"
    metadata_path = output_dir / f"{output_stem}_grounded_sam2.json"

    image_np = np.array(image)
    annotated = draw_boxes_and_mask(
        image_np,
        detection_records,
        combined_mask,
        show_boxes=should_show_boxes(args),
        show_labels=should_show_labels(args),
    )
    mask_png = combined_mask.astype(np.uint8) * 255

    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_png)

    top_detection = max(detection_records, key=lambda item: item["score"])
    average_mask_score = mask_score_total / mask_score_count if mask_score_count else None
    metadata_path.write_text(
        json.dumps(
            {
                "media_type": "image",
                "image": str(image_path),
                "prompt": args.prompt,
                "detector_prompt": ", ".join(class_prompts),
                "class_prompts": class_prompts,
                "class_results": class_metadata,
                "detector": args.detector,
                "dino_model": args.dino_model,
                "yolo_model": args.yolo_model if args.detector == "yolo" else None,
                **sam_metadata(args),
                "device": device,
                "multi_class_mode": True,
                "sahi_mode": True,
                "combined_mode": "multi_class_sahi",
                "sahi_slice_size": args.sahi_slice_size,
                "sahi_overlap": args.sahi_overlap,
                "sahi_slice_count": len(slices),
                "sahi_slices": slices,
                "raw_detection_count": len(raw_detections),
                "detection_count": len(detection_records),
                "final_detection_limit": "per_class",
                "max_detections_per_class": args.max_detections_per_class,
                "nms_threshold": args.nms_threshold,
                "detection_label": f"Multiple + SAHI {len(detection_records)} detections",
                "detection_score": top_detection["score"],
                "sam2_mask_score": average_mask_score,
                "box_xyxy": top_detection["box"],
                "top_detection": top_detection,
                "detections": detection_records,
                "annotated_output": str(annotated_path),
                "mask_output": str(mask_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved multiple + SAHI annotated image: {annotated_path}")
    print(f"Saved multiple + SAHI mask image: {mask_path}")
    print(f"Saved multiple + SAHI metadata: {metadata_path}")


def run_sahi_image(args, device: str, dtype, detector_prompt: str) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    slices = make_sahi_slices(width, height, args.sahi_slice_size, args.sahi_overlap)

    print(f"Device: {device}")
    print(
        f"SAHI mode: {len(slices)} slices, slice_size={args.sahi_slice_size}, "
        f"overlap={args.sahi_overlap}, max_detections={args.max_detections}"
    )
    detector = load_image_detector(args, device, dtype)

    raw_detections = []
    slice_metadata = []
    print(f"SAHI detecting: {detector_prompt}")
    for slice_info in slices:
        x = slice_info["x"]
        y = slice_info["y"]
        slice_w = slice_info["width"]
        slice_h = slice_info["height"]
        crop = image.crop((x, y, x + slice_w, y + slice_h))
        slice_detections = detect_with_image_detector(
            crop,
            detector_prompt,
            detector,
            device,
            args.box_threshold,
            args.text_threshold,
            max_detections=args.max_detections,
            nms_threshold=args.nms_threshold,
            yolo_imgsz=args.yolo_imgsz,
        )

        slice_record = {
            **slice_info,
            "detection_count": len(slice_detections),
        }
        slice_metadata.append(slice_record)

        for detection in slice_detections:
            local_box = detection["box"]
            global_box = [
                local_box[0] + x,
                local_box[1] + y,
                local_box[2] + x,
                local_box[3] + y,
            ]
            raw_detections.append(
                {
                    "slice_index": slice_info["index"],
                    "label": detection["label"],
                    "score": detection["score"],
                    "box": global_box,
                    "local_box_xyxy": local_box,
                    "slice_xywh": [x, y, slice_w, slice_h],
                }
            )
        print(
            f"SAHI slice {slice_info['index'] + 1}/{len(slices)}: "
            f"{len(slice_detections)} detections"
        )

    merged_detections = merge_overlapping_detections(
        raw_detections,
        args.max_detections,
        args.nms_threshold,
    )

    if not merged_detections:
        raise SystemExit(
            "No detections found in SAHI slices. Try a shorter object prompt or lower "
            "--box-threshold and --text-threshold, e.g. 0.15."
        )

    sam_processor = sam_model = None
    if not args.box_only:
        sam_processor, sam_model = load_sam2(args.sam_model, device, dtype)
    combined_mask = np.zeros((height, width), dtype=bool)
    detection_records = []
    mask_score_total = 0.0
    mask_score_count = 0

    for index, detection in enumerate(merged_detections):
        box = detection["box"]
        if args.box_only:
            mask_score = None
        else:
            mask, mask_score = segment_box(image, box, sam_processor, sam_model, device)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            combined_mask |= mask

            if mask_score is not None:
                mask_score_total += mask_score
                mask_score_count += 1

        detection_record = {
            "index": index,
            "slice_index": detection["slice_index"],
            "label": detection["label"],
            "score": detection["score"],
            "mask_score": mask_score,
            "box": box,
            "local_box_xyxy": detection["local_box_xyxy"],
            "slice_xywh": detection["slice_xywh"],
        }
        detection_records.append(detection_record)
        print(
            f"SAHI merged detection {index + 1}/{len(merged_detections)}: "
            f"{detection['label']} score={detection['score']:.3f} box={box}"
        )

    output_stem = safe_stem(image_path)
    annotated_path = output_dir / f"{output_stem}_grounded_sam2_annotated.png"
    mask_path = output_dir / f"{output_stem}_grounded_sam2_mask.png"
    metadata_path = output_dir / f"{output_stem}_grounded_sam2.json"

    image_np = np.array(image)
    annotated = draw_boxes_and_mask(
        image_np,
        detection_records,
        combined_mask,
        show_boxes=should_show_boxes(args),
        show_labels=should_show_labels(args),
    )
    mask_png = combined_mask.astype(np.uint8) * 255

    cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask_png)

    top_detection = max(detection_records, key=lambda item: item["score"])
    average_mask_score = mask_score_total / mask_score_count if mask_score_count else None
    metadata_path.write_text(
        json.dumps(
            {
                "media_type": "image",
                "image": str(image_path),
                "prompt": args.prompt,
                "detector_prompt": detector_prompt,
                "detector": args.detector,
                "dino_model": args.dino_model,
                "yolo_model": args.yolo_model if args.detector == "yolo" else None,
                **sam_metadata(args),
                "device": device,
                "sahi_mode": True,
                "sahi_slice_size": args.sahi_slice_size,
                "sahi_overlap": args.sahi_overlap,
                "sahi_slice_count": len(slices),
                "sahi_slices": slice_metadata,
                "raw_detection_count": len(raw_detections),
                "detection_count": len(detection_records),
                "max_detections": args.max_detections,
                "nms_threshold": args.nms_threshold,
                "detection_label": f"SAHI {len(detection_records)} detections",
                "detection_score": top_detection["score"],
                "sam2_mask_score": average_mask_score,
                "box_xyxy": top_detection["box"],
                "top_detection": top_detection,
                "detections": detection_records,
                "annotated_output": str(annotated_path),
                "mask_output": str(mask_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved SAHI annotated image: {annotated_path}")
    print(f"Saved SAHI mask image: {mask_path}")
    print(f"Saved SAHI metadata: {metadata_path}")


def open_video_writer(path: Path, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise SystemExit(f"Could not create video writer for {path}")
    return writer


def run_video(args, device: str, dtype, detector_prompt: str) -> None:
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        capture.release()
        raise SystemExit("Could not read video dimensions.")

    output_stem = safe_stem(video_path)
    annotated_path = output_dir / f"{output_stem}_grounded_sam2_annotated.mp4"
    mask_video_path = output_dir / f"{output_stem}_grounded_sam2_mask.mp4"
    preview_path = output_dir / f"{output_stem}_grounded_sam2_annotated_preview.png"
    mask_preview_path = output_dir / f"{output_stem}_grounded_sam2_mask_preview.png"
    metadata_path = output_dir / f"{output_stem}_grounded_sam2.json"
    stop_file = Path(args.stop_file) if args.stop_file else None

    detect_every = max(1, int(args.detect_every))
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else None
    total_frames = min(frame_count, max_frames) if frame_count and max_frames else max_frames or frame_count or None

    print(f"Device: {device}")
    print(f"Video: {width}x{height} at {fps:.2f} fps, frames={frame_count or 'unknown'}")
    emit_progress(
        "loading",
        processed=0,
        total=total_frames,
        message="Loading detection model" if args.box_only else "Loading detection and segmentation models",
    )
    emit_progress("loading-dino", processed=0, total=total_frames, message="Loading detector")
    detector = load_image_detector(args, device, dtype)
    sam_processor = sam_model = None
    if args.box_only:
        emit_progress("segmenting", processed=0, total=total_frames, message="Drawing detection boxes")
    else:
        emit_progress("loading-sam2", processed=0, total=total_frames, message="Loading SAM 2")
        sam_processor, sam_model = load_sam2(args.sam_model, device, dtype)
        emit_progress("segmenting", processed=0, total=total_frames, message="Segmenting frames")

    annotated_writer = open_video_writer(annotated_path, fps, width, height)
    mask_writer = open_video_writer(mask_video_path, fps, width, height)

    last_box = None
    last_label = detector_prompt.rstrip(".")
    last_score = 0.0
    last_mask_score = None
    first_detection = None
    detections_found = 0
    missed_detections = 0
    processed_frames = 0
    mask_score_total = 0.0
    mask_score_count = 0
    first_annotated = None
    first_mask_bgr = None
    preview_written = False
    stopped_early = False

    print(f"Detecting video target: {detector_prompt}")

    try:
        while True:
            if stop_requested(stop_file):
                stopped_early = True
                emit_progress(
                    "stopping",
                    processed=processed_frames,
                    total=total_frames,
                    message="Stopping after last completed frame",
                )
                break

            ok, frame_bgr = capture.read()
            if not ok:
                break
            if max_frames is not None and processed_frames >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            should_detect = processed_frames % detect_every == 0 or last_box is None
            if should_detect:
                detection = detect_best_with_image_detector(
                    image,
                    detector_prompt,
                    detector,
                    device,
                    args.box_threshold,
                    args.text_threshold,
                    args.yolo_imgsz,
                )
                if detection is not None:
                    last_box = detection["box"]
                    last_label = detection["label"]
                    last_score = detection["score"]
                    detections_found += 1
                    if first_detection is None:
                        first_detection = {
                            "frame_index": processed_frames,
                            "label": last_label,
                            "score": last_score,
                            "box_xyxy": last_box,
                        }
                else:
                    missed_detections += 1

            if last_box is not None:
                if args.box_only:
                    last_mask_score = None
                    annotated_rgb = draw_box_and_mask(
                        frame_rgb,
                        last_box,
                        last_label,
                    last_score,
                    None,
                    show_boxes=should_show_boxes(args),
                    show_labels=should_show_labels(args),
                )
                    mask_gray = np.zeros((height, width), dtype=np.uint8)
                else:
                    mask, last_mask_score = segment_box(image, last_box, sam_processor, sam_model, device)
                    if last_mask_score is not None:
                        mask_score_total += last_mask_score
                        mask_score_count += 1
                    annotated_rgb = draw_box_and_mask(
                        frame_rgb,
                        last_box,
                        last_label,
                        last_score,
                        mask,
                        show_boxes=should_show_boxes(args),
                        show_labels=should_show_labels(args),
                    )
                    mask_gray = mask.astype(np.uint8) * 255
            else:
                annotated_rgb = frame_rgb
                mask_gray = np.zeros((height, width), dtype=np.uint8)

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
            annotated_writer.write(annotated_bgr)
            mask_writer.write(mask_bgr)

            if first_annotated is None:
                first_annotated = annotated_bgr.copy()
                first_mask_bgr = mask_bgr.copy()
            if not preview_written and last_box is not None:
                cv2.imwrite(str(preview_path), annotated_bgr)
                cv2.imwrite(str(mask_preview_path), mask_bgr)
                preview_written = True

            processed_frames += 1
            emit_progress(
                "segmenting",
                processed=processed_frames,
                total=total_frames,
                message=f"Processed {processed_frames}/{total_frames or '?'} frames",
            )

            if stop_requested(stop_file):
                stopped_early = True
                emit_progress(
                    "stopping",
                    processed=processed_frames,
                    total=total_frames,
                    message=f"Stopped at frame {processed_frames}",
                )
                break
    finally:
        capture.release()
        annotated_writer.release()
        mask_writer.release()

    if processed_frames == 0:
        raise SystemExit("No frames were read from the video.")

    emit_progress(
        "finalizing",
        processed=processed_frames,
        total=total_frames,
        message="Finalizing output videos",
    )

    if not preview_written and first_annotated is not None and first_mask_bgr is not None:
        cv2.imwrite(str(preview_path), first_annotated)
        cv2.imwrite(str(mask_preview_path), first_mask_bgr)

    average_mask_score = mask_score_total / mask_score_count if mask_score_count else None
    metadata_path.write_text(
        json.dumps(
            {
                "media_type": "video",
                "video": str(video_path),
                "prompt": args.prompt,
                "detector_prompt": detector_prompt,
                "detector": args.detector,
                "dino_model": args.dino_model,
                "yolo_model": args.yolo_model if args.detector == "yolo" else None,
                **sam_metadata(args),
                "device": device,
                "width": width,
                "height": height,
                "fps": fps,
                "source_frame_count": frame_count,
                "processed_frames": processed_frames,
                "stopped_early": stopped_early,
                "detect_every": detect_every,
                "detections_found": detections_found,
                "missed_detections": missed_detections,
                "first_detection": first_detection,
                "detection_label": first_detection["label"] if first_detection else None,
                "detection_score": first_detection["score"] if first_detection else None,
                "sam2_mask_score": average_mask_score,
                "last_sam2_mask_score": last_mask_score,
                "annotated_output": str(annotated_path),
                "mask_output": str(mask_video_path),
                "preview_output": str(preview_path),
                "mask_preview_output": str(mask_preview_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if detections_found == 0:
        print(
            "No detections were found in sampled frames. The output videos were still saved, "
            "but the mask will be empty.",
            flush=True,
        )

    print(f"Saved annotated video: {annotated_path}")
    print(f"Saved mask video: {mask_video_path}")
    print(f"Saved metadata: {metadata_path}")
    emit_progress(
        "done",
        processed=processed_frames,
        total=processed_frames,
        message="Partial video ready" if stopped_early else "Video segmentation complete",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="testimage.jpg")
    parser.add_argument("--video")
    parser.add_argument("--prompt", default="segment black gaming headset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sam-model", default="facebook/sam2-hiera-tiny")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--detector", choices=["dino", "yolo"], default="dino")
    parser.add_argument("--yolo-model", default="yolo11x.pt")
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--hide-boxes", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--box-only", "--no-sam", action="store_true")
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--all-detections", action="store_true")
    parser.add_argument("--multi-class", action="store_true")
    parser.add_argument("--class-prompts", default="[]")
    parser.add_argument("--max-detections", type=int, default=24)
    parser.add_argument("--max-detections-per-class", type=int, default=12)
    parser.add_argument("--nms-threshold", type=float, default=0.6)
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument("--sahi-slice-size", type=int, default=256)
    parser.add_argument("--sahi-overlap", type=float, default=0.25)
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stop-file")
    args = parser.parse_args()

    detector_prompt = detector_prompt_from_user_prompt(args.prompt)
    class_prompts = []
    if args.multi_class:
        try:
            raw_classes = json.loads(args.class_prompts)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Could not parse --class-prompts JSON: {error}") from error
        if not isinstance(raw_classes, list):
            raise SystemExit("--class-prompts must be a JSON list.")
        class_prompts = [
            detector_prompt_from_user_prompt(str(item))
            for item in raw_classes
            if str(item).strip()
        ]
        seen = set()
        class_prompts = [
            prompt
            for prompt in class_prompts
            if not (prompt.lower() in seen or seen.add(prompt.lower()))
        ]
        if not class_prompts:
            raise SystemExit("Multiple mode needs at least one class prompt.")
    device, dtype = select_device_and_dtype()

    if args.video:
        run_video(args, device, dtype, detector_prompt)
    elif args.multi_class and args.sahi:
        run_sahi_multi_class_image(args, device, dtype, class_prompts)
    elif args.multi_class:
        run_multi_class_image(args, device, dtype, class_prompts)
    elif args.all_detections:
        run_everything_image(args, device, dtype, detector_prompt)
    elif args.sahi:
        run_sahi_image(args, device, dtype, detector_prompt)
    else:
        run_image(args, device, dtype, detector_prompt)


if __name__ == "__main__":
    try:
        main()
    except ModelLoadError as error:
        raise SystemExit(str(error)) from None
