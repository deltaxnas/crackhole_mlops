"""Evaluate YOLO models against LabelMe ground-truth annotations.

Import ``evaluate_models`` from this module and call it from your pipeline.
All run settings are grouped in the configuration section below.
"""

import os
import json
import cv2
import numpy as np
import tqdm
# import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
from ultralytics import YOLO
from .ID2NAME import MAP


# =============================================================================
# CONFIGURATION - change these values when needed
# =============================================================================

EVALUATION_DATA_DIR = "./experiments/blending_experiment/evaluation"
MODELS_DIR = "./weights_exp"
OUTPUT_DIR = "./test"
PANEL_GT_DIR = "./assets/panel_GT"
TARGET_SUBGROUPS = ["SIDEOTR"]

CONFIDENCE_VALUES = [0.6]
GT_IOU_THRESHOLD = 0.001
DUPLICATE_IOU_THRESHOLD = 0.6
INFERENCE_DEVICE = "cuda:1"

PANEL_LABEL = "500"
GT_LABEL = "1"
ENABLE_MASK = False
MASK_COLOR = 0
ENABLE_CROP_LETTERBOX = True
LETTERBOX_SHAPE = (1088, 1280)

OUTPUT_POSTFIX = ""
SAVE_ANNOTATED_IMAGES = False
ANNOTATED_IMAGES_DIR = "./per_subgroup_/per_panel_pairs_per_model"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def extract_panel_id_from_filename(filename):
    pattern = r'P\d+_S\d+'
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def load_all_subgroup_models(base_model_dir):
    """
    Load all YOLO models for specific subgroups.

    Returns:
        dict: {subgroup: [(model, model_name), ...]}
    """
    subgroup_models = {}

    for subgroup in TARGET_SUBGROUPS:
        subgroup_dir = os.path.join(base_model_dir, subgroup)
        if not os.path.exists(subgroup_dir):
            print(f"Warning: Subgroup directory {subgroup_dir} does not exist")
            continue

        entries = []    
        for file in sorted(os.listdir(subgroup_dir)):
            if file.endswith(f'.pt'):
            # if file.startswith("method") and file.endswith(f'{POSTFIX}.pt'):
                model_path = os.path.join(subgroup_dir, file)
                model = YOLO(model_path, verbose=False)
                model_name = file.replace('.pt', '')
                entries.append((model, model_name))
                print(f"Loaded {subgroup}/{file}  →  model #{len(entries)}")

        if entries:
            subgroup_models[subgroup] = entries
            print(f"Total {len(entries)} models for subgroup '{subgroup}'")
        else:
            print(f"No .pt models found in {subgroup_dir}")

    return subgroup_models


def get_panel_bounding_box(labelme_json_path, panel_label="500"):
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_points = []
    for shape in data.get('shapes', []):
        if shape.get('label') == panel_label:
            points = shape.get('points', [])
            if points:
                all_points.extend(points)

    if not all_points:
        return None

    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    return (int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords)))


def crop_image_by_bbox(image, bbox, new_shape=(1088, 1280),
                       color=(114, 114, 114), auto=False,
                       scaleFill=False, scaleup=True):
    if bbox is None:
        return letterbox_resize(image, new_shape, color, auto, scaleFill, scaleup)

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    cropped = image[y1:y2, x1:x2]
    return letterbox_resize(cropped, new_shape, color, auto, scaleFill, scaleup)


def letterbox_resize(im, new_shape=(1088, 1280), color=(114, 114, 114),
                     auto=False, scaleFill=False, scaleup=True):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, int(dh), int(dh), int(dw), int(dw),
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, [dw, dh]


def apply_mask_from_labelme(image, labelme_json_path, mask_label,
                             mask_color, bbox_offset=None):
    try:
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {labelme_json_path}: {e}")
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    panel_exists = False

    for shape_idx, shape in enumerate(data.get('shapes', [])):
        if shape.get('label') == mask_label:
            polygon = shape.get('points', [])
            if polygon and len(polygon) >= 3:
                try:
                    if bbox_offset:
                        x_off, y_off = bbox_offset
                        adjusted = [[p[0] - x_off, p[1] - y_off] for p in polygon
                                    if 0 <= p[0] - x_off < image.shape[1]
                                    and 0 <= p[1] - y_off < image.shape[0]]
                    else:
                        adjusted = polygon

                    if len(adjusted) < 3:
                        continue

                    cv2.fillPoly(mask, [np.array(adjusted, dtype=np.int32)], 255)
                    panel_exists = True
                except Exception as e:
                    print(f"Error processing polygon in {labelme_json_path}, "
                          f"shape {shape_idx}: {e}")

    if not panel_exists:
        return image

    try:
        masked = image.copy()
        alpha = np.expand_dims(mask / 255.0, axis=2)
        for c in range(3):
            masked[:, :, c] = (image[:, :, c] * alpha[:, :, 0]
                               + mask_color * (1 - alpha[:, :, 0]))
        return masked
    except Exception as e:
        print(f"Error applying mask in {labelme_json_path}: {e}")
        return image


def inference(model, image_array, conf):
    result = model.predict(
        image_array, device=INFERENCE_DEVICE, conf=conf, verbose=False
    )
    detections_con = [
        [(int(box.cls), float(box.conf), [int(c) for c in box.xyxy[0]])
         for box in r.boxes if int(box.cls) == 1]
        for r in result
    ]
    return detections_con[0] if detections_con else []


def iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def remove_duplicate_detections(predictions, iou_threshold=0.6):
    if not predictions:
        return []
    filtered, used = [], [False] * len(predictions)
    for i, pb in enumerate(predictions):
        if used[i]:
            continue
        filtered.append(pb)
        used[i] = True
        for j in range(i + 1, len(predictions)):
            if not used[j] and iou(pb, predictions[j]) > iou_threshold:
                used[j] = True
    return filtered


def check_TP_FP_FN(image, predictions, gt_bboxes,
                   iou_threshold=0.5, iou_to_remove_double_dets=0.1):
    predictions = remove_duplicate_detections(
        predictions, iou_threshold=iou_to_remove_double_dets)

    tp, fp = 0, 0
    used_gt = [False] * len(gt_bboxes)

    for pred_box in predictions:
        conf_val = pred_box[4] if len(pred_box) >= 5 else 1.0
        coords = pred_box[:4]
        best_iou, best_idx = 0, -1
        for gi, gt_box in enumerate(gt_bboxes):
            if not used_gt[gi]:
                cur = iou(coords, gt_box)
                if cur > best_iou:
                    best_iou, best_idx = cur, gi

        if best_iou > iou_threshold and best_idx >= 0:
            tp += 1
            used_gt[best_idx] = True
            if image is not None:
                cv2.rectangle(image, (coords[0], coords[1]),
                              (coords[2], coords[3]), (0, 255, 0), 2)
                cv2.putText(image, f"{conf_val:.2f}",
                            (coords[0], coords[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:
            fp += 1
            if image is not None:
                cv2.rectangle(image, (coords[0], coords[1]),
                              (coords[2], coords[3]), (0, 0, 255), 2)
                cv2.putText(image, f"{conf_val:.2f}",
                            (coords[0], coords[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    fn = sum(1 for used in used_gt if not used)
    for i, gt_box in enumerate(gt_bboxes):
        if not used_gt[i] and image is not None:
            cv2.rectangle(image, (gt_box[0], gt_box[1]),
                          (gt_box[2], gt_box[3]), (255, 0, 0), 2)

    return tp, fp, fn, image


def calculate_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def get_gt_bboxes_from_labelme(labelme_json_path, gt_label):
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bboxes = []
    for shape in data.get('shapes', []):
        if shape.get('label') == gt_label:
            pts = shape.get('points', [])
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bboxes.append([int(min(xs)), int(min(ys)),
                               int(max(xs)), int(max(ys))])
    return bboxes


def print_results_table(results):
    """
    Pretty-print a table: rows = subgroups, columns = models.

    results: {subgroup: {model_name: {'tp','fp','fn','precision','recall'}}}
    """
    if not results:
        print("No evaluation results to display.")
        return

    # Collect all model names (preserve insertion order)
    all_models = []
    for sg_data in results.values():
        for mn in sg_data:
            if mn not in all_models:
                all_models.append(mn)

    col_w = max(12, max(len(m) for m in all_models) + 2)
    sg_w  = max(10, max(len(sg) for sg in results) + 2)

    header = f"{'Subgroup':<{sg_w}}" + "".join(
        f"{'Model':<{col_w}}{'P':>7}{'R':>7}{'TP':>5}{'FP':>5}{'FN':>5}"
        for _ in all_models
    )
    # Simpler readable table
    sep = "-" * (sg_w + len(all_models) * (col_w + 24))
    print("\n" + sep)
    print(f"{'Subgroup':<{sg_w}}", end="")
    for mn in all_models:
        label = f"[{mn}]"
        print(f"{label:^{col_w + 24}}", end="")
    print()
    print(f"{'':>{sg_w}}", end="")
    for _ in all_models:
        print(f"{'Prec':>8}{'Rec':>8}{'TP':>6}{'FP':>6}{'FN':>6}", end="")
    print()
    print(sep)

    for sg, sg_data in sorted(results.items()):
        print(f"{sg:<{sg_w}}", end="")
        for mn in all_models:
            if mn in sg_data:
                m = sg_data[mn]
                print(f"{m['precision']:8.4f}{m['recall']:8.4f}"
                      f"{m['tp']:6d}{m['fp']:6d}{m['fn']:6d}", end="")
            else:
                print(f"{'N/A':>8}{'N/A':>8}{'':>6}{'':>6}{'':>6}", end="")
        print()
    print(sep + "\n")


def save_results_csv(results, conf, output_path):
    """Save per-model per-subgroup results to CSV."""
    rows = []
    for sg, sg_data in sorted(results.items()):
        for mn, m in sg_data.items():
            rows.append({
                'subgroup': sg,
                'model': mn,
                'conf': conf,
                'TP': m['tp'],
                'FP': m['fp'],
                'FN': m['fn'],
                'precision': round(m['precision'], 6),
                'recall': round(m['recall'], 6),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df



def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0


def filter_preds_outside_mask(preds, json_path, mask_label):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        return preds  # if no json, keep all

    mask_polygons = [
        shape["points"]
        for shape in data.get("shapes", [])
        if shape.get("label") == mask_label
    ]

    if not mask_polygons:
        return preds

    filtered_preds = []

    for pred in preds:
        x1, y1, x2, y2, conf = pred

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        inside_any_mask = any(
            is_point_in_polygon((cx, cy), poly)
            for poly in mask_polygons
        )

        if not inside_any_mask:
            filtered_preds.append(pred)

    return filtered_preds


from shapely.geometry import Polygon, box


def compute_iou_poly_bbox(bbox, polygon_points):
    """
    bbox: [x1, y1, x2, y2]
    polygon_points: [[x, y], ...]
    """
    try:
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        poly = Polygon(polygon_points)

        if not poly.is_valid:
            poly = poly.buffer(0)

        inter = bbox_poly.intersection(poly).area
        union = bbox_poly.union(poly).area

        if union == 0:
            return 0.0

        return inter / union

    except Exception:
        return 0.0


def filter_preds_outside_mask_iou(preds, json_path, mask_label, iou_thresh=0.05):
    """
    Remove predictions that overlap mask polygon using IoU
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except:
        return preds

    mask_polygons = [
        shape["points"]
        for shape in data.get("shapes", [])
        if shape.get("label") == mask_label
    ]

    if not mask_polygons:
        return preds

    filtered_preds = []

    for pred in preds:
        x1, y1, x2, y2, conf = pred

        overlaps = False
        for poly in mask_polygons:
            iou = compute_iou_poly_bbox([x1, y1, x2, y2], poly)
            if iou > iou_thresh:
                overlaps = True
                break

        if not overlaps:
            filtered_preds.append(pred)
    return filtered_preds

# ─────────────────────────────────────────────────────────────────────────────
def evaluate_models():
    """Run the configured evaluation and return results by confidence value."""
    mapping = MAP()

    gt_dir = EVALUATION_DATA_DIR
    output_dir = OUTPUT_DIR
    temp_output_dir = ANNOTATED_IMAGES_DIR

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {gt_dir}")
    if not os.path.isdir(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    if SAVE_ANNOTATED_IMAGES:
        os.makedirs(temp_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load models: {subgroup: [(model, model_name), ...]}
    subgroup_models_dict = load_all_subgroup_models(MODELS_DIR)
    print(f"\nLoaded models for {len(subgroup_models_dict)} subgroup(s)\n")
    if not subgroup_models_dict:
        raise RuntimeError("No models found for the configured subgroups")

    panel_subgroup_cache = {}
    all_results = {}

    for conf in CONFIDENCE_VALUES:
        print(f"\n{'='*60}")
        print(f"  Evaluating  conf = {conf}")
        print(f"{'='*60}")

        # results[subgroup][model_name] = {tp, fp, fn, precision, recall}
        results: dict[str, dict[str, dict]] = {}

        # Initialise metric accumulators for every known model
        for subgroup, entries in subgroup_models_dict.items():
            results[subgroup] = {}
            for _, model_name in entries:
                results[subgroup][model_name] = {'tp': 0, 'fp': 0, 'fn': 0}

        images = [
            filename for filename in os.listdir(gt_dir)
            if os.path.splitext(filename)[1].lower()
            in {'.jpg', '.jpeg', '.png', '.tiff'}
        ]

        for img_file in tqdm.tqdm(sorted(images),
                                  desc=f"conf={conf}"):
            img_path = os.path.join(gt_dir, img_file)

            panel_id = extract_panel_id_from_filename(img_file)
            if not panel_id:
                continue

            if panel_id in panel_subgroup_cache:
                subgroup = panel_subgroup_cache[panel_id]
            else:
                try:
                    subgroup = mapping.get_panel_subgroup_by_id(panel_id)
                    panel_subgroup_cache[panel_id] = subgroup
                except Exception as e:
                    print(f"  Subgroup lookup failed for {panel_id}: {e}")
                    continue

            if subgroup not in subgroup_models_dict:
                continue   # no models loaded for this subgroup

            panel_name = mapping.get_panel_name(panel_id)
            base = img_path.split('/')[-1].split('.')[0]
            cam_id = base[2] if base.startswith("C0") else base[1:3]
            gt_json_path = f"{PANEL_GT_DIR}/{panel_name}/{cam_id}.json"

            try:
                image = cv2.imread(img_path)

                USING_GT_JSON = False
                json_path = img_path.replace('.jpg', '.json').replace('.png', '.json')
                if not os.path.exists(json_path):
                    json_path = gt_json_path
                    USING_GT_JSON = True
                    if not os.path.exists(json_path):
                        continue

                panel_bbox = get_panel_bounding_box(
                    json_path, panel_label=PANEL_LABEL
                )
                if panel_bbox is None:
                    if not USING_GT_JSON:
                        panel_bbox = get_panel_bounding_box(
                            gt_json_path, panel_label=PANEL_LABEL
                        )
                    if panel_bbox is None:
                        continue

                # Masking
                if ENABLE_MASK:
                    processed_image = apply_mask_from_labelme(
                        image, gt_json_path, PANEL_LABEL, MASK_COLOR, None)
                else:
                    processed_image = image

                # Crop + letterbox
                if ENABLE_CROP_LETTERBOX:
                    processed_image, scale_ratio, padding_info = crop_image_by_bbox(
                        processed_image, panel_bbox, new_shape=LETTERBOX_SHAPE)
                    x1_offset, y1_offset = panel_bbox[0], panel_bbox[1]
                    dw, dh = padding_info
                else:
                    x1_offset = y1_offset = dw = dh = 0
                    scale_ratio = 1

                # Adjust GT bboxes to letterbox space
                raw_gt = (
                    get_gt_bboxes_from_labelme(json_path, GT_LABEL)
                    if not USING_GT_JSON else []
                )
                adjusted_gt = []
                for gt_box in raw_gt:
                    lb = [
                        int((gt_box[0] - x1_offset) * scale_ratio + dw),
                        int((gt_box[1] - y1_offset) * scale_ratio + dh),
                        int((gt_box[2] - x1_offset) * scale_ratio + dw),
                        int((gt_box[3] - y1_offset) * scale_ratio + dh),
                    ]
                    if (lb[0] >= 0 and lb[1] >= 0
                            and lb[2] <= processed_image.shape[1]
                            and lb[3] <= processed_image.shape[0]):
                        adjusted_gt.append(lb)

                # ── Evaluate EACH model independently ──────────────────────
                for model, model_name in subgroup_models_dict[subgroup]:
                    image_array = np.array(processed_image)
                    dets = inference(model, image_array, conf)

                    preds = [[*box, c] for _, c, box in dets] if dets else []
                    # 🚫 Remove detections inside mask polygon
                    # preds = filter_preds_outside_mask(preds, gt_json_path, mask_label)
                    # 🚫 IoU-based filtering with mask
                    # preds = filter_preds_outside_mask_iou(
                    #     preds,
                    #     gt_json_path,
                    #     mask_label,
                    #     iou_thresh=0.02  # tune this
                    # )


                    tp, fp, fn, annotated = check_TP_FP_FN(
                        processed_image.copy(),
                        preds,
                        adjusted_gt,
                        iou_threshold=GT_IOU_THRESHOLD,
                        iou_to_remove_double_dets=DUPLICATE_IOU_THRESHOLD
                    )

                    results[subgroup][model_name]['tp'] += tp
                    results[subgroup][model_name]['fp'] += fp
                    results[subgroup][model_name]['fn'] += fn

                    # Save annotated image per model
                    out_name = (f"{subgroup}_{model_name}_{panel_id}"
                                f"_{base}_cropped.png")
                    if SAVE_ANNOTATED_IMAGES:
                        cv2.imwrite(os.path.join(temp_output_dir, out_name), annotated)

            except Exception as e:
                print(f"  Error on {img_file}: {e}")
                continue

        # ── Compute precision / recall after all images ────────────────────
        for sg, sg_data in results.items():
            for mn, m in sg_data.items():
                p, r = calculate_precision_recall(m['tp'], m['fp'], m['fn'])
                m['precision'] = p
                m['recall']    = r

        # ── Print summary table ────────────────────────────────────────────
        print_results_table(results)
        all_results[conf] = results

        # ── Save CSV ───────────────────────────────────────────────────────
        csv_path = os.path.join(
            output_dir,
            f"per_model_per_subgroup_conf{conf}{OUTPUT_POSTFIX}.csv"
        )
        results_df = save_results_csv(results, conf, csv_path)

        # ── Save charts ────────────────────────────────────────────────────
        # create_visualizations(results_df, output_dir, OUTPUT_POSTFIX, conf)

    return all_results


def _pipeline_gt_bboxes(labelme_json_path, labels):
    """Read configured Labelme rectangle/polygon labels as bounding boxes."""
    with open(labelme_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    accepted = {str(label) for label in labels}
    boxes = []
    for shape in data.get("shapes", []):
        if str(shape.get("label")) not in accepted:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
    return boxes


def _pipeline_predictions(model, image, confidence, device, class_id):
    predictions = []
    for result in model.predict(image, device=device, conf=confidence, verbose=False):
        for detection in result.boxes:
            if int(detection.cls) != class_id:
                continue
            coords = [int(value) for value in detection.xyxy[0]]
            predictions.append([*coords, float(detection.conf)])
    return predictions


def _pipeline_panel_gt_path(image_path, panel_gt_dir):
    """Resolve the optional fixed panel annotation used by the old evaluator."""
    if not panel_gt_dir:
        return None
    panel_id = extract_panel_id_from_filename(image_path.name)
    panel_name = MAP().get_panel_name(panel_id) if panel_id else None
    if not panel_name:
        return None
    stem = image_path.stem
    camera_id = stem[2] if stem.startswith("C0") else stem[1:3]
    candidate = Path(panel_gt_dir) / panel_name / f"{camera_id}.json"
    return candidate if candidate.is_file() else None


class CrackEvaluationError(RuntimeError):
    """Evaluation execution failed after a diagnostic report was written."""

    def __init__(self, message, report_path):
        super().__init__(message)
        self.report_path = str(report_path)


def evaluate_crack_model(
    model_path, config, batch_id, dataset_yaml=None, evaluation_attempt=1
):
    """Evaluate one pipeline-produced crack model and write a JSON report.

    This is the pipeline-oriented counterpart to ``evaluate_models`` above.
    It evaluates exactly one model, takes settings from the task configuration,
    and returns the pass/fail metrics required by ``evaluation_adapter``.
    """
    options = config["evaluation"]
    data_dir = Path(options["data_dir"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"crack evaluation directory not found: {data_dir}")
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"trained model not found: {model_file}")

    output_dir = (
        Path(options["output_dir"]) / batch_id / f"eval_{evaluation_attempt:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    save_annotated = bool(options.get("save_annotated_images", False))
    if save_annotated:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    confidence = float(options.get("confidence", 0.6))
    gt_iou = float(options.get("gt_iou_threshold", 0.001))
    duplicate_iou = float(options.get("duplicate_iou_threshold", 0.6))
    device = options.get("device", config["training"].get("device"))
    prediction_class_id = int(options.get("prediction_class_id", 1))
    gt_labels = options.get("ground_truth_labels", ["1"])
    panel_label = str(options.get("panel_label", "500"))
    enable_crop = bool(options.get("enable_crop_letterbox", True))
    enable_mask = bool(options.get("enable_mask", False))
    letterbox_shape = tuple(options.get("letterbox_shape", [1088, 1280]))
    panel_gt_dir = options.get("panel_gt_dir")
    image_extensions = {
        extension.lower() for extension in config["dataset"]["image_extensions"]
    }

    model = YOLO(str(model_file), verbose=False)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    images_evaluated = 0
    errors = []
    images = sorted(
        path for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not images:
        raise ValueError(f"no evaluation images found in {data_dir}")

    for image_path in tqdm.tqdm(images, desc=f"evaluate {batch_id}"):
        try:
            annotation_path = image_path.with_suffix(".json")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"missing Labelme annotation: {annotation_path.name}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"OpenCV could not read {image_path.name}")

            panel_gt_path = _pipeline_panel_gt_path(image_path, panel_gt_dir)
            panel_annotation = panel_gt_path or annotation_path
            if enable_mask:
                image = apply_mask_from_labelme(
                    image, str(panel_annotation), panel_label,
                    int(options.get("mask_color", 0)), None,
                )

            panel_bbox = get_panel_bounding_box(str(annotation_path), panel_label)
            if panel_bbox is None and panel_gt_path:
                panel_bbox = get_panel_bounding_box(str(panel_gt_path), panel_label)
            if enable_crop and panel_bbox is None:
                raise ValueError(f"no panel bounding box for {image_path.name}")

            raw_gt = _pipeline_gt_bboxes(annotation_path, gt_labels)
            if enable_crop:
                processed, scale, padding = crop_image_by_bbox(
                    image, panel_bbox, new_shape=letterbox_shape
                )
                x_offset, y_offset = panel_bbox[0], panel_bbox[1]
                dw, dh = padding
                adjusted_ground_truth = [
                    [
                        int((box[0] - x_offset) * scale + dw),
                        int((box[1] - y_offset) * scale + dh),
                        int((box[2] - x_offset) * scale + dw),
                        int((box[3] - y_offset) * scale + dh),
                    ]
                    for box in raw_gt
                ]
                ground_truth = [
                    box for box in adjusted_ground_truth
                    if box[0] >= 0 and box[1] >= 0
                    and box[2] <= processed.shape[1]
                    and box[3] <= processed.shape[0]
                ]
            else:
                processed = image
                ground_truth = raw_gt

            predictions = _pipeline_predictions(
                model, processed, confidence, device, prediction_class_id
            )
            tp, fp, fn, annotated = check_TP_FP_FN(
                processed.copy() if save_annotated else None,
                predictions,
                ground_truth,
                iou_threshold=gt_iou,
                iou_to_remove_double_dets=duplicate_iou,
            )
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            images_evaluated += 1
            if save_annotated:
                cv2.imwrite(str(annotated_dir / image_path.name), annotated)
        except Exception as error:
            errors.append({"image": image_path.name, "error": str(error)})

    precision, recall = calculate_precision_recall(
        totals["tp"], totals["fp"], totals["fn"]
    )
    min_precision = float(options.get("min_precision", 0.0))
    min_recall = float(options.get("min_recall", 0.0))
    min_images = int(options.get("min_images", 1))
    passed = (
        not errors
        and images_evaluated >= min_images
        and precision >= min_precision
        and recall >= min_recall
    )
    report_path = output_dir / "evaluation.json"
    report = {
        "batch_id": batch_id,
        "evaluation_attempt": evaluation_attempt,
        "task": config["task_name"],
        "model_path": str(model_file.resolve()),
        "training_dataset_yaml": (
            str(Path(dataset_yaml).resolve()) if dataset_yaml else None
        ),
        "evaluation_data": str(data_dir.resolve()),
        "confidence": confidence,
        "gt_iou_threshold": gt_iou,
        "duplicate_iou_threshold": duplicate_iou,
        "prediction_class_id": prediction_class_id,
        "ground_truth_labels": [str(label) for label in gt_labels],
        **totals,
        "precision": precision,
        "recall": recall,
        "images_found": len(images),
        "images_evaluated": images_evaluated,
        "errors": errors,
        "thresholds": {
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_images": min_images,
        },
        "passed": passed,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors and options.get("fail_on_image_error", True):
        raise CrackEvaluationError(
            f"{len(errors)} crack evaluation image(s) failed; see {report_path}",
            report_path,
        )
    report["passed"] = (
        images_evaluated >= min_images
        and precision >= min_precision
        and recall >= min_recall
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report_path": str(report_path)}
