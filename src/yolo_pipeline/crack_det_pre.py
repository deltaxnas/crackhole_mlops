"""Create a cropped YOLO dataset from panel images and LabelMe annotations.

Import this file and call ``create_yolo_dataset()`` from another process.
"""

import cv2
import numpy as np
import json
import re
from shapely.geometry import Polygon
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from ultralytics import YOLO
from .ID2NAME import MAP


# =============================================================================
# CONFIGURATION - change these values when needed
# =============================================================================

SOURCE_PATH = "./saved_exp_0825"
TARGET_PATH = "./yolo_exp_0825"
SEGMENTATION_LABELS_PATH = "./segmentation_label"
PANEL_GT_PATH = "/home/deltax/Desktop/crackhole/assets/panel_GT"
HOLE_MODEL_PATH = "/home/deltax/Desktop/crackhole/assets/weights/hwaseong_finetune/All_panel_hole_model_v2.pt"

# Panel IDs in filenames must follow this pattern: _P<number>_S<number>.
# Example: image_D20260903_P1_S2_C01.json -> P1_S2
PANEL_ID_PATTERN = r"(?:^|_)(P\d+_S\d+)(?:_|$)"

USE_HOLE_MODEL = True
# Keep reviewed images without cracks. Existing class-0 boxes are preserved;
# images with no boxes receive an empty YOLO label file. Set False for the
# previous positive-only behavior. This is intentionally a script option.
INCLUDE_NEGATIVE_IMAGES = True
HOLE_CONFIDENCE = 0.2
HOLE_IOU = 0.5

MASK_COLOR = (0, 0, 0)
PANEL_POLYGON_LABEL = "500"
CRACK_LABELS = [701]
NO_CRACK_LABELS = range(0, 150)
IGNORE_AREA_LABELS = ["555", "666"]

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def crop_resize_image(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop and resize image to target size.
    
    Args:
        img: Input image
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        Resized image
    """
    x1, y1, x2, y2 = bbox
    img = img[int(y1):int(y2), int(x1):int(x2)]
    # img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img


def mask_image(img: np.ndarray, json_path: Path, additional_label_path=None, gt_json_path=None) -> np.ndarray:
        """Apply masking to an image based on JSON annotations.
        
        Args:
            img: Input image
            json_path: Path to LabelMe JSON file
            
        Returns:
            Masked image
        """
        
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        GT_POLY_EXISTS = False
        bbox_for_resizing = []

        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        for shape in data["shapes"]:
            if shape['label'] == PANEL_POLYGON_LABEL:
                poly = np.array(shape["points"], np.int32)
                if len(poly) < 2:
                    continue
                
                # Handle rectangle polygons (2 points) by converting to 4-point polygon
                if len(poly) == 2:
                    # Convert 2 corner points to 4-point rectangle
                    x1, y1 = poly[0]
                    x2, y2 = poly[1]
                    rect_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    panel_polygon = Polygon(rect_points)
                else:
                    # Use original polygon points
                    panel_polygon = Polygon(shape["points"])
                
                polygon = np.array(panel_polygon.exterior.coords, np.int32)
                
                bbox_for_resizing.append(panel_polygon.bounds)

                cv2.fillPoly(mask, [polygon], 0)
                GT_POLY_EXISTS = True
                # break
            
        if not GT_POLY_EXISTS and additional_label_path is not None:
            if additional_label_path.endswith(".json"):
                with open(additional_label_path, 'r') as file:
                    data = json.load(file)

                mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                for shape in data["shapes"]:
                    if shape['label'] == PANEL_POLYGON_LABEL:
                        poly = np.array(shape["points"], np.int32)
                        if len(poly) < 2:
                            continue
                        
                        # Handle rectangle polygons (2 points) by converting to 4-point polygon
                        if len(poly) == 2:
                            # Convert 2 corner points to 4-point rectangle
                            x1, y1 = poly[0]
                            x2, y2 = poly[1]
                            rect_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            panel_polygon = Polygon(rect_points)
                        else:
                            # Use original polygon points
                            panel_polygon = Polygon(shape["points"])
                        
                        polygon = np.array(panel_polygon.exterior.coords, np.int32)
                        
                        bbox_for_resizing.append(panel_polygon.bounds)

                        cv2.fillPoly(mask, [polygon], 0)
                        GT_POLY_EXISTS = True

            elif additional_label_path.endswith(".txt"):
                # Load YOLO format segmentation mask data
                try:
                    with open(additional_label_path, 'r') as file:
                        lines = file.readlines()
                    
                    # mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                    img_height, img_width = img.shape[:2]
                    drawn_masks = 0
                    ignore_masks = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            print(f"{additional_label_path} has empty line")
                            continue
                        
                        parts = line.split()
                        if len(parts) < 5:  # At least class_id + 4 points for polygon
                            print(f"{additional_label_path} has less than 5 parts: {parts}")
                            continue
                        
                        class_id = int(parts[0])
                        if not class_id in [0, 1]:
                            print(f"{additional_label_path} has class_id {class_id} which is not 0 or 1")
                            continue
                        
                        # Extract polygon points (YOLO format: normalized coordinates)
                        points = []
                        for i in range(1, len(parts), 2):
                            if i + 1 < len(parts):
                                norm_x = float(parts[i])
                                norm_y = float(parts[i + 1])
                                # Convert to pixel coordinates
                                pixel_x = int(norm_x * img_width)
                                pixel_y = int(norm_y * img_height)
                                points.append([pixel_x, pixel_y])
                        
                        if len(points) >= 3:  # Need at least 3 points for a polygon
                            poly = np.array(points, np.int32)
                            
                            # Handle rectangle polygons (2 points) by converting to 4-point polygon
                            if len(poly) == 2:
                                # Convert 2 corner points to 4-point rectangle
                                x1, y1 = poly[0]
                                x2, y2 = poly[1]
                                rect_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                                panel_polygon = Polygon(rect_points)
                            else:
                                # Use original polygon points
                                panel_polygon = Polygon(points)
                            
                            polygon = np.array(panel_polygon.exterior.coords, np.int32)
                            
                            bbox_for_resizing.append(panel_polygon.bounds)
                            
                            if class_id == 1:
                                cv2.fillPoly(mask, [polygon], 0)
                                GT_POLY_EXISTS = True
                            elif class_id==0:
                                ignore_masks.append(polygon)
                                # cv2.fillPoly(mask, [polygon], 127)
                                
                            else:
                                print(f"Unknown class id: {class_id}")
                            # elif class_id==2:
                                # cv2.fillPoly(mask, [polygon], 0)
                            
                            
                            # break  # Use first valid polygon found

                        else:
                            print(f"{additional_label_path} has {len(points)} points")

                    for ignore_mask in ignore_masks:
                        cv2.fillPoly(mask, [ignore_mask], 127)

                    # print(f"Loaded {len(lines)} lines from {additional_label_path}, drawn {drawn_masks} masks")
                except Exception as e:
                    print(f"Error loading YOLO segmentation file {additional_label_path}: {e}")
                    # Continue with next processing step

        if not GT_POLY_EXISTS and gt_json_path is not None:
            with open(gt_json_path, 'r') as file:
                data = json.load(file)

            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            for gt_shape in data["shapes"]:
                if gt_shape['label'] == PANEL_POLYGON_LABEL:
                    gt_poly = np.array(gt_shape["points"], np.int32)
                    if len(gt_poly) < 2:
                        continue
                    
                    # Handle rectangle polygons (2 points) by converting to 4-point polygon
                    if len(gt_poly) == 2:
                        # Convert 2 corner points to 4-point rectangle
                        x1, y1 = gt_poly[0]
                        x2, y2 = gt_poly[1]
                        rect_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        panel_polygon = Polygon(rect_points)
                    else:
                        # Use original polygon points
                        panel_polygon = Polygon(gt_poly)
                    
                    polygon = np.array(panel_polygon.exterior.coords, np.int32)
                    
                    bbox_for_resizing.append(panel_polygon.bounds)

                    cv2.fillPoly(mask, [polygon], 0)
                    GT_POLY_EXISTS = True

        if GT_POLY_EXISTS:
            if len(bbox_for_resizing) > 0:
                # Find the overall bounding box that contains all polygons
                min_x = min(bbox[0] for bbox in bbox_for_resizing)
                min_y = min(bbox[1] for bbox in bbox_for_resizing)
                max_x = max(bbox[2] for bbox in bbox_for_resizing)
                max_y = max(bbox[3] for bbox in bbox_for_resizing)
                bbox_for_resizing = (min_x, min_y, max_x, max_y)
            elif len(bbox_for_resizing) == 1:
                bbox_for_resizing = bbox_for_resizing[0]
            else:
                raise ValueError("No bounding boxes found")
            
            for shape in data["shapes"]:
                if shape['label'] in IGNORE_AREA_LABELS:
                    poly = np.array(shape["points"], np.int32)
                    if len(poly) < 2:
                        continue
                    
                    # Handle rectangle polygons (2 points) by converting to 4-point polygon
                    if len(poly) == 2:
                        # Convert 2 corner points to 4-point rectangle
                        x1, y1 = poly[0]
                        x2, y2 = poly[1]
                        rect_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        polygon = Polygon(rect_points)
                    else:
                        # Use original polygon points
                        polygon = Polygon(shape["points"])
                    
                    polygon = np.array(polygon.exterior.coords, np.int32)
                    
                    if shape["label"] in ["555"]:
                        cv2.fillPoly(mask, [polygon], 255)
                    elif shape["label"] in IGNORE_AREA_LABELS:
                        cv2.fillPoly(mask, [polygon], 127)

            img[mask == 255] = MASK_COLOR
            img[mask == 127] = (255, 255, 255)
            img = crop_resize_image(img, bbox_for_resizing)
            return img, bbox_for_resizing, GT_POLY_EXISTS
        
        return img, (0, 0, img.shape[1], img.shape[0]), False

    
def process_annotation(img, resized_img,json_path, output_txt_path, upd_img_bbox, hole_model):
    """Process a single annotation file.
    
    Args:
        json_path: Path to LabelMe JSON file
        output_txt_path: Path to output text file
        upd_img_bbox: Updated bounding box after cropping (x1, y1, x2, y2)
        
    Returns:
        Processed image
    """
    shapes = []
    
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
            
        # Calculate cropped image dimensions
        cropped_width = int(upd_img_bbox[2] - upd_img_bbox[0])
        cropped_height = int(upd_img_bbox[3] - upd_img_bbox[1])

        crack_exists = False    
        for shape in data["shapes"]:
            try:
                label = int(shape["label"])
            except (ValueError, TypeError):
                print(f"Invalid label in {json_path}: {shape['label']}")
                return False  # Invalid annotations must not become negative examples.
            
            if label not in CRACK_LABELS and label not in NO_CRACK_LABELS:
                continue
            
            points = np.array(shape["points"], np.int32)
            try:
                if len(points) == 2:
                    x1, y1, x2, y2 = points[0][0], points[0][1], points[1][0], points[1][1]
                elif len(points) > 2:
                    xs = points[:, 0]
                    ys = points[:, 1]
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                else:
                    return False
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Clamp coordinates to panel bbox
                if x1 < upd_img_bbox[0]:
                    x1 = upd_img_bbox[0]
                if y1 < upd_img_bbox[1]:
                    y1 = upd_img_bbox[1]
                if x2 > upd_img_bbox[2]:
                    x2 = upd_img_bbox[2]
                if y2 > upd_img_bbox[3]:
                    y2 = upd_img_bbox[3]

                # Convert to cropped image coordinates (subtract top-left corner)
                x1, y1, x2, y2 = x1 - upd_img_bbox[0], y1 - upd_img_bbox[1], x2 - upd_img_bbox[0], y2 - upd_img_bbox[1]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                if x2 - x1 < 2 or y2 - y1 < 2:
                    print(f"Skipping crop with invalid dimensions: {x2 - x1}x{y2 - y1} {x1},{y1},{x2},{y2} {upd_img_bbox}")
                    continue

                # Extract the cropped region
                crop = resized_img[int(y1):int(y2), int(x1):int(x2)]
                
                # Check if crop contains actual image content (not just black background)
                # Calculate percentage of non-black pixels
                non_black_pixels = np.sum(crop > 0)  # Count pixels with any color > 0
                total_pixels = crop.size
                try:
                    non_black_ratio = non_black_pixels / total_pixels
                except ZeroDivisionError:
                    print(f"Zero division error in {json_path}")
                    continue
                
                # Skip if crop is mostly black background (less than 5% content)
                # if non_black_ratio < 0.05:
                #     print(f"Skipping mostly black crop in {json_path}: {non_black_ratio:.3f} content ratio")
                #     continue
                
                # Determine class label
                if label in CRACK_LABELS:
                    class_id = 1  # Crack
                    crack_exists = True
                elif label in NO_CRACK_LABELS:
                    class_id = 0  # No crack
                else:
                    continue
                
                # Convert to YOLO format using CROPPED image dimensions
                x_center = (x1 + x2) / (2 * cropped_width)
                y_center = (y1 + y2) / (2 * cropped_height)
                width = (x2 - x1) / cropped_width
                height = (y2 - y1) / cropped_height
                
                # Ensure values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                shapes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
            except Exception as e:
                print(f"Error processing shape in {json_path}: {e}")
                return False
        
        if USE_HOLE_MODEL and hole_model is not None:
            hole_preds = hole_model.predict(
                img, conf=HOLE_CONFIDENCE, iou=HOLE_IOU, verbose=False
            )[0]
            hole_boxes = hole_preds.boxes.xyxy.cpu().numpy()
            for hole_box in hole_boxes:
                x1, y1, x2, y2 = hole_box

                # Clamp coordinates to panel bbox
                if x1 < upd_img_bbox[0]:
                    x1 = upd_img_bbox[0]
                if y1 < upd_img_bbox[1]:
                    y1 = upd_img_bbox[1]
                if x2 > upd_img_bbox[2]:
                    x2 = upd_img_bbox[2]
                if y2 > upd_img_bbox[3]:
                    y2 = upd_img_bbox[3]

                # Convert to cropped image coordinates (subtract top-left corner)
                x1, y1, x2, y2 = x1 - upd_img_bbox[0], y1 - upd_img_bbox[1], x2 - upd_img_bbox[0], y2 - upd_img_bbox[1]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                if x2 - x1 < 2 or y2 - y1 < 2:
                    print(f"Skipping crop with invalid dimensions: {x2 - x1}x{y2 - y1} {x1},{y1},{x2},{y2} {upd_img_bbox}")
                    continue

                x_center = (x1 + x2) / (2 * cropped_width)
                y_center = (y1 + y2) / (2 * cropped_height)
                width = (x2 - x1) / cropped_width
                height = (y2 - y1) / cropped_height
                
                # Ensure values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                shapes.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Save annotations to file
        
        if crack_exists or INCLUDE_NEGATIVE_IMAGES:
            with open(output_txt_path, 'w') as file:
                file.writelines(shapes)
            
            return True
        else:
            print(f"No shapes found in {json_path}")
            return False

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def get_panel_id(filename):
    """Extract a panel ID such as P1_S2 from anywhere in the filename."""
    match = re.search(PANEL_ID_PATTERN, Path(filename).stem)
    if match is None:
        raise ValueError(
            f"Panel ID not found in {filename}; expected _P<number>_S<number>"
        )
    return match.group(1)


def find_image_for_json(json_path):
    """Find an image beside a JSON file with the same filename stem."""
    for extension in (".jpg", ".png", ".tiff", ".jpeg"):
        image_path = json_path.with_suffix(extension)
        if image_path.exists():
            return image_path
    return None


def create_yolo_dataset(source_dir, config):
    """Process one flat directory of images and matching JSON annotations."""
    # source_dir = Path(SOURCE_PATH)
    output_dir = source_dir
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    hole_model = YOLO(HOLE_MODEL_PATH) if USE_HOLE_MODEL else None
    checked = saved = skipped_no_panel = skipped_no_crack = errors = 0
    json_files = sorted(
        path for path in source_dir.glob("*.json") if "meta" not in path.name
    )

    for json_path in tqdm(json_files, desc="Creating YOLO dataset"):
        checked += 1
        # Remove this image's previous conversion so changing the negative-image
        # option (or rejecting an annotation) cannot retain a stale example.
        label_path = output_labels / json_path.with_suffix(".txt").name
        output_image_path = output_images / json_path.with_suffix(".png").name
        label_path.unlink(missing_ok=True)
        output_image_path.unlink(missing_ok=True)
        try:
            image_path = find_image_for_json(json_path)
            if image_path is None:
                raise FileNotFoundError(f"Matching image not found for {json_path.name}")

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            panel_id = get_panel_id(json_path.name)
            prefix = json_path.name.split("_")[0]
            camera_id = prefix[2] if prefix.startswith("C0") else prefix[1:3]
            panel_name = MAP().get_panel_name(panel_id)
            gt_json_path = Path(PANEL_GT_PATH) / panel_name / f"{camera_id}.json"

            additional_label_path = (
                Path(SEGMENTATION_LABELS_PATH) / json_path.with_suffix(".txt").name
            )
            if not additional_label_path.exists():
                additional_label_path = None

            cropped_image, crop_bbox, panel_exists = mask_image(
                image,
                json_path,
                additional_label_path=additional_label_path,
                gt_json_path=gt_json_path,
            )
            if not panel_exists:
                skipped_no_panel += 1
                continue

            annotation_created = process_annotation(
                image,
                cropped_image,
                json_path,
                label_path,
                crop_bbox,
                hole_model,
            )
            if not annotation_created:
                skipped_no_crack += 1
                continue

            if not cv2.imwrite(str(output_image_path), cropped_image):
                raise OSError(f"Could not write image: {output_image_path}")
            saved += 1

        except Exception as error:
            errors += 1
            print(f"Error processing {json_path.name}: {error}")

    print(
        "\nFinished"
        f" | checked={checked}"
        f" | saved={saved}"
        f" | no panel mask={skipped_no_panel}"
        f" | no crack={skipped_no_crack}"
        f" | errors={errors}"
    )
    return {
        "checked": checked,
        "saved": saved,
        "no_panel_mask": skipped_no_panel,
        "no_crack": skipped_no_crack,
        "errors": errors,
    }
