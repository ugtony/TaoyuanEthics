# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#  Workaround for Streamlit/PyTorch watcher issue with torch.classes
# -----------------------------------------------------------------------------
import torch
if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []

# -----------------------------------------------------------------------------
#  Standard Libraries
# -----------------------------------------------------------------------------
import cv2
import tempfile
import os
import time
import logging
from collections import defaultdict
import shutil
import datetime
import math
import sys

# -----------------------------------------------------------------------------
#  Third-Party Libraries
# -----------------------------------------------------------------------------
import streamlit as st
from ultralytics import YOLO
import numpy as np

# -----------------------------------------------------------------------------
#  Page Config (Must be the first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="YOLO 物件追蹤 App", layout="wide")

# -----------------------------------------------------------------------------
#  Helper Function for PyInstaller Paths (MUST BE DEFINED EARLY)
# -----------------------------------------------------------------------------
def get_resource_path(relative_path):
    """ 獲取資源的絕對路徑，對開發環境和 PyInstaller 都有效 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# -----------------------------------------------------------------------------
#  Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
#  Model Parameters
# -----------------------------------------------------------------------------
MODEL_TYPES = ["Standard YOLO", "YOLO-World"]

STANDARD_MODEL_CONFIG = {
    "model_path": get_resource_path("models/yolov8n.pt"),
    "target_classes_ids": [0, 1, 2, 3, 5, 7],
    "target_classes_names": {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
    "confidence_threshold": 0.3,
    "display_name": "YOLOv8n (People & Vehicles)"
}

WORLD_MODEL_CONFIG = {
    "model_path": get_resource_path("models/yolov8s-worldv2.pt"),
    "default_prompt": "person, car, bicycle, traffic light, backpack",
    "confidence_threshold": 0.1,
    "display_name": "YOLOv8s-World v2"
}

# -----------------------------------------------------------------------------
#  Processing & Drawing Parameters
# -----------------------------------------------------------------------------
MAX_PROCESSING_FPS = 3.0
# --- 修正 TRACKER_CONFIG 的路徑 ---
TRACKER_CONFIG = get_resource_path("models/bytetrack.yaml")
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
TEXT_COLOR_ON_BG = (0, 0, 0)
TEXT_BG_COLOR = (0, 255, 0)
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
BASE_OUTPUT_DIR = "yolo_detection_results"


# (以下所有函式定義與您的版本相同，此處省略以保持簡潔)
# ... (從 def format_timestamp 開始的所有函式) ...
# ... (整個 Streamlit UI 渲染的程式碼) ...

# -----------------------------------------------------------------------------
#  Helper — Timestamp Formatter
# -----------------------------------------------------------------------------
def format_timestamp(seconds, for_filename=False):
    """Converts seconds to HH:MM:SS or HH_MM_SS string format."""
    if seconds is None or seconds < 0:
        return "00_00_00" if for_filename else "00:00:00"
    td = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds_val = divmod(remainder, 60)
    if for_filename:
        return f"{hours:02d}_{minutes:02d}_{seconds_val:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds_val:02d}"

# -----------------------------------------------------------------------------
#  Helper — Load Model
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_unified(model_path):
    try:
        logging.info(f"Loading model: {model_path}")
        model = YOLO(model_path)
        logging.info(f"Model {model_path} loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
        logging.exception(f"Error loading model '{model_path}': {e}")
        return None

# -----------------------------------------------------------------------------
#  Helper — Draw Bounding Box
# -----------------------------------------------------------------------------
def draw_bounding_box_unified(frame, box, track_id, class_name, conf):
    """Draws a single bounding box on a copy of the frame."""
    img = frame.copy()
    x1, y1, x2, y2 = map(int, box)
    label = f"ID:{track_id} {class_name} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
    (w, h), _ = cv2.getTextSize(label, TEXT_FONT, TEXT_FONT_SCALE, TEXT_THICKNESS)
    label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
    label_x1 = max(0, x1)
    label_x2 = label_x1 + w
    cv2.rectangle(img, (label_x1, label_y - h - 5), (label_x2, label_y), TEXT_BG_COLOR, -1)
    cv2.putText(img, label, (label_x1, label_y - 3), TEXT_FONT, TEXT_FONT_SCALE, TEXT_COLOR_ON_BG, TEXT_THICKNESS, cv2.LINE_AA)
    return img

# -----------------------------------------------------------------------------
#  Helper — On-Demand Frame Extraction from Video
# -----------------------------------------------------------------------------
def extract_frames_by_indices(video_path, frame_indices):
    if not frame_indices:
        return {}
    extracted_frames = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video file for frame extraction: {video_path}")
        return {}
    sorted_indices = sorted(list(set(frame_indices)))
    for index in sorted_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            extracted_frames[index] = frame
        else:
            logging.warning(f"Could not read frame at index {index} from {video_path}. It may be out of bounds.")
    cap.release()
    return extracted_frames

# -----------------------------------------------------------------------------
#  Helper — Process Video (Detect + Track)
# -----------------------------------------------------------------------------
def process_video_unified(video_path, model, is_world_model, current_model_settings,
                          yolo_world_custom_classes, confidence_thresh, progress_bar_element):
    tracked_object_info = defaultdict(list)
    representative_frames = {}
    max_box_areas = defaultdict(float)
    if not hasattr(progress_bar_element, "progress"):
        class DummyProgressBar:
            def progress(self, *_args, **_kw): pass
            def empty(self): pass
        progress_bar_element = DummyProgressBar()
        logging.warning("process_video_unified: No valid progress_bar_element passed, using dummy.")
    try:
        active_classes_for_tracking_param = None
        class_name_source_map_or_list = {}
        if is_world_model:
            if not yolo_world_custom_classes:
                st.warning("YOLO-World model requires at least one class to detect.")
                logging.warning("YOLO-World: Processing attempted without specified classes.")
                return {}, {}
            model.set_classes(yolo_world_custom_classes)
            class_name_source_map_or_list = yolo_world_custom_classes
            active_classes_for_tracking_param = list(range(len(yolo_world_custom_classes)))
            logging.info(f"YOLO-World: Set classes: {', '.join(yolo_world_custom_classes)}, Tracking indices: {active_classes_for_tracking_param}")
        else:
            active_classes_for_tracking_param = current_model_settings["target_classes_ids"]
            class_name_source_map_or_list = current_model_settings["target_classes_names"]
            logging.info(f"Standard YOLO: Using fixed class IDs: {active_classes_for_tracking_param}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Cannot open video file.")
            logging.error(f"Cannot open video file: {video_path}")
            return {}, {}
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            logging.warning(f"Video FPS read as zero or negative ({video_fps}). Timestamps may be inaccurate. Defaulting to 30 FPS.")
            video_fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pb_instance = progress_bar_element.progress(0, text="Processing video…")
        start_time = time.time()
        frame_idx = 0
        last_processed_time = -1.0
        time_increment = 1.0 / MAX_PROCESSING_FPS
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            current_timestamp_seconds = frame_idx / video_fps
            if current_timestamp_seconds >= last_processed_time + time_increment:
                last_processed_time = current_timestamp_seconds
                track_params = dict(
                    source=frame, tracker=TRACKER_CONFIG, conf=confidence_thresh,
                    persist=True, verbose=False, classes=active_classes_for_tracking_param
                )
                results = model.track(**track_params)
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes_coords = results[0].boxes.xyxy.cpu().numpy()
                    track_ids_list = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids_from_model = results[0].boxes.cls.cpu().numpy().astype(int)
                    confidences_list = results[0].boxes.conf.cpu().numpy()
                    for current_box, track_id, cls_id, conf_score in zip(boxes_coords, track_ids_list, class_ids_from_model, confidences_list):
                        if conf_score < confidence_thresh: continue
                        object_class_name = f"class_{cls_id}"
                        if is_world_model:
                            if 0 <= cls_id < len(class_name_source_map_or_list):
                                object_class_name = class_name_source_map_or_list[cls_id]
                            else:
                                logging.error(f"YOLO-World: Detected unexpected class ID {cls_id}")
                                continue
                        else:
                            object_class_name = class_name_source_map_or_list.get(cls_id, f"class_{cls_id}")
                        timestamp_str_display = format_timestamp(current_timestamp_seconds, for_filename=False)
                        tracked_object_info[track_id].append((frame_idx, tuple(current_box), conf_score, timestamp_str_display))
                        x1, y1, x2, y2 = current_box
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > max_box_areas[track_id]:
                            max_box_areas[track_id] = box_area
                            representative_frames[track_id] = (frame.copy(), object_class_name, tuple(current_box), conf_score, timestamp_str_display)
            progress_percent = int((frame_idx + 1) / total_frames * 100) if total_frames > 0 else 0
            processing_fps = (frame_idx + 1) / (time.time() - start_time + 1e-6)
            pb_instance.progress(progress_percent, text=f"Scanning video… {progress_percent}% (Scan FPS: {processing_fps:.2f})")
            frame_idx += 1
        if cap: cap.release()
        pb_instance.progress(100, text="Video processing complete!")
        time.sleep(0.5)
        progress_bar_element.empty()
        logging.info("Video processing finished successfully.")
        return tracked_object_info, representative_frames
    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        logging.exception(f"An unexpected error occurred in process_video_unified: {e}")
        if hasattr(progress_bar_element, "empty"): progress_bar_element.empty()
        return {}, {}

# -----------------------------------------------------------------------------
#  Helper — Persist Annotated Results (for Download)
# -----------------------------------------------------------------------------
def persist_annotated_results(output_target_dir, representative_frames_data,
                              tracked_data_info, original_video_path):
    if not representative_frames_data and not tracked_data_info:
        logging.info("No results to save to disk.")
        return False
    try:
        if os.path.exists(output_target_dir):
            logging.info(f"Output directory {output_target_dir} exists, removing...")
            shutil.rmtree(output_target_dir, ignore_errors=True)
        os.makedirs(output_target_dir, exist_ok=True)
        logging.info(f"Saving annotated results to: {output_target_dir}")
        rep_frames_dir = os.path.join(output_target_dir, "representative_frames")
        os.makedirs(rep_frames_dir, exist_ok=True)
        if representative_frames_data:
            for track_id, (frame_bgr, class_name, box, conf, timestamp_str) in representative_frames_data.items():
                annotated_frame = draw_bounding_box_unified(frame_bgr, box, track_id, class_name, conf)
                safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
                timestamp_for_file = format_timestamp(None)
                if timestamp_str and ":" in timestamp_str:
                     try:
                        parts = list(map(int, timestamp_str.split(':')))
                        seconds_from_str = parts[0]*3600 + parts[1]*60 + parts[2]
                        timestamp_for_file = format_timestamp(seconds_from_str, for_filename=True)
                     except (ValueError, IndexError): logging.warning(f"Malformed timestamp for rep frame: {timestamp_str}")
                output_filename = f"rep_track_{track_id:03d}_{safe_class_name}_time_{timestamp_for_file}_conf_{conf:.2f}.jpg"
                output_filepath = os.path.join(rep_frames_dir, output_filename)
                cv2.imwrite(output_filepath, annotated_frame)
        logging.info("Saved annotated representative frames.")
        tracked_details_dir = os.path.join(output_target_dir, "tracked_object_details")
        os.makedirs(tracked_details_dir, exist_ok=True)
        if tracked_data_info:
            all_required_indices = set()
            for frame_entries in tracked_data_info.values():
                for entry in frame_entries:
                    all_required_indices.add(entry[0])
            logging.info(f"Extracting {len(all_required_indices)} unique frames for download.")
            all_extracted_frames = extract_frames_by_indices(original_video_path, all_required_indices)
            logging.info("Frame extraction for download complete.")
            for track_id, frame_entries in tracked_data_info.items():
                class_name_for_folder = "unknown_class"
                if track_id in representative_frames_data:
                    class_name_for_folder = representative_frames_data[track_id][1]
                safe_class_name_for_folder = "".join(c if c.isalnum() else "_" for c in class_name_for_folder)
                track_specific_dir = os.path.join(tracked_details_dir, f"track_{track_id:03d}_{safe_class_name_for_folder}")
                os.makedirs(track_specific_dir, exist_ok=True)
                for idx, (frame_idx_detail, box, conf, timestamp_str) in enumerate(frame_entries):
                    original_frame_bgr = all_extracted_frames.get(frame_idx_detail)
                    if original_frame_bgr is not None:
                        annotated_frame_for_detail = draw_bounding_box_unified(original_frame_bgr, box, track_id, class_name_for_folder, conf)
                        timestamp_for_file_detail = format_timestamp(None)
                        if timestamp_str and ":" in timestamp_str:
                             try:
                                parts_detail = list(map(int, timestamp_str.split(':')))
                                seconds_from_str_detail = parts_detail[0]*3600 + parts_detail[1]*60 + parts_detail[2]
                                timestamp_for_file_detail = format_timestamp(seconds_from_str_detail, for_filename=True)
                             except (ValueError, IndexError): logging.warning(f"Malformed timestamp for detail frame: {timestamp_str}")
                        detail_frame_filename = f"frame_{idx:05d}_time_{timestamp_for_file_detail}_conf_{conf:.2f}.jpg"
                        detail_frame_save_path = os.path.join(track_specific_dir, detail_frame_filename)
                        cv2.imwrite(detail_frame_save_path, annotated_frame_for_detail)
        logging.info("Saved annotated detailed tracking frames.")
        return True
    except Exception as e:
        logging.exception(f"Error saving annotated results to {output_target_dir}: {e}")
        return False

# -----------------------------------------------------------------------------
#  Helper — Trigger ZIP Preparation
# -----------------------------------------------------------------------------
def trigger_zip_preparation():
    ss = st.session_state
    ss.prepared_zip_bytes = None
    if not (ss.video_processed and ss.representative_frames and ss.uploaded_file_name and ss.video_path):
        logging.warning("trigger_zip_preparation call missing required session state data.")
        st.toast("Cannot prepare download: Missing processed data.", icon="⚠️")
        return
    video_name_without_ext = os.path.splitext(ss.uploaded_file_name)[0]
    safe_video_name = "".join(c if c.isalnum() else "_" for c in video_name_without_ext)
    final_results_output_dir = os.path.join(BASE_OUTPUT_DIR, f"{safe_video_name}_results_hierarchical")
    save_success = persist_annotated_results(
        final_results_output_dir,
        ss.representative_frames,
        ss.tracked_data,
        ss.video_path
    )
    if not save_success:
        logging.error("Failed to persist detailed results in persist_annotated_results.")
        st.toast("Error: Failed to save results before zipping.", icon="🚨")
        return
    ss.user_output_dir_path = os.path.abspath(final_results_output_dir)
    logging.info(f"Annotated results saved to (for zipping): {ss.user_output_dir_path}")
    archive_temp_basename = os.path.join(tempfile.gettempdir(), f"{safe_video_name}_results_archive")
    generated_zip_full_path = None
    try:
        generated_zip_full_path = shutil.make_archive(
            base_name=archive_temp_basename,
            format='zip',
            root_dir=os.path.dirname(final_results_output_dir),
            base_dir=os.path.basename(final_results_output_dir)
        )
    except Exception as e_zip:
        logging.exception(f"Error creating ZIP file: {e_zip}")
        st.toast(f"Error: Failed to create archive: {e_zip}", icon="🚨")
        return
    if generated_zip_full_path and os.path.exists(generated_zip_full_path):
        with open(generated_zip_full_path, "rb") as f:
            ss.prepared_zip_bytes = f.read()
        st.toast("Files are ready for download!", icon="✅")
        try:
            os.remove(generated_zip_full_path)
        except Exception as e:
            logging.warning(f"Could not clean up temp zip file {generated_zip_full_path}: {e}")
    else:
        logging.error("Could not read the newly generated ZIP file.")
        st.toast("Error: Could not read the archive file.", icon="🚨")


# -----------------------------------------------------------------------------
#  Streamlit Interface Setup
# -----------------------------------------------------------------------------
ss = st.session_state
_default_session_values = {
    "selected_model_type": MODEL_TYPES[0],
    "active_model_config": STANDARD_MODEL_CONFIG,
    "loaded_model_object": None,
    "tracked_data": None,
    "representative_frames": None,
    "selected_track_id": None,
    "video_processed": False,
    "uploaded_file_name": None,
    "video_path": None,
    "current_prompt_world": WORLD_MODEL_CONFIG["default_prompt"],
    "last_processed_settings": "",
    "view_mode": "all_objects",
    "confidence_threshold": STANDARD_MODEL_CONFIG["confidence_threshold"],
    "user_output_dir_path": None,
    "prepared_zip_bytes": None,
    "all_objects_current_page": 1,
    "all_objects_items_per_page": 48,
}
for key, value in _default_session_values.items():
    if key not in ss:
        ss[key] = value

main_area_progress_bar_placeholder = st.empty()

def reset_app_state():
    """Resets state when model or video changes."""
    ss.tracked_data = None
    ss.representative_frames = None
    ss.selected_track_id = None
    ss.video_processed = False
    ss.last_processed_settings = ""
    ss.view_mode = 'all_objects'
    ss.user_output_dir_path = None
    ss.all_objects_current_page = 1
    ss.prepared_zip_bytes = None

with st.sidebar:
    st.header("⚙️ 控制面板")
    previous_selected_model_type = ss.selected_model_type
    ss.selected_model_type = st.radio(
        "選擇模型類型:",
        MODEL_TYPES,
        key="model_type_radio_selector",
        horizontal=True
    )
    is_currently_world_model = (ss.selected_model_type == "YOLO-World")
    if previous_selected_model_type != ss.selected_model_type:
        ss.active_model_config = WORLD_MODEL_CONFIG if is_currently_world_model else STANDARD_MODEL_CONFIG
        ss.loaded_model_object = None
        ss.confidence_threshold = ss.active_model_config["confidence_threshold"]
        if is_currently_world_model:
            ss.current_prompt_world = WORLD_MODEL_CONFIG["default_prompt"]
        reset_app_state()
        st.rerun()
    st.caption(f"使用模型: {ss.active_model_config.get('display_name', ss.active_model_config['model_path'])}")
    if is_currently_world_model:
        ss.current_prompt_world = st.text_area(
            "輸入要偵測的物件 (以逗號分隔):",
            value=ss.current_prompt_world,
            height=100,
            key="world_model_prompt_input"
        )
    else:
        fixed_classes_display = ", ".join(STANDARD_MODEL_CONFIG['target_classes_names'].values())
        st.info(f"固定偵測目標: {fixed_classes_display}")
    ss.confidence_threshold = st.slider(
        "信賴度閾值:",
        0.05, 0.95,
        ss.confidence_threshold,
        0.05,
        key="confidence_level_slider"
    )
    uploaded_video_file = st.file_uploader(
        "選擇影片檔案",
        ["mp4", "avi", "mov", "mkv"],
        key="video_file_uploader_widget"
    )
    if uploaded_video_file is not None and ss.uploaded_file_name != uploaded_video_file.name:
        reset_app_state()
        ss.uploaded_file_name = uploaded_video_file.name
        if ss.video_path and os.path.exists(ss.video_path):
            try:
                os.remove(ss.video_path)
            except OSError as e:
                logging.warning(f"Could not remove old temp video file: {ss.video_path}, error: {e}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video_file.name)[1]) as tmp_vid_file:
                tmp_vid_file.write(uploaded_video_file.getvalue())
                ss.video_path = tmp_vid_file.name
            logging.info(f"New temp video file created: {ss.video_path}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to create temporary video file: {e}")
            logging.exception(f"Error creating temp video file: {e}")
            ss.video_path = None
    if ss.video_path:
        if ss.loaded_model_object is None:
            model_path_for_loading = ss.active_model_config['model_path']
            with st.spinner(f"正在載入模型 ({model_path_for_loading})..."):
                ss.loaded_model_object = load_model_unified(model_path_for_loading)
                if ss.loaded_model_object is None:
                    st.error(f"模型 {model_path_for_loading} 載入失敗，無法處理影片。")
                else:
                    st.success(f"模型 {model_path_for_loading} 載入成功！")
                    st.rerun()
        if ss.loaded_model_object:
            current_processing_config_summary = (
                f"Model: {ss.selected_model_type} | "
                f"Prompt: {ss.current_prompt_world if is_currently_world_model else 'Standard Predefined'} | "
                f"Confidence: {ss.confidence_threshold:.2f}"
            )
            button_label = "🚀 開始處理影片"
            if ss.video_processed and ss.last_processed_settings != current_processing_config_summary:
                button_label = "🔄 使用新設定重新處理"
            elif ss.video_processed:
                button_label = "✅ 處理完成"
            if st.button(button_label, use_container_width=True, type="primary", key="process_button_key", disabled=ss.video_processed and ss.last_processed_settings == current_processing_config_summary):
                yolo_world_custom_classes_list = []
                if is_currently_world_model:
                    yolo_world_custom_classes_list = [c.strip() for c in ss.current_prompt_world.split(',') if c.strip()]
                    if not yolo_world_custom_classes_list:
                        st.warning("YOLO-World 需要至少一個有效提示詞！")
                        st.stop()
                reset_app_state()
                with st.spinner(f"{ss.selected_model_type} 影片處理中…"):
                    tracked_data_result, representative_frames_result = process_video_unified(
                        ss.video_path,
                        ss.loaded_model_object,
                        is_currently_world_model,
                        ss.active_model_config,
                        yolo_world_custom_classes_list,
                        ss.confidence_threshold,
                        main_area_progress_bar_placeholder
                    )
                    ss.tracked_data = tracked_data_result
                    ss.representative_frames = representative_frames_result
                    ss.video_processed = True
                    ss.last_processed_settings = current_processing_config_summary
                st.success("影片處理完成！結果已可供檢視。")
                st.rerun()
    if ss.video_processed:
        st.markdown("---")
        st.subheader("👁️ 檢視選項")
        current_view_mode = ss.view_mode
        if st.button("所有追蹤物件",
                      type="primary" if ss.view_mode == 'all_objects' else "secondary",
                      use_container_width=True, key="view_all_objects_btn_key"):
            if current_view_mode != 'all_objects':
                ss.all_objects_current_page = 1
            ss.view_mode = 'all_objects'
            ss.selected_track_id = None
            st.rerun()
        if ss.representative_frames:
            sorted_track_ids = sorted(ss.representative_frames.keys())
            if st.button("特定物件所有畫面",
                          type="primary" if ss.view_mode == 'single_object' else "secondary",
                          use_container_width=True, key="view_specific_object_btn_key"):
                ss.view_mode = 'single_object'
                ss.selected_track_id = sorted_track_ids[0] if sorted_track_ids else None
                st.rerun()
            if ss.view_mode == 'single_object' and sorted_track_ids:
                selected_id_choice = st.selectbox(
                    "選擇物件 ID:",
                    sorted_track_ids,
                    index=sorted_track_ids.index(ss.selected_track_id) if ss.selected_track_id in sorted_track_ids else 0,
                    format_func=lambda tid_key: f"ID:{tid_key} ({ss.representative_frames[tid_key][1]})",
                    key="select_specific_object_id_selectbox"
                )
                if selected_id_choice != ss.selected_track_id:
                    ss.selected_track_id = selected_id_choice
                    st.rerun()
        st.markdown("---")
        st.subheader("📥 下載結果")
        if ss.representative_frames and ss.uploaded_file_name:
            if st.button("準備下載", use_container_width=True, key="prepare_download_button"):
                with st.spinner("正在準備並壓縮結果檔案，請稍候..."):
                    trigger_zip_preparation()
                st.rerun()
            _video_name_no_ext_dl = os.path.splitext(ss.uploaded_file_name)[0]
            _safe_video_name_dl = "".join(c if c.isalnum() else "_" for c in _video_name_no_ext_dl)
            _download_zip_filename = f"{_safe_video_name_dl}_results.zip"
            st.download_button(
                label="下載標註結果 (ZIP)",
                data=ss.prepared_zip_bytes if ss.prepared_zip_bytes else b"",
                file_name=_download_zip_filename,
                mime="application/zip",
                key="download_annotated_results_zip_button",
                use_container_width=True,
                help="請先點擊上方的「準備下載」，按鈕啟用後即可下載。",
                disabled=not ss.prepared_zip_bytes
            )
            if ss.user_output_dir_path and os.path.exists(ss.user_output_dir_path):
                st.caption(f"提示：未壓縮的結果儲存於 {ss.user_output_dir_path}")

# -----------------------------------------------------------------------------
#  Main Area Content
# -----------------------------------------------------------------------------
st.title("🎬 YOLO 通用物件偵測與追蹤")

if not ss.video_path:
    st.info("👋 歡迎！請在左側控制面板選擇模型類型、設定偵測目標並上傳影片檔案。")
    st.markdown(f"""
        - **Standard YOLO**: 偵測預定義物件 (例如: {", ".join(STANDARD_MODEL_CONFIG['target_classes_names'].values())})。
        - **YOLO-World**: 輸入您想偵測的任意物件名稱 (例如: `a red apple, a blue car`)。
    """)
elif not ss.loaded_model_object:
     st.warning("模型正在載入或載入失敗。請檢查 Sidebar。")
else:
    video_col, empty_col = st.columns([2, 1])
    with video_col:
        st.subheader("🎞️ 影片預覽")
        st.video(ss.video_path)
    if not ss.video_processed:
        st.info("影片已上傳。請點擊左側 Sidebar 的「🚀 開始處理影片」按鈕。")

if ss.video_processed:
    processed_model_type_disp, processed_prompt_text_disp, processed_conf_text_disp = "N/A", "N/A", "N/A"
    if ss.last_processed_settings:
        parts = {p.split(': ', 1)[0].lower(): p.split(': ', 1)[1] for p in ss.last_processed_settings.split(' | ')}
        processed_model_type_disp = parts.get('model', 'N/A')
        processed_prompt_text_disp = parts.get('prompt', 'N/A')
        processed_conf_text_disp = parts.get('confidence', 'N/A')
    if not ss.representative_frames:
        st.info(f"處理完成。模型 {processed_model_type_disp} 在 '{processed_prompt_text_disp}' 的條件下（信賴度 ≥ {processed_conf_text_disp}）未偵測到任何物件。")
    else:
        if ss.view_mode == 'all_objects':
            st.header("📊 所有追蹤物件 (代表畫面)")
            st.write(f"模型: {processed_model_type_disp} | 目標: '{processed_prompt_text_disp}' | 最低信賴度: {processed_conf_text_disp}")
            st.write(f"總共偵測並追蹤到 {len(ss.representative_frames)} 個獨立物件。")
            total_items = len(ss.representative_frames)
            items_per_page = st.select_slider("每頁顯示物件數:", options=[12, 24, 48, 96], value=ss.all_objects_items_per_page)
            if items_per_page != ss.all_objects_items_per_page:
                ss.all_objects_items_per_page = items_per_page
                ss.all_objects_current_page = 1
                st.rerun()
            total_pages = max(1, math.ceil(total_items / ss.all_objects_items_per_page))
            ss.all_objects_current_page = max(1, min(ss.all_objects_current_page, total_pages))
            start_index = (ss.all_objects_current_page - 1) * ss.all_objects_items_per_page
            end_index = start_index + ss.all_objects_items_per_page
            sorted_rep_ids = sorted(list(ss.representative_frames.keys()))
            current_page_track_ids = sorted_rep_ids[start_index:end_index]
            num_cols = st.slider("每行顯示物件數 (網格視圖):", 2, 8, 4)
            grid_cols = st.columns(num_cols)
            for i, track_id_val in enumerate(current_page_track_ids):
                with grid_cols[i % num_cols]:
                    frame_bgr, c_name, box, conf, ts = ss.representative_frames[track_id_val]
                    img_with_box = draw_bounding_box_unified(frame_bgr, box, track_id_val, c_name, conf)
                    st.image(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB), caption=f"ID: {track_id_val} ({c_name}, {conf:.2f}) at {ts}", use_container_width=True)
                    if st.button(f"檢視 ID {track_id_val} 所有畫面", key=f"view_all_btn_{track_id_val}", use_container_width=True):
                        ss.selected_track_id = track_id_val
                        ss.view_mode = 'single_object'
                        st.rerun()
            if total_pages > 1:
                st.markdown("---")
                nav_cols = st.columns([1, 2, 1])
                if nav_cols[0].button("⬅️ 上一頁", disabled=(ss.all_objects_current_page <= 1)):
                    ss.all_objects_current_page -= 1
                    st.rerun()
                nav_cols[1].markdown(f"<p style='text-align: center; margin-top: 0.5em;'>第 {ss.all_objects_current_page} / {total_pages} 頁</p>", unsafe_allow_html=True)
                if nav_cols[2].button("下一頁 ➡️", disabled=(ss.all_objects_current_page >= total_pages)):
                    ss.all_objects_current_page += 1
                    st.rerun()
        elif ss.view_mode == 'single_object' and ss.selected_track_id is not None:
            current_id = ss.selected_track_id
            if current_id in ss.tracked_data and current_id in ss.representative_frames:
                frames_info = ss.tracked_data[current_id]
                _, c_name_header, _, conf_header, ts_header = ss.representative_frames[current_id]
                st.header(f"🖼️ 物件 ID: {current_id} ({c_name_header}) 的所有畫面")
                st.write(f"此物件出現的總幀數: {len(frames_info)}。")
                frame_indices_to_show = [info[0] for info in frames_info]
                with st.spinner(f"正在擷取 {len(frame_indices_to_show)} 個畫面以供檢視..."):
                    extracted_frames_for_viewing = extract_frames_by_indices(ss.video_path, frame_indices_to_show)
                cols_per_row = st.number_input("每行顯示幀數:", min_value=2, max_value=10, value=4)
                detailed_cols = st.columns(cols_per_row)
                for idx, (frame_idx_detail, box_detail, conf_detail, ts_detail) in enumerate(frames_info):
                    with detailed_cols[idx % cols_per_row]:
                        frame_bgr_detail = extracted_frames_for_viewing.get(frame_idx_detail)
                        if frame_bgr_detail is not None:
                            annotated_detail = draw_bounding_box_unified(frame_bgr_detail, box_detail, current_id, c_name_header, conf_detail)
                            st.image(cv2.cvtColor(annotated_detail, cv2.COLOR_BGR2RGB), caption=f"幀 {idx+1} ({ts_detail}) 信賴度: {conf_detail:.2f}", use_container_width=True)
                        else:
                            st.warning(f"無法載入幀索引 {frame_idx_detail}。")
            else:
                st.warning(f"找不到軌跡 ID {ss.selected_track_id} 的資料。請從左側選單選擇有效的物件 ID 或返回總覽。")
                ss.view_mode = 'all_objects'
                ss.selected_track_id = None
                st.rerun()

# -----------------------------------------------------------------------------
#  Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("由 Ultralytics YOLO 和 Streamlit 驅動")
