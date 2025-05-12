# -*- coding: utf-8 -*-
"""
YOLO (Standard / YOLO‑World) 物件偵測 + 追蹤 — Streamlit App
==========================================================
此版本基於 merged_yolo_app_v6_optimized，修正了 YOLO-World 模式下
因嘗試呼叫不存在的 model.reset_classes() 方法而導致的 AttributeError。
保留了傳遞 classes=list(range(len(custom_prompt))) 給 model.track() 的修正。
"""

# -----------------------------------------------------------------------------
#  Workaround for Streamlit/PyTorch watcher issue with torch.classes
# -----------------------------------------------------------------------------
import torch
if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []  # 必須放在最頂端

# -----------------------------------------------------------------------------
#  標準函式庫
# -----------------------------------------------------------------------------
import cv2
import tempfile
import os
import time
import logging
from collections import defaultdict
import shutil # 用於刪除資料夾及建立壓縮檔
import datetime # 用於時間戳轉換

# -----------------------------------------------------------------------------
#  第三方函式庫
# -----------------------------------------------------------------------------
import streamlit as st
from ultralytics import YOLO
import numpy as np

# -----------------------------------------------------------------------------
#  Logging 設定
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------------------------------------------------------
#  模型參數
# -----------------------------------------------------------------------------
MODEL_TYPES = ["Standard YOLO", "YOLO-World"]

STANDARD_MODEL_CONFIG = {
    "model_path": "yolov8n.pt",
    "target_classes_ids": [0, 1, 2, 3, 5, 7], # 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck
    "target_classes_names": {
        0: "Person", 
        1: "Bicycle", 
        2: "Car", 
        3: "Motorcycle", 
        5: "Bus", 
        7: "Truck"
    },
    "confidence_threshold": 0.3,
    "display_name": "YOLOv8n (人車相關)"
}

WORLD_MODEL_CONFIG = {
    "model_path": "yolov8s-worldv2.pt",
    "default_prompt": "person, car, bicycle, traffic light, backpack", 
    "confidence_threshold": 0.1,
    "display_name": "YOLOv8s-World v2"
}

# -----------------------------------------------------------------------------
#  追蹤 / 繪圖 參數
# -----------------------------------------------------------------------------
TRACKER_CONFIG = "bytetrack.yaml"
BOX_COLOR = (0, 255, 0)        # BGR
BOX_THICKNESS = 2
TEXT_COLOR_ON_BG = (0, 0, 0)   # 黑色文字
TEXT_BG_COLOR = (0, 255, 0)    # 綠色背景
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- 輸出資料夾設定 ---
BASE_OUTPUT_DIR = "yolo_detection_results" 

# -----------------------------------------------------------------------------
#  Helper — 時間戳格式化
# -----------------------------------------------------------------------------
def format_timestamp(seconds, for_filename=False):
    """將秒數轉換為 HH:MM:SS 或 HH_MM_SS 格式的字串。"""
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
#  Helper — 載入模型
# -----------------------------------------------------------------------------
@st.cache_resource 
def load_model_unified(model_path):
    """載入 YOLO 模型 (通用於 Standard YOLO 和 YOLO-World)。"""
    try:
        logging.info(f"正在載入模型: {model_path}")
        model = YOLO(model_path) 
        logging.info(f"模型 {model_path} 載入成功。")
        return model
    except Exception as e:
        st.error(f"載入模型 '{model_path}' 失敗: {e}")
        logging.exception(f"載入模型 '{model_path}' 時發生錯誤: {e}") 
        return None

# -----------------------------------------------------------------------------
#  Helper — 繪製邊界框
# -----------------------------------------------------------------------------
def draw_bounding_box_unified(frame, box, track_id, class_name, conf):
    """在指定的幀上繪製單一物件的邊界框和標籤 (包含信賴度)。"""
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
#  Helper — 處理影片（偵測 + 追蹤）
# -----------------------------------------------------------------------------
def process_video_unified(video_path, model, is_world_model, current_model_settings,
                          yolo_world_custom_classes, confidence_thresh, progress_bar_element,
                          temp_frames_base_dir): 
    tracked_object_frames = defaultdict(list) 
    representative_frames = {} 

    if not hasattr(progress_bar_element, "progress"):
        class DummyProgressBar: 
            def progress(self, *_args, **_kw): pass
            def empty(self): pass
        progress_bar_element = DummyProgressBar()
        logging.warning("process_video_unified: 未傳入有效的 progress_bar_element，使用虛擬元件。")

    session_temp_frames_dir = tempfile.mkdtemp(dir=temp_frames_base_dir)
    logging.info(f"為本次處理建立暫存幀資料夾: {session_temp_frames_dir}")

    video_fps = 0.0 

    try:
        active_classes_for_tracking_param = None 
        class_name_source_map_or_list = {} 

        if is_world_model:
            if not yolo_world_custom_classes:
                st.warning("YOLO-World 模型需要至少一個偵測目標。")
                logging.warning("YOLO-World: 嘗試處理但未提供偵測目標。")
                return {}, {}, session_temp_frames_dir 
            
            # model.reset_classes() # --- 移除此行 ---
            model.set_classes(yolo_world_custom_classes) 
            class_name_source_map_or_list = yolo_world_custom_classes 
            active_classes_for_tracking_param = list(range(len(yolo_world_custom_classes)))
            logging.info(f"YOLO-World: 設定偵測目標: {', '.join(yolo_world_custom_classes)}, 追蹤索引: {active_classes_for_tracking_param}")
        else: # Standard YOLO
            active_classes_for_tracking_param = current_model_settings["target_classes_ids"]
            class_name_source_map_or_list = current_model_settings["target_classes_names"] 
            logging.info(f"Standard YOLO: 使用固定類別 IDs: {active_classes_for_tracking_param}")

        cap = cv2.VideoCapture(video_path) 
        if not cap.isOpened():
            st.error("無法開啟影片檔案。")
            logging.error(f"無法開啟影片檔案: {video_path}")
            return {}, {}, session_temp_frames_dir

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            logging.warning(f"影片 FPS 讀取錯誤或為零 ({video_fps})，時間戳可能不準確。將使用預設 FPS 30。")
            video_fps = 30.0 

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        pb_instance = progress_bar_element.progress(0, text="正在處理影片…") 
        start_time = time.time() 
        frame_idx = 0 
        saved_frame_counter = defaultdict(int) 

        while True:
            ok, frame = cap.read() 
            if not ok: 
                break
            
            current_timestamp_seconds = frame_idx / video_fps
            timestamp_str_display = format_timestamp(current_timestamp_seconds, for_filename=False)
            
            track_params = dict(
                source=frame, 
                tracker=TRACKER_CONFIG,
                conf=confidence_thresh, 
                persist=True,       
                verbose=False,
                classes=active_classes_for_tracking_param 
            )
            
            results = model.track(**track_params) 

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_coords = results[0].boxes.xyxy.cpu().numpy()
                track_ids_list = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids_from_model = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences_list = results[0].boxes.conf.cpu().numpy()

                for current_box, track_id, cls_id, conf_score in zip(boxes_coords, track_ids_list, class_ids_from_model, confidences_list):
                    if conf_score < confidence_thresh: 
                        continue
                    
                    object_class_name = f"class_{cls_id}" 
                    if is_world_model:
                        if 0 <= cls_id < len(class_name_source_map_or_list):
                            object_class_name = class_name_source_map_or_list[cls_id] 
                        else:
                             logging.error(f"YOLO-World: 偵測到預期外的類別 ID {cls_id} (有效索引範圍: 0-{len(class_name_source_map_or_list)-1})")
                             continue 
                    else: 
                        object_class_name = class_name_source_map_or_list.get(cls_id, f"class_{cls_id}")
                    
                    track_frame_dir = os.path.join(session_temp_frames_dir, f"track_{track_id}")
                    os.makedirs(track_frame_dir, exist_ok=True)
                    
                    frame_filename = f"frame_{saved_frame_counter[track_id]:05d}.jpg"
                    frame_save_path = os.path.join(track_frame_dir, frame_filename)
                    cv2.imwrite(frame_save_path, frame) 
                    saved_frame_counter[track_id] += 1
                    
                    tracked_object_frames[track_id].append((frame_save_path, tuple(current_box), conf_score, timestamp_str_display))
                    
                    if track_id not in representative_frames:
                        representative_frames[track_id] = (frame.copy(), object_class_name, tuple(current_box), conf_score, timestamp_str_display)
            
            frame_idx += 1 
            progress_percent = int(frame_idx / total_frames * 100) if total_frames > 0 else 0
            processing_fps = frame_idx / (time.time() - start_time + 1e-6)
            pb_instance.progress(progress_percent, text=f"處理中… {progress_percent}% (FPS: {processing_fps:.2f})")

        if cap: cap.release() 
        pb_instance.progress(100, text="影片處理完成！")
        time.sleep(1) 
        progress_bar_element.empty() 
        logging.info("影片處理完成。")
        return tracked_object_frames, representative_frames, session_temp_frames_dir 

    except Exception as e:
        st.error(f"處理影片時發生錯誤: {e}")
        logging.exception(f"處理影片時發生未預期錯誤: {e}") 
        if hasattr(progress_bar_element, "empty"): 
            progress_bar_element.empty()
        if os.path.exists(session_temp_frames_dir):
            try:
                shutil.rmtree(session_temp_frames_dir)
                logging.info(f"處理錯誤，已清理暫存幀資料夾: {session_temp_frames_dir}")
            except Exception as cleanup_err:
                logging.error(f"處理錯誤後清理暫存幀資料夾 {session_temp_frames_dir} 失敗: {cleanup_err}")
        return {}, {}, None 

# -----------------------------------------------------------------------------
#  Helper — 儲存辨識結果 (階層式)
# -----------------------------------------------------------------------------
def save_detection_results_hierarchical(base_output_dir, video_filename, 
                                        representative_frames_data, tracked_data_paths):
    if not representative_frames_data and not tracked_data_paths:
        logging.info("沒有任何結果可儲存。")
        return None

    video_name_without_ext = os.path.splitext(video_filename)[0]
    safe_video_name = "".join(c if c.isalnum() else "_" for c in video_name_without_ext)
    main_results_dir = os.path.join(base_output_dir, f"{safe_video_name}_results_hierarchical")

    try:
        if os.path.exists(main_results_dir): 
            shutil.rmtree(main_results_dir)
            logging.info(f"已刪除舊的結果資料夾: {main_results_dir}")
        os.makedirs(main_results_dir, exist_ok=True)
        logging.info(f"階層式辨識結果將儲存於: {main_results_dir}")

        rep_frames_dir = os.path.join(main_results_dir, "representative_frames")
        os.makedirs(rep_frames_dir, exist_ok=True)
        if representative_frames_data:
            for track_id, (frame_bgr, class_name, box, conf, timestamp_str) in representative_frames_data.items():
                annotated_frame = draw_bounding_box_unified(frame_bgr, box, track_id, class_name, conf)
                safe_class_name = "".join(c if c.isalnum() else "_" for c in class_name)
                timestamp_for_file = "00_00_00"
                if timestamp_str and ":" in timestamp_str: 
                    try:
                        parts = list(map(int, timestamp_str.split(':')))
                        seconds_from_str = parts[0]*3600 + parts[1]*60 + parts[2]
                        timestamp_for_file = format_timestamp(seconds_from_str, for_filename=True)
                    except ValueError:
                        logging.warning(f"代表幀的時間戳格式錯誤: {timestamp_str}")

                output_filename = f"rep_track_{track_id:03d}_{safe_class_name}_time_{timestamp_for_file}_conf_{conf:.2f}.jpg"
                output_filepath = os.path.join(rep_frames_dir, output_filename)
                cv2.imwrite(output_filepath, annotated_frame)
            logging.info(f"代表畫面已儲存至: {rep_frames_dir}")

        tracked_details_dir = os.path.join(main_results_dir, "tracked_object_details")
        os.makedirs(tracked_details_dir, exist_ok=True)
        if tracked_data_paths: 
            for track_id, frame_entries in tracked_data_paths.items():
                class_name_for_folder = "unknown_class"
                if track_id in representative_frames_data:
                     class_name_for_folder = representative_frames_data[track_id][1] 
                
                safe_class_name_for_folder = "".join(c if c.isalnum() else "_" for c in class_name_for_folder)
                track_specific_dir = os.path.join(tracked_details_dir, f"track_{track_id:03d}_{safe_class_name_for_folder}")
                os.makedirs(track_specific_dir, exist_ok=True)
                
                for idx, (frame_path, box, conf, timestamp_str) in enumerate(frame_entries): 
                    if os.path.exists(frame_path):
                        original_frame_bgr = cv2.imread(frame_path)
                        if original_frame_bgr is not None:
                            annotated_frame_for_detail = draw_bounding_box_unified(original_frame_bgr, box, track_id, class_name_for_folder, conf)
                            
                            timestamp_for_file_detail = "00_00_00"
                            if timestamp_str and ":" in timestamp_str:
                                try:
                                    parts_detail = list(map(int, timestamp_str.split(':')))
                                    seconds_from_str_detail = parts_detail[0]*3600 + parts_detail[1]*60 + parts_detail[2]
                                    timestamp_for_file_detail = format_timestamp(seconds_from_str_detail, for_filename=True)
                                except ValueError:
                                     logging.warning(f"詳細幀的時間戳格式錯誤: {timestamp_str}")

                            detail_frame_filename = f"frame_{idx:05d}_time_{timestamp_for_file_detail}_conf_{conf:.2f}.jpg"
                            detail_frame_save_path = os.path.join(track_specific_dir, detail_frame_filename)
                            cv2.imwrite(detail_frame_save_path, annotated_frame_for_detail)
                        else:
                            logging.warning(f"無法讀取幀圖片 {frame_path} (Track ID: {track_id})")
                    else:
                        logging.warning(f"幀圖片路徑不存在 {frame_path} (Track ID: {track_id})")
            logging.info(f"詳細追蹤幀已儲存至: {tracked_details_dir}")
        
        st.success(f"階層式辨識結果已儲存至: {os.path.abspath(main_results_dir)}")
        return os.path.abspath(main_results_dir)
    except Exception as e:
        st.error(f"儲存階層式辨識結果失敗: {e}")
        logging.exception(f"儲存階層式辨識結果到 {main_results_dir} 時發生錯誤: {e}")
        return None

# -----------------------------------------------------------------------------
#  Streamlit 介面設定 (其餘部分與 merged_yolo_app_v6_optimized 相同)
#  ... (以下省略與前一版本相同的 Streamlit UI 設定和主邏輯部分)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="YOLO 物件追蹤 App", layout="wide")

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
    "current_session_temp_frames_dir": None, 
    "user_output_dir_path": None, 
    "generated_zip_for_download_path": None, 
}
for key, value in _default_session_values.items():
    if key not in ss: 
        ss[key] = value

main_area_progress_bar_placeholder = st.empty()

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
        ss.video_processed = False    
        ss.tracked_data = None
        ss.representative_frames = None
        ss.confidence_threshold = ss.active_model_config["confidence_threshold"] 
        if is_currently_world_model: 
            ss.current_prompt_world = WORLD_MODEL_CONFIG["default_prompt"]
        
        if ss.current_session_temp_frames_dir and os.path.exists(ss.current_session_temp_frames_dir):
            try:
                shutil.rmtree(ss.current_session_temp_frames_dir)
                logging.info(f"模型類型變更，已清理舊的暫存幀資料夾: {ss.current_session_temp_frames_dir}")
                ss.current_session_temp_frames_dir = None
            except Exception as e:
                logging.error(f"清理舊的暫存幀資料夾 {ss.current_session_temp_frames_dir} 失敗: {e}")
        if ss.generated_zip_for_download_path and os.path.exists(ss.generated_zip_for_download_path):
            try:
                os.remove(ss.generated_zip_for_download_path)
                logging.info(f"模型類型變更，已清理舊的 ZIP 檔案: {ss.generated_zip_for_download_path}")
                ss.generated_zip_for_download_path = None
            except Exception as e:
                logging.error(f"清理舊的 ZIP 檔案 {ss.generated_zip_for_download_path} 失敗: {e}")
        st.rerun() 

    st.caption(f"使用模型: {ss.active_model_config.get('display_name', ss.active_model_config['model_path'])}")

    if ss.loaded_model_object is None:
        model_path_for_loading = ss.active_model_config['model_path']
        with st.spinner(f"正在載入模型 {model_path_for_loading} …"):
            ss.loaded_model_object = load_model_unified(model_path_for_loading)
            if ss.loaded_model_object is None: 
                st.stop() 

    if is_currently_world_model:
        ss.current_prompt_world = st.text_area(
            "輸入要偵測的物件 (以逗號分隔):",
            value=ss.current_prompt_world, 
            height=100,
            key="world_model_prompt_input" 
        )
    else:
        fixed_classes_display = ", ".join(STANDARD_MODEL_CONFIG['target_classes_names'].values())
        st.info(f"固定偵測目標：{fixed_classes_display}")

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
        ss.tracked_data = None
        ss.representative_frames = None
        ss.selected_track_id = None
        ss.video_processed = False
        ss.uploaded_file_name = uploaded_video_file.name
        ss.last_processed_settings = ""
        ss.view_mode = 'all_objects'
        ss.user_output_dir_path = None 
        
        if ss.current_session_temp_frames_dir and os.path.exists(ss.current_session_temp_frames_dir):
            try:
                shutil.rmtree(ss.current_session_temp_frames_dir)
                logging.info(f"上傳新影片，已清理舊的暫存幀資料夾: {ss.current_session_temp_frames_dir}")
                ss.current_session_temp_frames_dir = None
            except Exception as e:
                logging.error(f"清理舊的暫存幀資料夾 {ss.current_session_temp_frames_dir} 失敗: {e}")
        if ss.generated_zip_for_download_path and os.path.exists(ss.generated_zip_for_download_path):
            try:
                os.remove(ss.generated_zip_for_download_path)
                logging.info(f"上傳新影片，已清理舊的 ZIP 檔案: {ss.generated_zip_for_download_path}")
                ss.generated_zip_for_download_path = None
            except Exception as e:
                logging.error(f"清理舊的 ZIP 檔案 {ss.generated_zip_for_download_path} 失敗: {e}")

        if ss.video_path and os.path.exists(ss.video_path):
            try: 
                os.remove(ss.video_path)
                logging.info(f"已清理舊的影片暫存檔案: {ss.video_path}")
            except OSError as e: 
                logging.warning(f"清理舊的影片暫存檔案失敗: {ss.video_path}, 錯誤: {e}")
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video_file.name)[1]) as tmp_vid_file:
                tmp_vid_file.write(uploaded_video_file.getvalue())
                ss.video_path = tmp_vid_file.name
            logging.info(f"新的影片暫存檔已建立: {ss.video_path}")
            st.rerun() 
        except Exception as e:
            st.error(f"建立影片暫存檔失敗: {e}")
            logging.exception(f"建立影片暫存檔時發生錯誤: {e}")
            ss.video_path = None

    app_temp_base = os.path.join(tempfile.gettempdir(), "yolo_streamlit_temp_frames")
    if not os.path.exists(app_temp_base):
        os.makedirs(app_temp_base, exist_ok=True)

    if ss.video_path and ss.loaded_model_object:
        current_processing_config_summary = (
            f"模型: {ss.selected_model_type} | "
            f"提示詞: {ss.current_prompt_world if is_currently_world_model else 'Standard Predefined'} | "
            f"信賴度: {ss.confidence_threshold:.2f}"
        )
        
        button_label = "🚀 開始處理影片"
        if ss.video_processed and ss.last_processed_settings != current_processing_config_summary:
            button_label = "🔄 使用新設定重新處理"
        elif not ss.video_processed:
            button_label = "🚀 開始處理影片"

        if st.button(button_label, use_container_width=True, type="primary", key="process_button_key"):
            yolo_world_custom_classes_list = []
            if is_currently_world_model:
                yolo_world_custom_classes_list = [c.strip() for c in ss.current_prompt_world.split(',') if c.strip()]
                if not yolo_world_custom_classes_list:
                    st.warning("YOLO‑World 需要至少一個有效提示詞！")
                    st.stop() 
            
            if ss.current_session_temp_frames_dir and os.path.exists(ss.current_session_temp_frames_dir):
                try:
                    shutil.rmtree(ss.current_session_temp_frames_dir)
                    logging.info(f"開始新處理，已清理先前的暫存幀資料夾: {ss.current_session_temp_frames_dir}")
                except Exception as e:
                    logging.error(f"清理先前的暫存幀資料夾 {ss.current_session_temp_frames_dir} 失敗: {e}")
            ss.current_session_temp_frames_dir = None 
            ss.user_output_dir_path = None 
            if ss.generated_zip_for_download_path and os.path.exists(ss.generated_zip_for_download_path): 
                try:
                    os.remove(ss.generated_zip_for_download_path)
                    logging.info(f"開始新處理，已清理舊的 ZIP 檔案: {ss.generated_zip_for_download_path}")
                    ss.generated_zip_for_download_path = None
                except Exception as e:
                     logging.error(f"清理舊的 ZIP 檔案 {ss.generated_zip_for_download_path} 失敗: {e}")

            processing_message = f"{ss.selected_model_type} 影片處理中…"
            with st.spinner(processing_message): # 使用 st.spinner
                ss.tracked_data = None
                ss.representative_frames = None
                ss.selected_track_id = None

                tracked_data_result, representative_frames_result, temp_dir_for_this_run = process_video_unified(
                    ss.video_path, 
                    ss.loaded_model_object, 
                    is_currently_world_model, 
                    ss.active_model_config, 
                    yolo_world_custom_classes_list, 
                    ss.confidence_threshold, 
                    main_area_progress_bar_placeholder, # Pass the placeholder from main area
                    app_temp_base 
                )
                ss.current_session_temp_frames_dir = temp_dir_for_this_run 
                
                ss.tracked_data = tracked_data_result
                ss.representative_frames = representative_frames_result
                ss.video_processed = True
                ss.last_processed_settings = current_processing_config_summary
                ss.view_mode = 'all_objects' 
            
            # Spinner 區塊結束後才顯示成功訊息和儲存結果
            st.success("影片處理完成！") 
            
            if ss.representative_frames and ss.uploaded_file_name:
                output_save_path = save_detection_results_hierarchical(
                    BASE_OUTPUT_DIR, 
                    ss.uploaded_file_name, 
                    ss.representative_frames,
                    ss.tracked_data 
                )
                ss.user_output_dir_path = output_save_path 
            
            st.rerun() # 最後才 rerun

    elif not ss.loaded_model_object and ss.active_model_config.get('model_path'):
         st.warning("模型尚未載入或載入失敗。請檢查 Sidebar。")

    if ss.video_processed:
        st.markdown("---")
        st.subheader("👁️ 檢視選項")
        if st.button("所有追蹤物件", 
                      type="primary" if ss.view_mode == 'all_objects' else "secondary", 
                      use_container_width=True, 
                      key="view_all_objects_btn_key"): 
            ss.view_mode = 'all_objects'
            ss.selected_track_id = None
            st.rerun()
        
        if ss.representative_frames:
            sorted_track_ids = sorted(ss.representative_frames.keys())
            if st.button("特定物件所有畫面", 
                          type="primary" if ss.view_mode == 'single_object' else "secondary", 
                          use_container_width=True, 
                          key="view_specific_object_btn_key"): 
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
        elif ss.view_mode == 'single_object': 
            ss.view_mode = 'all_objects'
        
        if ss.user_output_dir_path and os.path.isdir(ss.user_output_dir_path):
            st.markdown("---")
            st.subheader("📥 下載結果")
            
            zip_file_name_base = os.path.basename(ss.user_output_dir_path) 
            download_zip_filename = f"{zip_file_name_base}.zip"
            
            if ss.generated_zip_for_download_path and os.path.exists(ss.generated_zip_for_download_path):
                try:
                    os.remove(ss.generated_zip_for_download_path)
                    logging.info(f"已清理舊的下載 ZIP 檔案: {ss.generated_zip_for_download_path}")
                except Exception as e:
                    logging.error(f"清理舊的 ZIP 檔案 {ss.generated_zip_for_download_path} 失敗: {e}")
                ss.generated_zip_for_download_path = None

            try:
                archive_temp_base = os.path.join(tempfile.gettempdir(), zip_file_name_base)
                
                generated_zip_path = shutil.make_archive(
                    base_name=archive_temp_base, 
                    format='zip',                 
                    root_dir=os.path.dirname(ss.user_output_dir_path), 
                    base_dir=os.path.basename(ss.user_output_dir_path) 
                )
                ss.generated_zip_for_download_path = generated_zip_path 

                if generated_zip_path and os.path.exists(generated_zip_path):
                    with open(generated_zip_path, "rb") as fp:
                        st.download_button(
                            label="下載結果 (ZIP)",
                            data=fp, 
                            file_name=download_zip_filename, 
                            mime="application/zip",
                            key="download_results_zip_button",
                            use_container_width=True
                        )
                else:
                    st.error("無法建立壓縮檔供下載。")
            except Exception as e:
                st.error(f"建立壓縮檔時發生錯誤: {e}")
                logging.exception(f"建立壓縮檔時發生錯誤: {e}")


# -----------------------------------------------------------------------------
#  Main Area 內容 (影片預覽 / 結果顯示)
# -----------------------------------------------------------------------------
st.title("🎬 YOLO 通用物件偵測與追蹤")

if not ss.loaded_model_object and ss.active_model_config.get('model_path'):
    st.warning("模型正在載入或載入失敗。請檢查 Sidebar。")
elif not ss.video_path:
    st.info("👋 歡迎！請在左側控制面板選擇模型類型、設定偵測目標並上傳影片檔案。")
    st.markdown(f"""
        - **Standard YOLO**: 偵測預定義物件 (例如: {", ".join(STANDARD_MODEL_CONFIG['target_classes_names'].values())})。
        - **YOLO-World**: 輸入您想偵測的任意物件名稱 (例如: `a red apple, a blue car`)。
    """)
else: 
    video_col, empty_col = st.columns([2, 1]) 
    with video_col:
        st.subheader("🎞️ 影片預覽")
        st.video(ss.video_path)
    
    if not ss.video_processed and ss.video_path:
        _current_settings_summary_for_main = (
            f"模型: {ss.selected_model_type}, "
            f"提示詞: {ss.current_prompt_world if is_currently_world_model else 'Standard Predefined'}, " 
            f"信賴度: {ss.confidence_threshold:.2f}"
        )
        _is_first_processing = not ss.video_processed
        _config_changed_after_processing = (ss.video_processed and 
                                            ss.last_processed_settings != _current_settings_summary_for_main)

        if _config_changed_after_processing:
             st.info("偵測設定已變更。請點擊左側 Sidebar 的「🔄 使用新設定重新處理」按鈕。")
        elif _is_first_processing:
            st.info("影片已上傳。請點擊左側 Sidebar 的「🚀 開始處理影片」按鈕。")


if ss.video_processed:
    processed_model_type_disp = "N/A"
    processed_prompt_text_disp = "N/A"
    processed_conf_text_disp = "N/A"

    if ss.last_processed_settings: 
        parts = ss.last_processed_settings.split(" | ")
        if len(parts) == 3: 
            try:
                model_part_val = parts[0].split(": ", 1)
                if len(model_part_val) > 1: processed_model_type_disp = model_part_val[1]
                
                prompt_part_val = parts[1].split(": ", 1)
                if len(prompt_part_val) > 1: processed_prompt_text_disp = prompt_part_val[1]

                conf_part_val = parts[2].split(": ", 1)
                if len(conf_part_val) > 1: processed_conf_text_disp = conf_part_val[1]
            except IndexError: 
                logging.warning(f"解析 last_processed_settings 時發生索引錯誤: {ss.last_processed_settings}")
        else:
            logging.warning(f"last_processed_settings 格式不符，無法完整解析: {ss.last_processed_settings}")
            if len(parts) > 0: processed_model_type_disp = parts[0] 

    if ss.user_output_dir_path: 
        st.info(f"💡 階層式辨識結果已儲存至: {ss.user_output_dir_path}")


    if not ss.representative_frames: 
        st.info(f"影片處理完成。模型 {processed_model_type_disp} 未偵測到符合 '{processed_prompt_text_disp}' 且信賴度 ≥ {processed_conf_text_disp} 的物件。")
    else:
        if ss.view_mode == 'all_objects':
            st.header("📊 所有追蹤物件 (代表畫面)")
            st.write(f"模型: {processed_model_type_disp} | 目標: '{processed_prompt_text_disp}' | 最低信賴度: {processed_conf_text_disp}")
            st.write(f"總共偵測並追蹤到 {len(ss.representative_frames)} 個獨立物件。")

            num_cols_for_all_objects_view = st.slider("每行顯示物件數 (網格視圖):", 2, 8, 4, key="all_objects_grid_cols_slider")
            grid_cols = st.columns(num_cols_for_all_objects_view)
            sorted_representative_track_ids = sorted(list(ss.representative_frames.keys())) 

            for i, track_id_val in enumerate(sorted_representative_track_ids):
                with grid_cols[i % num_cols_for_all_objects_view]: 
                    frame_bgr_rep, class_name_rep, box_rep, conf_rep, timestamp_rep_str = ss.representative_frames[track_id_val]
                    img_with_box_rep = draw_bounding_box_unified(frame_bgr_rep, box_rep, track_id_val, class_name_rep, conf_rep)
                    img_rgb_rep = cv2.cvtColor(img_with_box_rep, cv2.COLOR_BGR2RGB) 
                    st.image(img_rgb_rep, caption=f"ID: {track_id_val} ({class_name_rep}, {conf_rep:.2f}) - {timestamp_rep_str}", use_container_width=True)
                    if st.button(f"檢視 ID {track_id_val} 所有畫面", key=f"view_all_frames_for_id_{track_id_val}_button", use_container_width=True):
                        ss.selected_track_id = track_id_val
                        ss.view_mode = 'single_object' 
                        st.rerun()
            st.markdown("---") 

        elif ss.view_mode == 'single_object' and ss.selected_track_id is not None:
            current_selected_id = ss.selected_track_id
            if current_selected_id in ss.tracked_data and current_selected_id in ss.representative_frames:
                frames_data_for_id = ss.tracked_data[current_selected_id] 
                _, class_name_for_header, _, representative_conf_for_header, rep_timestamp_str = ss.representative_frames[current_selected_id] 
                
                st.header(f"🖼️ 物件 ID: {current_selected_id} ({class_name_for_header}) 的所有畫面")
                st.write(f"模型: {processed_model_type_disp} | 代表性信賴度: {representative_conf_for_header:.2f} (於 {rep_timestamp_str})")
                st.write(f"此物件出現的總幀數: {len(frames_data_for_id)}。")
                
                slider_max_val = max(10, len(frames_data_for_id)) 
                slider_default_val = min(50, len(frames_data_for_id)) 
                if slider_default_val == 0 and len(frames_data_for_id) > 0: 
                    slider_default_val = min(10, len(frames_data_for_id))
                elif len(frames_data_for_id) == 0: 
                    slider_default_val = 10 

                num_frames_to_show_slider = st.slider(
                    "最大顯示幀數:", 
                    min_value=10, 
                    max_value=slider_max_val, 
                    value=slider_default_val, 
                    step=10, 
                    key=f"max_frames_slider_for_id_{current_selected_id}",
                    disabled=(len(frames_data_for_id) <=10) 
                ) 
                
                data_for_detailed_display = frames_data_for_id[:num_frames_to_show_slider]
                
                if len(frames_data_for_id) > num_frames_to_show_slider:
                     st.warning(f"目前顯示前 {num_frames_to_show_slider} 幀 (共 {len(frames_data_for_id)} 幀)。可調整上方滑桿。")

                cols_per_row_for_detailed_view = st.number_input(
                    "每行顯示幀數 (詳細視圖):", 
                    min_value=2, max_value=10, value=4, 
                    key=f"cols_per_row_input_for_id_{current_selected_id}"
                )
                detailed_view_columns = st.columns(cols_per_row_for_detailed_view)

                for idx, (frame_path_detail, box_detail, conf_detail, timestamp_detail_str) in enumerate(data_for_detailed_display): 
                    with detailed_view_columns[idx % cols_per_row_for_detailed_view]: 
                        if os.path.exists(frame_path_detail):
                            frame_bgr_detail = cv2.imread(frame_path_detail)
                            if frame_bgr_detail is not None:
                                img_to_display_detail = frame_bgr_detail 
                                x1_detail, y1_detail, x2_detail, y2_detail = map(int, box_detail)
                                cv2.rectangle(img_to_display_detail, (x1_detail, y1_detail), (x2_detail, y2_detail), BOX_COLOR, BOX_THICKNESS)
                                img_rgb_detail = cv2.cvtColor(img_to_display_detail, cv2.COLOR_BGR2RGB)
                                st.image(img_rgb_detail, caption=f"幀 {idx+1} ({timestamp_detail_str}) 信賴度: {conf_detail:.2f}", use_container_width=True)
                            else:
                                st.warning(f"無法讀取幀圖片: {frame_path_detail}")
                                logging.error(f"無法讀取幀圖片: {frame_path_detail}")
                        else:
                            st.warning(f"幀圖片遺失: {frame_path_detail}")
                            logging.error(f"幀圖片遺失: {frame_path_detail}")

            else: 
                st.warning(f"找不到 Track ID {current_selected_id} 的資料。請從左側選單選擇有效的物件 ID 或返回總覽。")
                ss.view_mode = 'all_objects' 
                ss.selected_track_id = None
                st.rerun()

# -----------------------------------------------------------------------------
#  頁腳
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"由 Ultralytics YOLO 和 Streamlit 驅動")
