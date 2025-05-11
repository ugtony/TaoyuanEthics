# -*- coding: utf-8 -*-
"""
YOLO (Standard / YOLO‑World) 物件偵測 + 追蹤 — Streamlit App
==========================================================
此版本基於使用者提供的可運作版本 (merged_yolo_app_v4_fixed_syntax) 進行調整，
確保使用 st.spinner() 並在 spinner 區塊外呼叫 st.rerun()。
此版本調整了影片預覽大小並擴充了 Standard YOLO 的預設偵測類別。
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
    "default_prompt": "person, car, bicycle, traffic light, backpack", # 擴充預設提示詞
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

# -----------------------------------------------------------------------------
#  Helper — 載入模型
# -----------------------------------------------------------------------------
@st.cache_resource # 使用 Streamlit 的快取機制來加速模型載入
def load_model_unified(model_path):
    """載入 YOLO 模型 (通用於 Standard YOLO 和 YOLO-World)。"""
    try:
        logging.info(f"正在載入模型: {model_path}")
        model = YOLO(model_path) # 使用 YOLO() 載入模型
        logging.info(f"模型 {model_path} 載入成功。")
        return model
    except Exception as e:
        st.error(f"載入模型 '{model_path}' 失敗: {e}")
        logging.exception(f"載入模型 '{model_path}' 時發生錯誤: {e}") # 記錄包含堆疊追蹤的錯誤
        return None

# -----------------------------------------------------------------------------
#  Helper — 繪製邊界框
# -----------------------------------------------------------------------------
def draw_bounding_box_unified(frame, box, track_id, class_name, conf):
    """在指定的幀上繪製單一物件的邊界框和標籤 (包含信賴度)。"""
    img = frame.copy() # 在副本上操作，避免修改原始幀
    x1, y1, x2, y2 = map(int, box) # 將座標轉換為整數
    label = f"ID:{track_id} {class_name} {conf:.2f}" # 格式化標籤內容

    # 繪製邊界框
    cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # 計算文字大小以決定標籤背景尺寸
    (w, h), _ = cv2.getTextSize(label, TEXT_FONT, TEXT_FONT_SCALE, TEXT_THICKNESS)
    # 計算標籤位置，使其盡量不超出圖片頂部
    label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
    label_x1 = max(0, x1) # 確保標籤背景不超出圖片左邊界
    label_x2 = label_x1 + w
    # 繪製填滿的標籤背景矩形
    cv2.rectangle(img, (label_x1, label_y - h - 5), (label_x2, label_y), TEXT_BG_COLOR, -1)
    # 繪製標籤文字
    cv2.putText(img, label, (label_x1, label_y - 3), TEXT_FONT, TEXT_FONT_SCALE, TEXT_COLOR_ON_BG, TEXT_THICKNESS, cv2.LINE_AA)

    return img

# -----------------------------------------------------------------------------
#  Helper — 處理影片（偵測 + 追蹤）
# -----------------------------------------------------------------------------
def process_video_unified(video_path, model, is_world_model, current_model_settings,
                          yolo_world_custom_classes, confidence_thresh, progress_bar_element):
    """
    統一的影片處理函式，適用於 Standard YOLO 和 YOLO-World。
    Args:
        video_path (str): 影片檔案的路徑。
        model: 已載入的 YOLO 模型物件。
        is_world_model (bool): 指示是否為 YOLO-World 模型。
        current_model_settings (dict): 當前選定模型的設定檔 (主要用於 Standard YOLO)。
        yolo_world_custom_classes (list): YOLO-World 使用的自訂類別名稱列表。
        confidence_thresh (float): 信賴度閾值。
        progress_bar_element: Streamlit 的 st.empty() 元件，用於顯示進度條。
    Returns:
        tuple: (tracked_object_frames, representative_frames)
    """
    tracked_object_frames = defaultdict(list) # 儲存每個追蹤ID的所有幀
    representative_frames = {} # 儲存每個追蹤ID的代表幀

    # 防禦性檢查 progress_bar_element
    if not hasattr(progress_bar_element, "progress"):
        class DummyProgressBar: # 如果未傳入有效的進度條元件，則使用虛擬元件
            def progress(self, *_args, **_kw): pass
            def empty(self): pass
        progress_bar_element = DummyProgressBar()
        logging.warning("process_video_unified: 未傳入有效的 progress_bar_element，使用虛擬元件。")


    try:
        active_classes_for_tracking = [] # 用於 model.track 的 classes 參數 (Standard YOLO)
        class_name_source = {} # 用於從 cls_id 取得類別名稱的來源

        if is_world_model:
            if not yolo_world_custom_classes:
                st.warning("YOLO-World 模型需要至少一個偵測目標。")
                logging.warning("YOLO-World: 嘗試處理但未提供偵測目標。")
                return {}, {}
            model.set_classes(yolo_world_custom_classes) # 為 YOLO-World 設定偵測類別
            class_name_source = yolo_world_custom_classes # 類別名稱直接來自此列表 (索引對應 cls_id)
            logging.info(f"YOLO-World: 設定偵測目標: {', '.join(yolo_world_custom_classes)}")
        else: # Standard YOLO
            active_classes_for_tracking = current_model_settings["target_classes_ids"]
            class_name_source = current_model_settings["target_classes_names"] # 類別名稱來自此字典 (cls_id 對應 key)
            logging.info(f"Standard YOLO: 使用固定類別 IDs: {active_classes_for_tracking}")

        cap = cv2.VideoCapture(video_path) # 開啟影片檔案
        if not cap.isOpened():
            st.error("無法開啟影片檔案。")
            logging.error(f"無法開啟影片檔案: {video_path}")
            return {}, {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 獲取總幀數
        # 使用傳入的元件來顯示進度條
        pb_instance = progress_bar_element.progress(0, text="正在處理影片…") 
        start_time = time.time() # 開始計時
        frame_idx = 0 # 當前幀計數

        while True:
            ok, frame = cap.read() # 讀取一幀
            if not ok: # 如果讀取失敗或影片結束，則跳出迴圈
                break
            
            frame_idx += 1
            current_fps = frame_idx / (time.time() - start_time + 1e-6) # 計算FPS (避免除以零)

            # 準備 model.track() 的參數
            track_params = dict(
                source=frame, 
                tracker=TRACKER_CONFIG,
                conf=confidence_thresh, 
                persist=True,       # 保持追蹤ID的連續性
                verbose=False       # 減少控制台輸出
            )
            if not is_world_model: # Standard YOLO 需要明確指定 classes 參數
                track_params["classes"] = active_classes_for_tracking
            
            results = model.track(**track_params) # 執行物件追蹤

            # 處理追蹤結果
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_coords = results[0].boxes.xyxy.cpu().numpy()
                track_ids_list = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids_from_model = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences_list = results[0].boxes.conf.cpu().numpy()

                for current_box, track_id, cls_id, conf_score in zip(boxes_coords, track_ids_list, class_ids_from_model, confidences_list):
                    if conf_score < confidence_thresh: # 再次確認信賴度 (雖然 track 內部已過濾)
                        continue
                    
                    object_class_name = f"class_{cls_id}" # 預設類別名稱
                    if is_world_model:
                        # 對於 YOLO-World, cls_id 是 class_name_source (即 yolo_world_custom_classes) 的索引
                        if 0 <= cls_id < len(class_name_source):
                            object_class_name = class_name_source[cls_id]
                        else:
                             logging.warning(f"YOLO-World: 偵測到預期外的類別 ID {cls_id} (提示詞長度: {len(class_name_source)})")
                    else:
                        # 對於 Standard YOLO, cls_id 是 COCO ID, class_name_source 是 target_classes_names 字典
                        object_class_name = class_name_source.get(cls_id, f"class_{cls_id}")
                    
                    tracked_object_frames[track_id].append((frame.copy(), tuple(current_box), conf_score))
                    if track_id not in representative_frames:
                        representative_frames[track_id] = (frame.copy(), object_class_name, tuple(current_box), conf_score)

            # 更新進度條
            progress_percent = int(frame_idx / total_frames * 100) if total_frames > 0 else 0
            pb_instance.progress(progress_percent, text=f"處理中… {progress_percent}% (FPS: {current_fps:.2f})")

        if cap: cap.release() # 釋放影片資源
        pb_instance.progress(100, text="影片處理完成！")
        time.sleep(1) # 讓使用者能看到完成訊息
        progress_bar_element.empty() # 清空進度條元件
        logging.info("影片處理完成。")
        return tracked_object_frames, representative_frames

    except Exception as e:
        st.error(f"處理影片時發生錯誤: {e}")
        logging.exception(f"處理影片時發生未預期錯誤: {e}") # 記錄完整堆疊追蹤
        if hasattr(progress_bar_element, "empty"): # 確保在錯誤時也清空進度條
            progress_bar_element.empty()
        return {}, {}

# -----------------------------------------------------------------------------
#  Streamlit 介面設定
# -----------------------------------------------------------------------------
st.set_page_config(page_title="YOLO 物件追蹤 App", layout="wide")

# ── Session State 預設值 (使用 ss 作為 st.session_state 的別名) ──
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
    "last_processed_settings": "", # 儲存上次處理的設定摘要，用於比較是否有變更
    "view_mode": "all_objects",   # 'all_objects' 或 'single_object'
    "confidence_threshold": STANDARD_MODEL_CONFIG["confidence_threshold"],
}
for key, value in _default_session_values.items():
    if key not in ss: 
        ss[key] = value

# -----------------------------------------------------------------------------
#  Main 區域 — 先定義進度條的 placeholder
# -----------------------------------------------------------------------------
# 將進度條 placeholder 移到主腳本流程中定義，確保其在 sidebar 邏輯執行前已存在
main_area_progress_bar_placeholder = st.empty()

# -----------------------------------------------------------------------------
#  Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 控制面板")

    # 1. 模型類型選擇
    previous_selected_model_type = ss.selected_model_type
    ss.selected_model_type = st.radio(
        "選擇模型類型:", 
        MODEL_TYPES, 
        key="model_type_radio_selector", # 使用唯一的 key
        horizontal=True
    )
    is_currently_world_model = (ss.selected_model_type == "YOLO-World")

    # 如果模型類型改變，則重設相關狀態
    if previous_selected_model_type != ss.selected_model_type:
        ss.active_model_config = WORLD_MODEL_CONFIG if is_currently_world_model else STANDARD_MODEL_CONFIG
        ss.loaded_model_object = None # 標記為需要重新載入模型
        ss.video_processed = False    # 重設影片處理狀態
        ss.tracked_data = None
        ss.representative_frames = None
        ss.confidence_threshold = ss.active_model_config["confidence_threshold"] # 設定為新模型的預設信賴度
        if is_currently_world_model: # 如果切換到 YOLO-World，重設其提示詞
            ss.current_prompt_world = WORLD_MODEL_CONFIG["default_prompt"]
        st.rerun() # 重新執行腳本以套用變更

    st.caption(f"使用模型: {ss.active_model_config.get('display_name', ss.active_model_config['model_path'])}")

    # 2. 載入模型 (如果尚未載入或模型已變更)
    if ss.loaded_model_object is None:
        model_path_for_loading = ss.active_model_config['model_path']
        # 使用 st.spinner 顯示載入中的訊息
        with st.spinner(f"正在載入模型 {model_path_for_loading} …"):
            ss.loaded_model_object = load_model_unified(model_path_for_loading)
            if ss.loaded_model_object is None: # 如果載入失敗
                # load_model_unified 內部已顯示 st.error
                st.stop() # 停止腳本執行
            # 載入成功後不需要立即 st.rerun()，讓腳本自然流動或由其他互動觸發 rerun

    # 3. 類別輸入 (YOLO-World) 或固定類別顯示 (Standard YOLO)
    if is_currently_world_model:
        ss.current_prompt_world = st.text_area(
            "輸入要偵測的物件 (以逗號分隔):",
            value=ss.current_prompt_world, 
            height=100,
            key="world_model_prompt_input" # 使用唯一的 key
        )
    else:
        fixed_classes_display = ", ".join(STANDARD_MODEL_CONFIG['target_classes_names'].values())
        st.info(f"固定偵測目標：{fixed_classes_display}")

    # 4. 信賴度閾值滑桿
    ss.confidence_threshold = st.slider(
        "信賴度閾值:", 
        0.05, 0.95, 
        ss.confidence_threshold, 
        0.05,
        key="confidence_level_slider" # 使用唯一的 key
    )

    # 5. 檔案上傳
    uploaded_video_file = st.file_uploader(
        "選擇影片檔案", 
        ["mp4", "avi", "mov", "mkv"],
        key="video_file_uploader_widget" # 使用唯一的 key
    )
    if uploaded_video_file is not None and ss.uploaded_file_name != uploaded_video_file.name:
        ss.tracked_data = None
        ss.representative_frames = None
        ss.selected_track_id = None
        ss.video_processed = False
        ss.uploaded_file_name = uploaded_video_file.name
        ss.last_processed_settings = ""
        ss.view_mode = 'all_objects'
        
        # 清理舊的暫存檔案
        if ss.video_path and os.path.exists(ss.video_path):
            try: 
                os.remove(ss.video_path)
                logging.info(f"已清理舊的暫存檔案: {ss.video_path}")
            except OSError as e: 
                logging.warning(f"清理舊的暫存檔案失敗: {ss.video_path}, 錯誤: {e}")
                pass # 忽略清理錯誤，但記錄下來
        
        # 建立新的暫存檔案
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video_file.name)[1]) as tmp_vid_file:
                tmp_vid_file.write(uploaded_video_file.getvalue())
                ss.video_path = tmp_vid_file.name
            logging.info(f"新的影片暫存檔已建立: {ss.video_path}")
            st.rerun() # 上傳新檔案後重新執行以更新影片預覽
        except Exception as e:
            st.error(f"建立影片暫存檔失敗: {e}")
            logging.exception(f"建立影片暫存檔時發生錯誤: {e}")
            ss.video_path = None


    # 6. ── 開始處理影片 按鈕 ──
    if ss.video_path and ss.loaded_model_object:
        # 產生目前設定的摘要字串，用於比較是否有變更
        current_processing_config_summary = (
            f"模型: {ss.selected_model_type} | "
            f"提示詞: {ss.current_prompt_world if is_currently_world_model else 'Standard Predefined'} | "
            f"信賴度: {ss.confidence_threshold:.2f}"
        )
        
        button_label = "🚀 開始處理影片"
        # 檢查設定是否自上次處理後已變更，或影片尚未處理
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
                    st.stop() # 如果提示詞為空，則停止執行
            
            # 使用 st.spinner 包住長時間執行的影片處理任務
            processing_message = f"{ss.selected_model_type} 影片處理中 ({'目標: ' + ss.current_prompt_world if is_currently_world_model else '固定目標'}, 信賴度: {ss.confidence_threshold:.2f})…"
            with st.spinner(processing_message):
                # 重設結果
                ss.tracked_data = None
                ss.representative_frames = None
                ss.selected_track_id = None

                # 呼叫影片處理函式，傳入主區域的進度條 placeholder
                tracked_data_result, representative_frames_result = process_video_unified(
                    ss.video_path, 
                    ss.loaded_model_object, 
                    is_currently_world_model, 
                    ss.active_model_config, # Standard YOLO 會用到裡面的 IDs 和 Names
                    yolo_world_custom_classes_list, # YOLO-World 會用到這個列表
                    ss.confidence_threshold, 
                    main_area_progress_bar_placeholder # 傳入在主區域定義的 placeholder
                )
                
                # 更新 session state
                ss.tracked_data = tracked_data_result
                ss.representative_frames = representative_frames_result
                ss.video_processed = True
                ss.last_processed_settings = current_processing_config_summary
                ss.view_mode = 'all_objects' # 處理完畢後預設顯示所有物件
            
            # Spinner 結束後顯示成功訊息，然後再 rerun
            st.success("影片處理完成！") 
            st.rerun() # 重新執行以刷新並顯示結果

    elif not ss.loaded_model_object and ss.active_model_config.get('model_path'):
         st.warning("模型尚未載入或載入失敗。請檢查 Sidebar。")

    # 7. 檢視選項 (影片處理完成後顯示)
    if ss.video_processed:
        st.markdown("---")
        st.subheader("👁️ 檢視選項")
        # "所有追蹤物件" 按鈕
        if st.button("所有追蹤物件", 
                      type="primary" if ss.view_mode == 'all_objects' else "secondary", 
                      use_container_width=True, 
                      key="view_all_objects_btn_key"): # 使用唯一的 key
            ss.view_mode = 'all_objects'
            ss.selected_track_id = None
            st.rerun()
        
        # 如果有偵測到物件，才顯示 "特定物件" 相關選項
        if ss.representative_frames:
            sorted_track_ids = sorted(ss.representative_frames.keys())
            # "特定物件所有畫面" 按鈕
            if st.button("特定物件所有畫面", 
                          type="primary" if ss.view_mode == 'single_object' else "secondary", 
                          use_container_width=True, 
                          key="view_specific_object_btn_key"): # 使用唯一的 key
                ss.view_mode = 'single_object'
                ss.selected_track_id = sorted_track_ids[0] if sorted_track_ids else None # 預設選第一個
                st.rerun()
            
            # 如果在 "特定物件" 模式且有物件可選，則顯示下拉選單
            if ss.view_mode == 'single_object' and sorted_track_ids:
                # format_func 用於在下拉選單中顯示更友好的名稱
                selected_id_choice = st.selectbox(
                    "選擇物件 ID:", 
                    sorted_track_ids, 
                    index=sorted_track_ids.index(ss.selected_track_id) if ss.selected_track_id in sorted_track_ids else 0,
                    format_func=lambda tid_key: f"ID:{tid_key} ({ss.representative_frames[tid_key][1]})", # [1] 是 class_name
                    key="select_specific_object_id_selectbox" # 使用唯一的 key
                )
                if selected_id_choice != ss.selected_track_id:
                    ss.selected_track_id = selected_id_choice
                    st.rerun()
        elif ss.view_mode == 'single_object': # 如果在單一物件模式但沒有物件，則切回
            ss.view_mode = 'all_objects'
            # st.rerun() # 這裡可以考慮是否需要 rerun，或者讓下一次互動觸發

# -----------------------------------------------------------------------------
#  Main Area 內容 (影片預覽 / 結果顯示)
# -----------------------------------------------------------------------------
st.title("🎬 YOLO 通用物件偵測與追蹤")

# 根據 Session State 決定主區域顯示內容
if not ss.loaded_model_object and ss.active_model_config.get('model_path'):
    # 只有在模型路徑已設定但模型物件未載入時（通常表示載入失敗）才顯示此警告
    st.warning("模型正在載入或載入失敗。請檢查 Sidebar。")
elif not ss.video_path:
    st.info("👋 歡迎！請在左側控制面板選擇模型類型、設定偵測目標並上傳影片檔案。")
    st.markdown(f"""
        - **Standard YOLO**: 偵測預定義物件 (例如: {", ".join(STANDARD_MODEL_CONFIG['target_classes_names'].values())})。
        - **YOLO-World**: 輸入您想偵測的任意物件名稱 (例如: `a red apple, a blue car`)。
    """)
else: # video_path 存在，顯示影片預覽
    # 使用 st.columns 來限制影片播放器的寬度
    video_col, empty_col = st.columns([2, 1]) # 影片佔 2/3，右側留空 1/3
    with video_col:
        st.subheader("🎞️ 影片預覽")
        st.video(ss.video_path)
    
    # main_area_progress_bar_placeholder 已在主流程頂部定義
    # 如果影片已上傳但未處理，提示使用者
    if not ss.video_processed and ss.video_path:
        # 重新計算 current_processing_settings_summary 以便比較
        _current_settings_summary_for_main = (
            f"模型: {ss.selected_model_type}, "
            f"提示詞: {ss.current_prompt_world if is_currently_world_model else 'Standard Predefined'}, " # is_currently_world_model 來自 sidebar 範圍
            f"信賴度: {ss.confidence_threshold:.2f}"
        )
        # _settings_changed_for_main = (ss.video_processed and # 這裡應該是 !ss.video_processed 或者 settings_changed
        #                              ss.last_processed_settings != _current_settings_summary_for_main and
        #                              ss.video_path)
        
        # 修正: settings_changed 應該是針對"已處理過但設定改變"的情況
        # 如果尚未處理，則顯示"開始處理"
        # 如果已處理但設定改變，則顯示"重新處理"
        _is_first_processing = not ss.video_processed
        _config_changed_after_processing = (ss.video_processed and 
                                            ss.last_processed_settings != _current_settings_summary_for_main)

        if _config_changed_after_processing:
             st.info("偵測設定已變更。請點擊左側 Sidebar 的「🔄 使用新設定重新處理」按鈕。")
        elif _is_first_processing:
            st.info("影片已上傳。請點擊左側 Sidebar 的「🚀 開始處理影片」按鈕。")


# --- 結果顯示 (Main Area) ---
if ss.video_processed:
    # 從 last_processed_settings 解析上次處理的資訊以供顯示
    processed_model_type_disp = "N/A"
    processed_prompt_text_disp = "N/A"
    processed_conf_text_disp = "N/A"

    if ss.last_processed_settings: # 確保字串非空
        parts = ss.last_processed_settings.split(" | ")
        if len(parts) == 3: # 預期格式 "模型: X | 提示詞: Y | 信賴度: Z"
            try:
                model_part_val = parts[0].split(": ", 1)
                if len(model_part_val) > 1: processed_model_type_disp = model_part_val[1]
                
                prompt_part_val = parts[1].split(": ", 1)
                if len(prompt_part_val) > 1: processed_prompt_text_disp = prompt_part_val[1]

                conf_part_val = parts[2].split(": ", 1)
                if len(conf_part_val) > 1: processed_conf_text_disp = conf_part_val[1]
            except IndexError: # 防禦性處理，如果分割不如預期
                logging.warning(f"解析 last_processed_settings 時發生索引錯誤: {ss.last_processed_settings}")
                # 保留預設的 "N/A"
        else:
            logging.warning(f"last_processed_settings 格式不符，無法完整解析: {ss.last_processed_settings}")
            # 嘗試部分解析或保留預設
            if len(parts) > 0: processed_model_type_disp = parts[0] # 至少顯示部分資訊


    if not ss.representative_frames: # 如果沒有偵測到任何代表幀
        st.info(f"影片處理完成。模型 {processed_model_type_disp} 未偵測到符合 '{processed_prompt_text_disp}' 且信賴度 ≥ {processed_conf_text_disp} 的物件。")
    else:
        # 顯示 "所有追蹤物件" 的網格視圖
        if ss.view_mode == 'all_objects':
            st.header("📊 所有追蹤物件 (代表畫面)")
            st.write(f"模型: {processed_model_type_disp} | 目標: '{processed_prompt_text_disp}' | 最低信賴度: {processed_conf_text_disp}")
            st.write(f"總共偵測並追蹤到 {len(ss.representative_frames)} 個獨立物件。")

            num_cols_for_all_objects_view = st.slider("每行顯示物件數 (網格視圖):", 2, 8, 4, key="all_objects_grid_cols_slider")
            grid_cols = st.columns(num_cols_for_all_objects_view)
            sorted_representative_track_ids = sorted(list(ss.representative_frames.keys())) 

            for i, track_id_val in enumerate(sorted_representative_track_ids):
                with grid_cols[i % num_cols_for_all_objects_view]: 
                    # 從 representative_frames 獲取資料
                    frame_bgr_rep, class_name_rep, box_rep, conf_rep = ss.representative_frames[track_id_val]
                    # 繪製帶有完整標籤的邊界框
                    img_with_box_rep = draw_bounding_box_unified(frame_bgr_rep, box_rep, track_id_val, class_name_rep, conf_rep)
                    img_rgb_rep = cv2.cvtColor(img_with_box_rep, cv2.COLOR_BGR2RGB) 
                    st.image(img_rgb_rep, caption=f"ID: {track_id_val} ({class_name_rep}, {conf_rep:.2f})", use_container_width=True)
                    # "檢視所有畫面" 按鈕
                    if st.button(f"檢視 ID {track_id_val} 所有畫面", key=f"view_all_frames_for_id_{track_id_val}_button", use_container_width=True):
                        ss.selected_track_id = track_id_val
                        ss.view_mode = 'single_object' 
                        st.rerun()
            st.markdown("---") 

        # 顯示 "特定物件 ID" 的詳細幀視圖
        elif ss.view_mode == 'single_object' and ss.selected_track_id is not None:
            current_selected_id = ss.selected_track_id
            # 確保選定的 ID 的資料存在
            if current_selected_id in ss.tracked_data and current_selected_id in ss.representative_frames:
                frames_to_display_for_id = ss.tracked_data[current_selected_id]
                # 從 representative_frames 獲取類別名稱和代表性信賴度以供顯示
                _, class_name_for_header, _, representative_conf_for_header = ss.representative_frames[current_selected_id] 
                
                st.header(f"🖼️ 物件 ID: {current_selected_id} ({class_name_for_header}) 的所有畫面")
                st.write(f"模型: {processed_model_type_disp} | 代表性信賴度: {representative_conf_for_header:.2f}")
                st.write(f"此物件出現的總幀數: {len(frames_to_display_for_id)}。")
                
                # 設定滑桿的最大值和預設值
                slider_max_val = max(10, len(frames_to_display_for_id)) 
                slider_default_val = min(50, len(frames_to_display_for_id)) 
                if slider_default_val == 0 and len(frames_to_display_for_id) > 0: 
                    slider_default_val = min(10, len(frames_to_display_for_id))
                elif len(frames_to_display_for_id) == 0: # 理論上不應發生，因為已檢查 frames_to_display_for_id
                    slider_default_val = 10 

                num_frames_to_show_slider = st.slider(
                    "最大顯示幀數:", 
                    min_value=10, 
                    max_value=slider_max_val, 
                    value=slider_default_val, 
                    step=10, 
                    key=f"max_frames_slider_for_id_{current_selected_id}",
                    disabled=(len(frames_to_display_for_id) <=10) 
                ) 
                
                data_for_detailed_display = frames_to_display_for_id[:num_frames_to_show_slider]
                
                if len(frames_to_display_for_id) > num_frames_to_show_slider:
                     st.warning(f"目前顯示前 {num_frames_to_show_slider} 幀 (共 {len(frames_to_display_for_id)} 幀)。可調整上方滑桿。")

                cols_per_row_for_detailed_view = st.number_input(
                    "每行顯示幀數 (詳細視圖):", 
                    min_value=2, max_value=10, value=4, # 預設改為4欄
                    key=f"cols_per_row_input_for_id_{current_selected_id}"
                )
                detailed_view_columns = st.columns(cols_per_row_for_detailed_view)

                for idx, (frame_bgr_detail, box_detail, conf_detail) in enumerate(data_for_detailed_display):
                    with detailed_view_columns[idx % cols_per_row_for_detailed_view]: 
                        img_to_display_detail = frame_bgr_detail.copy()
                        # 在詳細視圖中，只繪製簡單的方框，不加完整標籤，以避免畫面混亂
                        x1_detail, y1_detail, x2_detail, y2_detail = map(int, box_detail)
                        cv2.rectangle(img_to_display_detail, (x1_detail, y1_detail), (x2_detail, y2_detail), BOX_COLOR, BOX_THICKNESS)
                        img_rgb_detail = cv2.cvtColor(img_to_display_detail, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb_detail, caption=f"幀 {idx+1} (信賴度: {conf_detail:.2f})", use_container_width=True)
            else: # 如果找不到選定 ID 的資料
                st.warning(f"找不到 Track ID {current_selected_id} 的資料。請從左側選單選擇有效的物件 ID 或返回總覽。")
                ss.view_mode = 'all_objects' # 還原到總覽模式以避免狀態損壞
                ss.selected_track_id = None
                st.rerun()

# -----------------------------------------------------------------------------
#  頁腳
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"由 Ultralytics YOLO 和 Streamlit 驅動")
