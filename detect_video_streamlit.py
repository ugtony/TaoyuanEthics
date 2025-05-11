import cv2
import tempfile
import os

import torch
torch.classes.__path__ = [] # 把 torch.classes 裡面的 __path__ 清掉，Streamlit 就無路可走、也不會拋例外

import streamlit as st
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time # 引入 time 模組

# --- 設定 ---
MODEL_PATH = 'yolov8n.pt'  # 您可以選擇不同的 YOLOv8 模型
TARGET_CLASSES = [0, 2] # COCO 資料集中: 0: person, 2: car
TARGET_CLASS_NAMES = {0: 'Person', 2: 'Car'} # 方便顯示類別名稱
CONFIDENCE_THRESHOLD = 0.3
TRACKER_CONFIG = 'bytetrack.yaml'
BOX_COLOR = (0, 255, 0) # BGR 格式的綠色
BOX_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Helper 函式 ---

@st.cache_resource # 快取模型載入
def load_model(model_path):
    """載入 YOLO 模型"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"載入模型失敗: {e}")
        return None

def draw_bounding_box(frame, box, track_id, class_name):
    """在指定的幀上繪製單一物件的方框和標籤"""
    img_with_box = frame.copy() # 建立副本以避免修改原始幀
    x1, y1, x2, y2 = map(int, box) # 確保座標是整數
    label = f'ID:{track_id} {class_name}'

    # 繪製方框
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # 繪製標籤背景
    (w, h), _ = cv2.getTextSize(label, TEXT_FONT, TEXT_FONT_SCALE, TEXT_THICKNESS)
    label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10 # 防止標籤超出圖片頂部
    cv2.rectangle(img_with_box, (x1, label_y - h - 5), (x1 + w, label_y), BOX_COLOR, -1) # -1 表示填滿

    # 繪製標籤文字 (白色字體在綠色背景上更清晰)
    cv2.putText(img_with_box, label, (x1, label_y - 3), TEXT_FONT, TEXT_FONT_SCALE, (255, 255, 255), TEXT_THICKNESS, lineType=cv2.LINE_AA)

    return img_with_box


def process_video(video_path, model):
    """
    處理影片，偵測並追蹤物件，儲存幀和對應的方框資訊。

    Returns:
        tuple: (tracked_object_frames, representative_frames)
               tracked_object_frames: dict[int, list[tuple(np.ndarray, tuple)]]
                   - 儲存每個 track_id 對應的 (原始幀, 方框座標) 列表
               representative_frames: dict[int, tuple(np.ndarray, str, tuple)]
                   - 儲存每個 track_id 的 (原始幀, 類別名稱, 方框座標)
    """
    tracked_object_frames = defaultdict(list)
    representative_frames = {}
    object_classes = {}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("無法開啟影片檔案。")
            return {}, {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0, text="正在處理影片...")
        frame_count = 0
        start_time = time.time() # 開始計時

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # 使用 YOLO 模型進行追蹤
            results = model.track(
                source=frame,
                tracker=TRACKER_CONFIG,
                classes=TARGET_CLASSES,
                conf=CONFIDENCE_THRESHOLD,
                persist=True,
                verbose=False
            )

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy() # 保留 float 方便後續處理
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    # 儲存這個物件的原始幀和方框座標
                    tracked_object_frames[track_id].append((frame.copy(), tuple(box))) # 儲存幀副本和 box

                    # 如果是第一次看到這個 track_id，儲存代表幀資訊
                    if track_id not in representative_frames:
                        class_name = TARGET_CLASS_NAMES.get(cls_id, f'Class {cls_id}')
                        representative_frames[track_id] = (frame.copy(), class_name, tuple(box)) # 儲存幀副本, 名稱, box
                        object_classes[track_id] = cls_id

            # 更新進度條和 FPS
            progress_percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            progress_bar.progress(progress_percent, text=f"處理中... {progress_percent}% (FPS: {fps:.2f})")

        cap.release()
        progress_bar.progress(100, text="影片處理完成！")

        return tracked_object_frames, representative_frames

    except Exception as e:
        st.error(f"處理影片時發生錯誤: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return {}, {}
    finally:
        if 'progress_bar' in locals():
            progress_bar.empty()


# --- Streamlit 介面 ---

st.set_page_config(page_title="YOLO 物件追蹤 App", layout="wide")
st.title("🎬 YOLOv8 物件偵測與追蹤展示 (含方框標示)")
st.write(f"使用 **{MODEL_PATH}** 模型偵測 **人員 (Person)** 和 **汽車 (Car)**，並在畫面上標示物件。")
st.write("注意：影片處理可能需要一些時間。")

# 初始化 session state
if 'tracked_data' not in st.session_state:
    st.session_state.tracked_data = None
if 'representative_frames' not in st.session_state:
    st.session_state.representative_frames = None
if 'selected_track_id' not in st.session_state:
    st.session_state.selected_track_id = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None


# --- 檔案上傳 ---
uploaded_file = st.file_uploader("請選擇一個影片檔案", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    if not st.session_state.uploaded_file_name or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.tracked_data = None
        st.session_state.representative_frames = None
        st.session_state.selected_track_id = None
        st.session_state.video_processed = False
        st.session_state.uploaded_file_name = uploaded_file.name
        if 'video_path' in st.session_state:
            # 嘗試清理舊的暫存檔
            path_to_clean = st.session_state.get('video_path')
            if path_to_clean and os.path.exists(path_to_clean):
                 try:
                     os.remove(path_to_clean)
                 except Exception:
                     pass # 忽略清理錯誤
            del st.session_state['video_path']


    video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
            st.session_state['video_path'] = video_path

        st.video(video_path)

        model = load_model(MODEL_PATH)

        if model and not st.session_state.video_processed:
            if st.button("🚀 開始處理影片", key="process_btn"):
                with st.spinner(f'模型 ({MODEL_PATH}) 正在努力工作中，請稍候...'):
                    st.session_state.tracked_data = None
                    st.session_state.representative_frames = None
                    st.session_state.selected_track_id = None

                    tracked_data, representative_frames = process_video(video_path, model)

                    st.session_state.tracked_data = tracked_data
                    st.session_state.representative_frames = representative_frames
                    st.session_state.video_processed = True
                    st.rerun()

    finally:
        # 這裡不再需要清理 video_path，因為會在下次上傳或程序結束時由 OS 清理 delete=False 的暫存文件
        # 如果希望更主動清理，需要更複雜的狀態管理
        pass


# --- 結果顯示 ---
if st.session_state.video_processed and st.session_state.representative_frames:
    st.header("📊 追蹤物件結果")
    st.write(f"共偵測並追蹤到 {len(st.session_state.representative_frames)} 個獨立物件 (人員或汽車)。")

    st.subheader("每個物件的代表畫面 (已標示方框)")
    st.write("點擊物件旁的按鈕查看該物件的所有畫面。")

    num_cols = 4 # 調整每行顯示數量
    cols = st.columns(num_cols)
    rep_frames_data = st.session_state.representative_frames
    track_ids = list(rep_frames_data.keys())

    for i, track_id in enumerate(track_ids):
        col_index = i % num_cols
        with cols[col_index]:
            # *** 更新點 1: 取得幀、類別名、方框座標並繪製 ***
            rep_frame_bgr, class_name, box = rep_frames_data[track_id]
            frame_with_box = draw_bounding_box(rep_frame_bgr, box, track_id, class_name)
            rep_frame_rgb = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)

            st.image(rep_frame_rgb, caption=f"物件 ID: {track_id} ({class_name})", use_container_width=True)
            button_key = f"view_btn_{track_id}"
            if st.button(f"查看 ID:{track_id} 所有畫面", key=button_key):
                st.session_state.selected_track_id = track_id
                st.rerun()

    st.markdown("---")

    if st.session_state.selected_track_id is not None:
        selected_id = st.session_state.selected_track_id
        if selected_id in st.session_state.tracked_data:
            frames_data_to_show = st.session_state.tracked_data[selected_id]
            if selected_id in st.session_state.representative_frames:
                _, class_name, _ = st.session_state.representative_frames[selected_id]
                st.subheader(f"🖼️ 物件 ID: {selected_id} ({class_name}) 的所有畫面 ({len(frames_data_to_show)} 幀)")
            else:
                 st.subheader(f"🖼️ 物件 ID: {selected_id} 的所有畫面 ({len(frames_data_to_show)} 幀)")

            max_frames_display = 50
            display_data = frames_data_to_show
            if len(frames_data_to_show) > max_frames_display:
                st.warning(f"此物件出現超過 {max_frames_display} 幀，僅顯示前 {max_frames_display} 幀。")
                display_data = frames_data_to_show[:max_frames_display]

            frame_cols = st.columns(4)
            for idx, frame_data in enumerate(display_data):
                with frame_cols[idx % 4]:
                    # *** 更新點 2: 取得幀和方框座標並繪製 ***
                    frame_bgr, box = frame_data
                    # 在這裡我們只需要框，可以不加 ID 和類別標籤，避免畫面混亂
                    frame_to_display = frame_bgr.copy()
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

                    frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"幀 {idx+1}", use_container_width=True)
        else:
            st.warning(f"找不到 Track ID {selected_id} 的資料。")
            st.session_state.selected_track_id = None

elif st.session_state.video_processed and not st.session_state.representative_frames:
    st.info("影片處理完成，但未偵測或追蹤到任何指定的物件。")

# --- 頁腳 ---
st.markdown("---")
st.caption("由 Ultralytics YOLOv8 和 Streamlit 驅動")