import cv2
import tempfile
import os

import torch
torch.classes.__path__ = [] # 把 torch.classes 裡面的 __path__ 清掉，Streamlit 就無路可走、也不會拋例外

import streamlit as st
from ultralytics import YOLO, YOLOWorld
import numpy as np
from collections import defaultdict
import time # 引入 time 模組
import logging # 引入 logging 模組

# --- 設定 Logging ---
# 配置 logging，以便更容易地追蹤錯誤
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 設定 ---
# 使用 YOLO-World 模型，可以選擇 'yolov8s-world', 'yolov8m-world', 'yolov8l-world', 'yolov8x-world'
MODEL_PATH = 'yolov8s-worldv2.pt'
CONFIDENCE_THRESHOLD = 0.1 # YOLO-World 可能需要較低的閾值來偵測更多物件，可自行調整
TRACKER_CONFIG = 'bytetrack.yaml' # ByteTrack 追蹤器
BOX_COLOR = (0, 255, 0) # BGR 格式的綠色
BOX_THICKNESS = 2
TEXT_COLOR = (0, 0, 0) # 黑色文字，在綠色背景上更清晰
TEXT_BG_COLOR = (0, 255, 0) # 綠色背景
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Helper 函式 ---

# 使用 Streamlit 快取來載入模型，避免每次重新執行都載入
@st.cache_resource
def load_model(model_path):
    """載入 YOLO-World 模型"""
    try:
        logging.info(f"正在嘗試載入模型: {model_path}")
        #model = YOLOWorld(model_path)
        model = YOLO(model_path)
        st.success(f"成功載入模型: {model_path}")
        logging.info(f"成功載入模型: {model_path}")
        return model
    except Exception as e:
        st.error(f"載入模型失敗: {e}")
        logging.error(f"載入模型失敗: {e}", exc_info=True) # 記錄詳細錯誤
        return None

def draw_bounding_box(frame, box, track_id, class_name, conf):
    """在指定的幀上繪製單一物件的方框和標籤 (包含信賴度)"""
    # 建立副本以避免修改原始幀
    img_with_box = frame.copy()
    # 確保座標是整數
    x1, y1, x2, y2 = map(int, box)
    # 格式化標籤，包含 Track ID, 類別名和信賴度
    label = f'ID:{track_id} {class_name} {conf:.2f}'

    # --- 繪製方框 ---
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

    # --- 繪製標籤背景 ---
    # 取得文字大小
    (w, h), _ = cv2.getTextSize(label, TEXT_FONT, TEXT_FONT_SCALE, TEXT_THICKNESS)
    # 計算標籤位置，避免超出圖片頂部
    label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
    # 確保背景框不會超出圖片左邊界
    label_x1 = max(0, x1)
    label_x2 = label_x1 + w
    # 繪製填滿的背景矩形
    cv2.rectangle(img_with_box, (label_x1, label_y - h - 5), (label_x2, label_y), TEXT_BG_COLOR, -1)

    # --- 繪製標籤文字 ---
    cv2.putText(img_with_box, label, (label_x1, label_y - 3), TEXT_FONT, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, lineType=cv2.LINE_AA)

    return img_with_box


def process_video(video_path, model, target_classes):
    """
    處理影片，偵測並追蹤由 target_classes 指定的物件類別，儲存幀和對應的方框資訊。

    Args:
        video_path (str): 影片檔案的路徑。
        model (YOLOWorld): 已載入的 YOLO-World 模型。
        target_classes (list): 使用者指定的目標類別名稱列表。

    Returns:
        tuple: (tracked_object_frames, representative_frames)
               tracked_object_frames: dict[int, list[tuple(np.ndarray, tuple, float)]]
                   - 儲存每個 track_id 對應的 (原始幀, 方框座標, 信賴度) 列表
               representative_frames: dict[int, tuple(np.ndarray, str, tuple, float)]
                   - 儲存每個 track_id 的 (原始幀, 類別名稱, 方框座標, 信賴度)
    """
    tracked_object_frames = defaultdict(list)
    representative_frames = {}
    cap = None # 初始化 cap
    progress_bar = None # 初始化 progress_bar

    try:
        # --- 設定模型偵測類別 ---
        logging.info(f"正在設定模型偵測目標: {', '.join(target_classes)}")
        model.set_classes(target_classes)
        logging.info("模型目標設定完成。")
        st.success("模型目標設定完成。") # 在介面顯示成功訊息

        # --- 開啟影片檔案 ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("無法開啟影片檔案。")
            logging.error(f"無法開啟影片檔案: {video_path}")
            return {}, {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 顯示進度條
        progress_bar = st.progress(0, text="正在處理影片...")
        frame_count = 0
        start_time = time.time() # 開始計時

        # --- 逐幀處理 ---
        while True:
            success, frame = cap.read()
            # 如果讀取失敗或影片結束，則跳出迴圈
            if not success:
                break

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            # 計算 FPS (每秒幀數)
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # --- 使用 YOLO-World 模型進行追蹤 ---
            # `classes` 參數不需要，因為已經透過 `set_classes` 設定
            results = model.track(
                source=frame,
                tracker=TRACKER_CONFIG, # 指定追蹤器
                conf=CONFIDENCE_THRESHOLD, # 套用信賴度閾值
                persist=True, # 保持追蹤狀態
                verbose=False # 減少控制台輸出
            )

            # --- 提取追蹤結果 ---
            # 檢查是否有結果、是否有方框、是否有追蹤 ID
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                # 獲取方框座標 (xyxy 格式)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                # 獲取追蹤 ID
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                # 獲取預測的類別 ID
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                # 獲取信賴度分數
                confidences = results[0].boxes.conf.cpu().numpy()
                # 獲取當前模型設定的類別名稱映射 (例如 {0: 'person', 1: 'car'})
                class_name_map = results[0].names

                # --- 處理每個偵測到的物件 ---
                for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                    # 再次確認信賴度 (雖然 track 內部已過濾，但保留檢查更安全)
                    if conf >= CONFIDENCE_THRESHOLD:
                        # 從映射中獲取類別名稱，如果找不到則使用預設值
                        #print(class_name_map)
                        #class_name = class_name_map.get(cls_id, f'未知類別 {cls_id}')
                        class_name = target_classes[cls_id]

                        # 儲存這個物件的原始幀、方框座標和信賴度
                        # 使用 frame.copy() 確保儲存的是獨立副本
                        tracked_object_frames[track_id].append((frame.copy(), tuple(box), conf))

                        # 如果是第一次看到這個 track_id，儲存其代表幀資訊
                        # 這通常是該物件第一次被偵測到的畫面
                        if track_id not in representative_frames:
                            representative_frames[track_id] = (frame.copy(), class_name, tuple(box), conf)

            # --- 更新進度條 ---
            # 計算處理進度百分比
            progress_percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            # 更新進度條顯示文字和進度
            progress_bar.progress(progress_percent, text=f"處理中... {progress_percent}% (FPS: {fps:.2f})")

        # --- 處理完成 ---
        cap.release() # 釋放影片資源
        progress_bar.progress(100, text="影片處理完成！") # 將進度條設為 100%
        logging.info("影片處理完成。")

        return tracked_object_frames, representative_frames

    except Exception as e:
        st.error(f"處理影片時發生錯誤: {e}")
        logging.error(f"處理影片時發生錯誤: {e}", exc_info=True) # 記錄詳細錯誤
        # 確保釋放資源
        if cap is not None and cap.isOpened():
            cap.release()
        return {}, {}
    finally:
        # 確保進度條被移除
        if progress_bar is not None:
            progress_bar.empty()


# --- Streamlit 介面 ---

st.set_page_config(page_title="YOLO-World 物件追蹤 App", layout="wide")
st.title("🎬 YOLO-World 開放詞彙物件偵測與追蹤")
st.write(f"使用 **{MODEL_PATH}** 模型，您可以輸入想偵測的物件名稱！")
st.info("提示：輸入多個物件請用逗號 (`,`) 分隔，例如：`person, dog, backpack`")
st.write("注意：影片處理可能需要一些時間，特別是第一次下載和載入 YOLO-World 模型時。")

# --- 初始化 Session State ---
# 使用 Session State 來保存應用程式狀態，避免每次互動都重置
if 'tracked_data' not in st.session_state:
    st.session_state.tracked_data = None # 儲存所有追蹤物件的幀數據
if 'representative_frames' not in st.session_state:
    st.session_state.representative_frames = None # 儲存每個物件的代表幀
if 'selected_track_id' not in st.session_state:
    st.session_state.selected_track_id = None # 儲存使用者選擇查看的物件 ID
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False # 標記影片是否已被處理
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None # 儲存上傳檔案的名稱，用於偵測新檔案
if 'last_processed_prompt' not in st.session_state:
    st.session_state.last_processed_prompt = "" # 儲存上次處理時使用的提示詞
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = "person, car" # 預設的偵測目標
if 'video_path' not in st.session_state:
    st.session_state.video_path = None # 儲存暫存影片檔案的路徑
if 'model' not in st.session_state:
    st.session_state.model = None # 儲存載入的模型

# --- 使用者輸入偵測目標 ---
# 允許使用者輸入想偵測的物件名稱
st.session_state.current_prompt = st.text_input(
    "請輸入想偵測的物件名稱 (用逗號分隔):",
    value=st.session_state.current_prompt, # 使用 session state 中的值作為預設值
    placeholder="例如: person, dog, backpack, traffic light"
)

# --- 檔案上傳 ---
uploaded_file = st.file_uploader("請選擇一個影片檔案", type=["mp4", "avi", "mov", "mkv"])

# --- 主要邏輯 ---
if uploaded_file is not None:
    # --- 延遲模型載入 ---
    # 只有在檔案上傳後才嘗試載入模型
    if st.session_state.model is None:
        with st.spinner(f"正在載入 YOLO-World 模型 ({MODEL_PATH})..."):
            st.session_state.model = load_model(MODEL_PATH)

    # 只有在模型成功載入後才繼續
    if st.session_state.model is not None:
        model = st.session_state.model # 取得載入的模型
        new_upload = False
        # 檢查是否是新的檔案上傳
        if st.session_state.uploaded_file_name != uploaded_file.name:
            logging.info(f"偵測到新檔案上傳: {uploaded_file.name}")
            # 清理舊狀態
            st.session_state.tracked_data = None
            st.session_state.representative_frames = None
            st.session_state.selected_track_id = None
            st.session_state.video_processed = False
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.last_processed_prompt = ""
            new_upload = True

            # --- 清理舊的暫存檔 (如果存在) ---
            old_path = st.session_state.get('video_path')
            if old_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                    logging.info(f"已清理舊暫存檔: {old_path}")
                    st.session_state.video_path = None # 清除舊路徑
                except Exception as clean_err:
                    logging.warning(f"清理舊暫存檔失敗: {clean_err}")
                    pass # 忽略清理錯誤

        # --- 處理影片檔案，保存到暫存檔 ---
        # 如果是新上傳或暫存檔不存在，則創建新的暫存檔
        if new_upload or st.session_state.video_path is None or not os.path.exists(st.session_state.video_path):
            try:
                # 創建一個具名的暫存檔，delete=False 表示檔案關閉後不會自動刪除
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    # 將上傳檔案的內容寫入暫存檔
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.video_path = tmp_file.name # 保存暫存檔路徑
                    logging.info(f"創建新的影片暫存檔: {st.session_state.video_path}")
            except Exception as tmp_err:
                st.error(f"創建暫存檔失敗: {tmp_err}")
                logging.error(f"創建暫存檔失敗: {tmp_err}", exc_info=True)
                st.session_state.video_path = None # 出錯時確保路徑為 None

        # --- 顯示影片預覽和處理按鈕 ---
        video_path = st.session_state.video_path
        if video_path and os.path.exists(video_path):
            st.video(video_path) # 顯示影片預覽

            # 檢查提示詞是否有變更
            prompt_changed = st.session_state.current_prompt != st.session_state.last_processed_prompt

            # 決定是否顯示處理按鈕
            # 條件：影片尚未處理，或者提示詞已變更
            show_process_button = not st.session_state.video_processed or prompt_changed

            if show_process_button:
                if prompt_changed and st.session_state.video_processed:
                    st.info("偵測目標已變更，需要重新處理影片。")
                    # 重置狀態以便重新處理
                    st.session_state.tracked_data = None
                    st.session_state.representative_frames = None
                    st.session_state.selected_track_id = None
                    st.session_state.video_processed = False

                # 設定按鈕標籤
                process_button_label = "🚀 開始處理影片"
                if prompt_changed and not new_upload: # 只有在提示詞改變且不是新上傳時才顯示"重新處理"
                    process_button_label = "🔄 使用新的目標重新處理影片"

                # 顯示處理按鈕
                if st.button(process_button_label, key="process_btn"):
                    # 1. 取得並處理使用者輸入的類別
                    # 分割字串、去除空白、過濾空字串
                    classes_to_detect = [s.strip() for s in st.session_state.current_prompt.split(',') if s.strip()]

                    if not classes_to_detect:
                        st.warning("請至少輸入一個有效的物件名稱才能開始處理！")
                    else:
                        # 顯示處理中的提示
                        with st.spinner(f'YOLO-World ({MODEL_PATH}) 正在設定目標並處理影片，請稍候...'):
                            try:
                                # 2. 清空舊結果並處理影片
                                # (模型設定已移至 process_video 內部)
                                st.session_state.tracked_data = None
                                st.session_state.representative_frames = None
                                st.session_state.selected_track_id = None

                                # 呼叫影片處理函式
                                tracked_data, representative_frames = process_video(video_path, model, classes_to_detect)

                                # 3. 更新 session state
                                st.session_state.tracked_data = tracked_data
                                st.session_state.representative_frames = representative_frames
                                st.session_state.video_processed = True
                                # 記錄這次處理使用的提示詞
                                st.session_state.last_processed_prompt = st.session_state.current_prompt
                                # 重新執行腳本以更新介面顯示結果
                                st.rerun()

                            except Exception as process_err:
                                st.error(f"處理影片過程中發生未預期的錯誤: {process_err}")
                                logging.error(f"處理影片過程中發生未預期的錯誤: {process_err}", exc_info=True)
                                # 重置處理狀態
                                st.session_state.video_processed = False

# --- 結果顯示區 ---
# 只有在影片處理完成且有結果時才顯示
if st.session_state.video_processed and st.session_state.representative_frames:
    st.header("📊 追蹤物件結果")
    st.write(f"基於您的提示詞 '{st.session_state.last_processed_prompt}'，共偵測並追蹤到 {len(st.session_state.representative_frames)} 個獨立物件。")

    st.subheader("每個物件的代表畫面 (已標示方框)")
    st.write("點擊物件旁的按鈕查看該物件的所有畫面。")

    num_cols = 4 # 設定每行顯示的代表畫面數量
    cols = st.columns(num_cols)
    # 獲取代表幀數據
    rep_frames_data = st.session_state.representative_frames
    # 對 Track ID 進行排序，確保顯示順序一致
    track_ids = sorted(list(rep_frames_data.keys()))

    # --- 顯示每個物件的代表幀 ---
    for i, track_id in enumerate(track_ids):
        col_index = i % num_cols # 計算當前物件應在哪一列顯示
        with cols[col_index]:
            # 從 representative_frames 獲取該物件的數據
            rep_frame_bgr, class_name, box, conf = rep_frames_data[track_id]
            # 使用 helper 函式繪製帶有完整資訊 (ID, 類別, 信賴度) 的方框
            frame_with_box = draw_bounding_box(rep_frame_bgr, box, track_id, class_name, conf)
            # 將 BGR 格式轉換為 RGB 格式以在 Streamlit 中正確顯示
            rep_frame_rgb = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)

            # 顯示圖片和標題
            st.image(rep_frame_rgb, caption=f"物件 ID: {track_id} ({class_name}, conf: {conf:.2f})", use_container_width=True)
            # 為每個物件創建一個按鈕，用於查看其所有幀
            button_key = f"view_btn_{track_id}"
            if st.button(f"查看 ID:{track_id} 所有畫面", key=button_key):
                # 如果按鈕被點擊，則更新 session state 中的 selected_track_id
                st.session_state.selected_track_id = track_id
                # 重新執行腳本以顯示選中物件的幀
                st.rerun()

    st.markdown("---") # 分隔線

    # --- 顯示選定物件的所有幀 ---
    # 檢查是否有選定的物件 ID
    if st.session_state.selected_track_id is not None:
        selected_id = st.session_state.selected_track_id
        # 檢查選定 ID 的數據是否存在
        if selected_id in st.session_state.tracked_data:
            # 獲取該物件的所有幀數據
            frames_data_to_show = st.session_state.tracked_data[selected_id]
            # 從 representative_frames 獲取類別名稱以顯示標題
            class_name_display = "物件" # 預設名稱
            if selected_id in st.session_state.representative_frames:
                _, class_name_display, _, _ = st.session_state.representative_frames[selected_id]

            st.subheader(f"🖼️ 物件 ID: {selected_id} ({class_name_display}) 的所有畫面 ({len(frames_data_to_show)} 幀)")

            max_frames_display = 60 # 設定最多顯示的幀數上限
            display_data = frames_data_to_show
            # 如果幀數超過上限，顯示警告並截斷數據
            if len(frames_data_to_show) > max_frames_display:
                st.warning(f"此物件出現超過 {max_frames_display} 幀，僅顯示前 {max_frames_display} 幀。")
                # 也可以考慮抽樣顯示，例如每隔幾幀顯示一幀
                # display_data = frames_data_to_show[::max(1, len(frames_data_to_show)//max_frames_display)]
                display_data = frames_data_to_show[:max_frames_display] # 目前只顯示前 N 幀

            frame_cols = st.columns(5) # 設定每行顯示的幀數
            # --- 顯示選定物件的每一幀 ---
            for idx, frame_data in enumerate(display_data):
                with frame_cols[idx % 5]: # 計算當前幀應在哪一列顯示
                    # 解包幀數據 (原始幀 BGR, 方框座標, 信賴度)
                    frame_bgr, box, conf = frame_data
                    # 繪製簡單的方框，不加標籤，避免畫面混亂
                    frame_to_display = frame_bgr.copy()
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                    # 可以在這裡加上信賴度文字 (可選)
                    # cv2.putText(frame_to_display, f"{conf:.2f}", (x1, y1 - 5), TEXT_FONT, 0.4, TEXT_COLOR, 1)

                    # 轉換為 RGB 格式
                    frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                    # 顯示單幀圖片
                    st.image(frame_rgb, caption=f"幀 {idx+1} (Conf: {conf:.2f})", use_container_width=True)
        else:
            # 如果找不到選定 ID 的數據，顯示警告
            st.warning(f"找不到 Track ID {selected_id} 的資料。")
            # 清除選擇，避免下次刷新時出錯
            st.session_state.selected_track_id = None

# 如果影片處理完成，但沒有偵測到任何物件
elif st.session_state.video_processed and not st.session_state.representative_frames:
    st.info(f"影片處理完成，但未偵測或追蹤到任何符合 '{st.session_state.last_processed_prompt}' 的物件。請檢查提示詞或調整信賴度閾值。")

# --- 清理暫存檔案 ---
# Streamlit 在 session 結束時通常會清理 delete=False 的 NamedTemporaryFile
# 但如果需要更明確的控制 (例如在應用程式關閉時)，可以使用 atexit 模組
# import atexit
# def cleanup():
#     video_path = st.session_state.get('video_path')
#     if video_path and os.path.exists(video_path):
#         try:
#             os.remove(video_path)
#             logging.info(f"應用程式結束，已清理暫存檔: {video_path}")
#         except Exception as e:
#             logging.warning(f"應用程式結束時清理暫存檔失敗: {e}")
# atexit.register(cleanup)
# 注意：在 Streamlit Cloud 等環境中，檔案系統可能是臨時的，atexit 可能不總是可靠

# --- 頁腳 ---
st.markdown("---")
st.caption(f"由 Ultralytics YOLO-World ({MODEL_PATH}) 和 Streamlit 驅動")