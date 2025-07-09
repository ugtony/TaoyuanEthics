import streamlit.web.cli as stcli
import os
import sys
import webbrowser
import threading
import time

def get_resource_path(relative_path):
    """ 獲取資源的絕對路徑，對開發環境和 PyInstaller 都有效 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def open_browser():
    """ 在伺服器啟動後，延遲一小段時間再開啟瀏覽器 """
    time.sleep(1.5) # 給伺服器一點啟動時間
    webbrowser.open("http://localhost:8501")

if __name__ == '__main__':
    # 使用一個獨立的執行緒來開啟瀏覽器，才不會卡住主程式
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True  # 設定為守護執行緒，主程式結束時會自動關閉
    browser_thread.start()

    # --- 以下是您原有的伺服器啟動邏輯 ---
    main_script_path = get_resource_path('app.py')

    streamlit_commands = [
        "run",
        main_script_path,
        "--server.port", "8501",
        "--global.developmentMode=false",
        "--server.headless", "true",
        "--server.maxUploadSize", "2048"
    ]

    sys.argv = [sys.argv[0]] + streamlit_commands

    stcli.main()
