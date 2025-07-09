# run_app.spec (偵錯專用版)

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

block_cipher = None

# 收集 streamlit 的所有必要檔案
streamlit_data = collect_data_files('streamlit')
streamlit_metadata = copy_metadata('streamlit')

# Analysis 的設定
a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=streamlit_data + streamlit_metadata + [
        ('models', 'models'),
        ('app.py', '.')
    ],
    hiddenimports=[
        'ultralytics.engine',
        'ultralytics.utils',
        'torch.nn.modules.utils',
        'streamlit.web.cli',
        'streamlit.runtime.scriptrunner.magic_funcs'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # --- 關鍵修改：將 console 設定為 True，強制顯示主控台視窗 ---
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 收集所有依賴的二進位檔案
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run_app',
    )
