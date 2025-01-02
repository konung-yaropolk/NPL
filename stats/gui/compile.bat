echo on
pyinstaller --name="Statistics" --onefile --windowed --clean --icon=pyc.ico --splash splash.png --optimize 1 --upx-dir upx gui.py
@REM dont use optimization level 2, it causes ignoring assertions in python code
pause