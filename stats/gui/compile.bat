echo on
pyinstaller --name="Statistics" --onefile --clean --icon=pyc.ico --optimize 2 --upx-dir upx  gui.py
pause