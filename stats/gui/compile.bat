echo on
pyinstaller --name="Statistics" --onefile --windowed --clean --icon=pyc.ico --splash splash.png --optimize 2 --upx-dir upx gui.py
pause