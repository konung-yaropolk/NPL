echo on
pyinstaller --name="Statistics" --windowed --onefile --icon=pyc.ico gui.py datas=('mainwindow.ui')
pause