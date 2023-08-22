

call %~dp0..\env\conda_activate.bat


python -O %SRC_DIR%\superd\hyd\_04evaluation.py

cmd.exe /k