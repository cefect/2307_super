

call %~dp0..\env\conda_activate.bat


python -O %SRC_DIR%\superd\hyd\asc_to_concat.py

cmd.exe /k