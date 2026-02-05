@echo off
call "%~dp0run_env.cmd" "set \"PY=%~dp0.venv\Scripts\python.exe\" && cd /d %~dp0external\stylegan-nada\ZSSGAN && \"%PY%\" train.py --help"
