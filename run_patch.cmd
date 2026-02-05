@echo off
call "%~dp0run_env.cmd" "set \"PY=%~dp0.venv\Scripts\python.exe\" && \"%PY%\" tools\patch_stylegan_nada_windows.py || exit /b 1 && \"%PY%\" tools\clean_torch_extensions_cache.py"
