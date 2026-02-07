@echo off
setlocal enabledelayedexpansion

REM Always run from repo root
pushd "%~dp0"

REM Load user-local config (ignored by git)
if exist "local_config.cmd" (
  call "local_config.cmd"
) else (
  echo WARNING: local_config.cmd not found. Copy local_config.example.cmd to local_config.cmd
)

REM Validate VS DevCmd
call "%~dp0local_config.cmd"

if not defined VSDEVCMD (
  echo WARNING: VSDEVCMD is not set. C++/CUDA builds may fail.
) else (
  if exist "%VSDEVCMD%" (
    call "%VSDEVCMD%" -arch=x64
  ) else (
    echo WARNING: VSDEVCMD path does not exist: "%VSDEVCMD%"
  )
)

REM CUDA vars (optional, but helps torch extensions)
if defined CUDA_HOME (
  set "PATH=%CUDA_HOME%\bin;%PATH%"
)

REM Run patcher + clean torch extensions cache
if exist "tools\patch_stylegan_nada_windows.py" (
  "%~dp0.venv\Scripts\python.exe" "tools\patch_stylegan_nada_windows.py"
) else (
  echo ERROR: tools\patch_stylegan_nada_windows.py not found
)

if exist "tools\clean_torch_extensions_cache.py" (
  "%~dp0.venv\Scripts\python.exe" "tools\clean_torch_extensions_cache.py"
)

popd
endlocal
