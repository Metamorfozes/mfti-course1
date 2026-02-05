@echo off
setlocal

set "REPO_ROOT=%~dp0"

if exist "%REPO_ROOT%local_config.cmd" (
  call "%REPO_ROOT%local_config.cmd"
)

if not defined TORCH_CUDA_ARCH_LIST (
  set "TORCH_CUDA_ARCH_LIST=7.5"
)

if defined VSDEVCMD (
  if exist "%VSDEVCMD%" (
    call "%VSDEVCMD%" -arch=x64
  ) else (
    echo WARNING: VSDEVCMD is set but does not exist: "%VSDEVCMD%"
  )
) else (
  echo WARNING: VSDEVCMD is not set. C++/CUDA builds may fail.
)

if defined CUDA_HOME (
  if exist "%CUDA_HOME%\bin" (
    set "PATH=%CUDA_HOME%\bin;%PATH%"
  ) else (
    echo WARNING: CUDA_HOME is set but CUDA bin not found: "%CUDA_HOME%\bin"
  )
) else (
  echo WARNING: CUDA_HOME is not set. CUDA builds may fail.
)

if exist "%REPO_ROOT%.venv\Scripts\activate.bat" (
  call "%REPO_ROOT%.venv\Scripts\activate.bat"
) else (
  echo WARNING: .venv not found. Activate your virtual environment manually.
)

if "%~1"=="" (
  cmd /k
) else (
  cmd /k "%*"
)
