@echo off
call "%~dp0run_env.cmd" "set \"PY=%~dp0.venv\Scripts\python.exe\" && cd /d %~dp0external\stylegan-nada\ZSSGAN && \"%PY%\" train.py --frozen_gen_ckpt ..\..\course1_anime_faces\models\stylegan2-ffhq-config-f.pt --output_dir ..\..\course1_anime_faces\results\anime_mapping --source_class photo --target_class \"anime face\""
