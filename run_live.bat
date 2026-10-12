@echo off
cd /d %~dp0
.venv\Scripts\python live.py --monitor
pause
