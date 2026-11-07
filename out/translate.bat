@echo off
setlocal DisableDelayedExpansion
if not defined LZTRANSLATE_APP_HOME set "LZTRANSLATE_APP_HOME=%~d0\APP\LzTranslate"
set "PY=%LZTRANSLATE_APP_HOME%\engine\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python.exe"
"%PY%" -B "%~dp0..\mod\translate\pipeline.py" %*
exit /b %ERRORLEVEL%
