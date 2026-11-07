@echo off
setlocal DisableDelayedExpansion
if not defined LZTRANSLATE_APP_HOME set "LZTRANSLATE_APP_HOME=%~d0\APP\LzTranslate"
set "ENGINE=%LZTRANSLATE_APP_HOME%\engine\ollama\ollama.exe"
if not exist "%ENGINE%" (
  echo [ERROR] Install the engine first using out\install.bat --engine
  exit /b 2
)
set "OLLAMA_HOST=127.0.0.1:12545"
set "OLLAMA_MODELS=%LZTRANSLATE_APP_HOME%\models\ollama"
set "OLLAMA_NO_CLOUD=1"
"%ENGINE%" serve
exit /b %ERRORLEVEL%
