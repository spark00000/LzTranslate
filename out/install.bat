@echo off
setlocal DisableDelayedExpansion
python.exe -B "%~dp0..\mod\translate\install.py" %*
exit /b %ERRORLEVEL%
