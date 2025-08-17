@echo off
setlocal
set PY="%~dp0..\python\python.exe"
set CLI=%~dp0whisper_cli.py
REM -W ignore silences Python warnings at the interpreter level
%PY% -W ignore "%CLI%" %*
