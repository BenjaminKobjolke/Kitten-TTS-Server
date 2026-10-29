@echo off
REM Signals the running server to shut down; it polls for this file every second.
type nul > "%~dp0..\_shutdown"
echo Shutdown requested.
exit /b 0
