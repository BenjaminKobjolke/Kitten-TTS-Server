@echo off
setlocal
if not exist "%~dp0analyze_code_config.bat" (
    echo ERROR: analyze_code_config.bat not found.
    echo Copy analyze_code_config.example.bat to analyze_code_config.bat and set CLI_ANALYZER_PATH and LANGUAGE.
    exit /b 1
)
call "%~dp0analyze_code_config.bat"
if errorlevel 1 exit /b %ERRORLEVEL%
cd /d "%~dp0.."
"%CLI_ANALYZER_PATH%\venv\Scripts\python.exe" "%CLI_ANALYZER_PATH%\main.py" --language %LANGUAGE% --path "." --verbosity minimal --output "code_analysis_results" --maxamountoferrors 50 --rules "code_analysis_rules.json" --fail-on-warnings
set ANALYZER_RESULT=%ERRORLEVEL%
cd /d "%~dp0"
exit /b %ANALYZER_RESULT%
