@echo off
REM Version bumping utility script for QFD CMB Module

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Function to display usage
if "%1"=="" goto usage
if "%1"=="help" goto usage
if "%1"=="-h" goto usage
if "%1"=="--help" goto usage

set "BUMP_TYPE=%1"
set "PRERELEASE=%2"

REM Validate bump type
if "%BUMP_TYPE%"=="major" goto valid
if "%BUMP_TYPE%"=="minor" goto valid
if "%BUMP_TYPE%"=="patch" goto valid
if "%BUMP_TYPE%"=="prerelease" goto valid

echo Error: Invalid bump type '%BUMP_TYPE%'
goto usage

:valid
REM Get current version
echo Current version:
python "%SCRIPT_DIR%version_manager.py" get

REM Run version bump
if "%BUMP_TYPE%"=="prerelease" (
    if not "%PRERELEASE%"=="" (
        python "%SCRIPT_DIR%version_manager.py" bump --bump-type "%BUMP_TYPE%" --prerelease "%PRERELEASE%"
    ) else (
        python "%SCRIPT_DIR%version_manager.py" bump --bump-type "%BUMP_TYPE%"
    )
) else (
    python "%SCRIPT_DIR%version_manager.py" bump --bump-type "%BUMP_TYPE%"
)

if %ERRORLEVEL% equ 0 (
    echo Version bump completed successfully!
) else (
    echo Error occurred during version bump
    exit /b 1
)

goto end

:usage
echo Usage: %0 [major^|minor^|patch^|prerelease] [prerelease-name]
echo.
echo Examples:
echo   %0 patch                    # 1.0.0 -^> 1.0.1
echo   %0 minor                    # 1.0.0 -^> 1.1.0
echo   %0 major                    # 1.0.0 -^> 2.0.0
echo   %0 prerelease alpha         # 1.0.0 -^> 1.0.0-alpha.1
echo.
exit /b 1

:end