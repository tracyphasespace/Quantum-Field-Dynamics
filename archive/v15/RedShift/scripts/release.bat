@echo off
REM Release automation script for QFD CMB Module

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Function to display usage
if "%1"=="" goto usage
if "%1"=="help" goto usage
if "%1"=="-h" goto usage
if "%1"=="--help" goto usage

set "COMMAND=%1"
shift

if "%COMMAND%"=="validate" goto validate
if "%COMMAND%"=="prepare" goto prepare
if "%COMMAND%"=="tag" goto tag

echo Error: Unknown command '%COMMAND%'
goto usage

:validate
echo 🔍 Validating release readiness...
python "%SCRIPT_DIR%prepare_release.py" validate
if %ERRORLEVEL% equ 0 (
    echo ✅ Ready for release!
) else (
    echo ❌ Release validation failed
    exit /b 1
)
goto end

:prepare
if "%1"=="" (
    echo Error: BUMP_TYPE required for prepare command
    goto usage
)

set "BUMP_TYPE=%1"
set "PRERELEASE=%2"

echo 🚀 Preparing %BUMP_TYPE% release...

if not "%PRERELEASE%"=="" (
    python "%SCRIPT_DIR%prepare_release.py" prepare --bump-type "%BUMP_TYPE%" --prerelease "%PRERELEASE%"
) else (
    python "%SCRIPT_DIR%prepare_release.py" prepare --bump-type "%BUMP_TYPE%"
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo 📝 Review the changes and then run:
    echo    git diff --cached
    echo    git commit -m "Prepare release"
    echo    %0 tag
) else (
    echo ❌ Release preparation failed
    exit /b 1
)
goto end

:tag
set "VERSION=%1"
set "MESSAGE=%2"

echo 🏷️ Creating release tag...

if not "%VERSION%"=="" (
    if not "%MESSAGE%"=="" (
        python "%SCRIPT_DIR%prepare_release.py" tag --version "%VERSION%" --message "%MESSAGE%"
    ) else (
        python "%SCRIPT_DIR%prepare_release.py" tag --version "%VERSION%"
    )
) else (
    python "%SCRIPT_DIR%prepare_release.py" tag
)

if %ERRORLEVEL% equ 0 (
    echo ✅ Tag created and pushed!
    echo 🚀 GitHub Actions will now build and publish the release.
) else (
    echo ❌ Tag creation failed
    exit /b 1
)
goto end

:usage
echo Usage: %0 [prepare^|validate^|tag] [options]
echo.
echo Commands:
echo   prepare BUMP_TYPE [PRERELEASE]  Prepare a new release
echo   validate                        Validate release readiness
echo   tag [VERSION] [MESSAGE]         Create and push git tag
echo.
echo Bump types: major, minor, patch, prerelease
echo.
echo Examples:
echo   %0 validate                     # Check if ready for release
echo   %0 prepare patch                # Prepare patch release
echo   %0 prepare minor                # Prepare minor release
echo   %0 prepare prerelease alpha     # Prepare alpha prerelease
echo   %0 tag                          # Tag current version
echo   %0 tag 1.0.0 "Release 1.0.0"   # Tag specific version
echo.
exit /b 1

:end