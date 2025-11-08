@echo off
REM ============================================================
REM Sync Script: Push Local Changes & Pull Claude's Updates
REM Branch: claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
REM ============================================================

echo.
echo ============================================================
echo Syncing with Claude's Branch
echo ============================================================
echo.

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo ERROR: Not in a git repository!
    echo Please run this script from your Quantum-Field-Dynamics directory.
    pause
    exit /b 1
)

REM Show current branch
echo Current branch:
git branch --show-current
echo.

REM Show current status
echo Current status:
git status --short
echo.

REM Ask user if they want to commit local changes
set /p commit_choice="Do you have local changes to commit? (y/n): "
if /i "%commit_choice%"=="y" (
    echo.
    set /p commit_msg="Enter commit message: "

    echo.
    echo Adding all changes...
    git add .

    echo Committing changes...
    git commit -m "!commit_msg!"

    if errorlevel 1 (
        echo WARNING: Commit failed. Continuing anyway...
    ) else (
        echo ✓ Changes committed successfully
    )
    echo.
)

REM Fetch latest from remote
echo Fetching latest from remote...
git fetch origin
if errorlevel 1 (
    echo ERROR: Failed to fetch from remote!
    pause
    exit /b 1
)
echo ✓ Fetch complete
echo.

REM Check out the Claude branch
echo Checking out Claude's branch...
git checkout claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
if errorlevel 1 (
    echo ERROR: Failed to checkout branch!
    pause
    exit /b 1
)
echo ✓ On Claude's branch
echo.

REM Pull latest changes
echo Pulling latest changes from Claude...
git pull origin claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
if errorlevel 1 (
    echo ERROR: Failed to pull changes!
    echo You may have merge conflicts. Please resolve them manually.
    pause
    exit /b 1
)
echo ✓ Pull complete
echo.

REM Push any local commits
echo Pushing your local commits (if any)...
git push origin claude/review-qfd-supernova-011CUqvpoueZRwQk1fHMY1D3
if errorlevel 1 (
    echo WARNING: Push failed. This is normal if you have no new commits.
) else (
    echo ✓ Push complete
)
echo.

REM Show final status
echo ============================================================
echo Final Status
echo ============================================================
git status
echo.

REM Show recent commits
echo ============================================================
echo Recent Commits (last 5)
echo ============================================================
git log --oneline -5
echo.

echo ============================================================
echo ✓ Sync Complete!
echo ============================================================
echo.
echo Your repository is now up to date with Claude's latest changes.
echo You can now view the files in your editor or IDE.
echo.

pause
