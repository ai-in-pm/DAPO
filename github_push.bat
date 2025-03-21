@echo off
REM GitHub Repository Setup Script
REM This script adds all files, commits them, and pushes to GitHub

echo.
echo Preparing to push to GitHub repository: https://github.com/ai-in-pm/DAPO
echo.

REM Add all files
echo Adding files to git...
git add .

REM Commit changes
echo.
echo Committing changes...
git commit -m "Initial DAPO implementation"

REM Set up the main branch
echo.
echo Setting up main branch...
git branch -M main

REM Push to GitHub
echo.
echo Ready to push to GitHub. 
echo You will need to authenticate with your GitHub credentials.
echo.
echo Run the following command to push to GitHub:
echo     git push -u origin main
echo.

pause
