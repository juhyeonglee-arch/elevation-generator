@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 입면도 자동 생성
set PYTHONHOME=
set PYTHONPATH=
set PYTHONSTARTUP=
set PYTHONNOUSERSITE=1
set PYTHONUTF8=1
if not exist "%~dp0python\python.exe" (
  echo 압축이 다 풀리지 않았습니다. zip 파일을 오른쪽 클릭 - "모두 압축 풀기" 한 뒤 풀린 폴더에서 실행하세요.
  pause
  exit /b 1
)
echo 브라우저에 화면이 열립니다. 이 검은 창을 닫으면 프로그램이 꺼집니다.
echo 결과는 "%USERPROFILE%\입면도" 폴더에 저장됩니다.
"%~dp0python\python.exe" -s -m elevation_ai %*
pause
