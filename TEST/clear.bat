pip freeze > temp.txt
pip uninstall -r temp.txt -y
del /q temp.txt
rem pip cache purge
rem rd /s /q "%USERPROFILE%\.cache\huggingface"
pause
