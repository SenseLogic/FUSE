pip freeze > temp.txt
pip uninstall -r temp.txt -y
del /q temp.txt
pip cache purge
pause
