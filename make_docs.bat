pdoc -o docs/ -d numpy --math mech447/
RMDIR /S /Q "mech447\__pycache__"
DEL "mech447\requirements.txt"
pipreqs mech447/