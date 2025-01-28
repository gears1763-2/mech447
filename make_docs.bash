#! /usr/bin/bash

pdoc -o docs/ -d numpy --math mech447/
rm -rf mech447/__pycache__
rm mech447/requirements.txt
pipreqs mech447/
