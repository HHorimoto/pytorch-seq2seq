#!/bin/sh -x

. ./.venv/bin/activate

now=`date "+%F_%T"`
echo $now
mkdir ./log/$now
python ./main.py 2>&1 | tee ./log/$now/log.txt

# move files
if [ -e "loss.png" ]; then
    mv loss.png ./log/$now/
fi

deactivate