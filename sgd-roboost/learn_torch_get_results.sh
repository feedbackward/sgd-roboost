#!/etc/bash

## A simple script to fetch lots of files.

IPADD="192.168.1.100"
MODEL="FF_L2"

for arg
do scp mjh@"${IPADD}:/home/mjh/20200829_roboost/results/torch/${arg}/*${MODEL}*" "./results/torch/${arg}/"
done



