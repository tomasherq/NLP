#/bin/sh



i=0
length=50
while [ $i -le 1000 ]
do
  python3 createClickBait.py $length
  ((i++))
done