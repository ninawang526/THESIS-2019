# shell script to automate packaging and moving process

python3 cleanup.py

python -m zipfile -c $1 website/

#read -p "password: " PASS                                                                                                

#sshpass -p $PASS                                                                                                         
scp $1 ninaw@cycles.cs.princeton.edu:/n/fs/thesis-ninaw

rm -r website/

rm $1