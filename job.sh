#copying 
#python copying_task.py DRUM -H 100 -T 200 -norm 1.0 -GN Search-9-18
#python copying_task.py GRU -H 100 -T 200 -I 20000 -GN Search-9-18

#ptb character
python nlp_task.py DRUM -H 2000 -NL 1 -O Adam -B 128 -T 100 -R 0.001 -norm 3.0 -KP 0.9 -LN True -E 100 -GN Search-9-27


 