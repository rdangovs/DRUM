#copying 
python copying_task.py DRUM -H 100 -T 200 -norm 1.0 -GN Search-9-18
python copying_task.py GRU -H 100 -T 200 -I 20000 -GN Search-9-18

#ptb character
#python nlp_task.py GRU -H 1000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 100 -GN Search-9-18
python nlp_task.py DRUM -H 1000 -NL 2 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 100 -norm 0.5 -GN Search-9-18

python nlp_task.py DRUM -H 2000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 100 -norm 0.5 -GN Search-9-18
python nlp_task.py DRUM -H 2000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 100 -norm 0.4 -GN Search-9-18
python nlp_task.py DRUM -H 2000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 100 -norm 0.3 -GN Search-9-18
python nlp_task.py DRUM -H 2000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.9 -E 100 -norm 0.5 -GN Search-9-18
python nlp_task.py DRUM -H 2000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.9 -E 100 -norm 0.4 -GN Search-9-18
python nlp_task.py DRUM -H 2000 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.9 -E 100 -norm 0.3 -GN Search-9-18

python nlp_task.py DRUM -H 1500 -NL 2 -RD 0.8 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 100 -norm 0.5 -GN Search-9-18




