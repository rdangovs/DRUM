#python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.001 -beta1 0.999 -E 40 -norm 1.0 -GN Search-9-11 #epoch # 30 
#python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.0001 -beta1 0.99 -E 100 -norm 1.2 -GN Search-9-11 #bad result
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.005 -beta1 0.999 -E 500 -norm 1.1 -GN Search-9-11
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.001 -beta1 0.99 -E 500 -norm 0.5 -GN Search-9-11
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.001 -beta1 0.999 -E 500 -norm 2.0 -GN Search-9-11
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.0001 -beta1 0.999 -E 500 -norm 0.9 -GN Search-9-11
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.0001 -beta1 0.9 -E 500 -norm 2.5 -GN Search-9-11
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.005 -beta1 0.9 -E 500 -norm 0.5 -GN Search-9-11
python nlp_task.py DRUM -H 2000 -O Adam -B 32 -R 0.005 -beta1 0.99 -E 500 -norm 0.8 -GN Search-9-11

