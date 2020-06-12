
N=10000
beta=1
gamma=0.18
p0=0
mu=0.02
sim='Fix'
repeats=500
python ../python_files/SIS_run_vaccination.py $N $beta $gamma $p0 $mu $sim $repeats
