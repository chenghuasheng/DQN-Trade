import os

for i in range(0, 4):
    start_str = str(i * 3 + 1).zfill(2)
    end_str = str(i * 3 + 3).zfill(2)
    begin_date = '2017-' + start_str + '-01'
    end_date = '2017-' + end_str + '-31'
    cmd = 'python trade_learning.py ' + begin_date + ' ' + end_date + ' -n 40 -i True'
    print(cmd)
    n = os.system(cmd)
    print(n)
