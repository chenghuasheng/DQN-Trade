import os

for i in range(9, 11):
    i_str = str(i).zfill(2)
    begin_date = '2018-' + i_str + '-01'
    end_date = '2018-' + i_str + '-31'
    cmd = 'python trade_learning.py ' + begin_date + ' ' + end_date + ' -n 80 -i True'
    print(cmd)
    n = os.system(cmd)
    print(n)
