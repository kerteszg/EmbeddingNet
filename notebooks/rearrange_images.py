import os
import glob
import shutil
from random import shuffle

# TODO !!!
dirs = [1,2,3,4,5]
# !!!

data = []
idx = 0

keves = 0

for d in dirs:
    dira = 'frames\\{}A'.format(d)
    dirb = 'frames\\{}B'.format(d)

    lst = [line.rstrip('\n') for line in open('e:\\{}AB_pairs.txt'.format(d))]
    lst = [line.split('\t') for line in lst]

    print('loading pairs')
    for i in range(len(lst)):
        curr_data = []
        dest_dir = '{}'.format(idx)
        for file in glob.glob(r'{}\{}_*.jpg'.format(dira, lst[i][0])):
            if os.path.getsize(file) > 0:
                curr_data.append((file, '{}\\A_{}'.format(dest_dir, os.path.basename(file))))
        for file in glob.glob(r'{}\{}_*.jpg'.format(dirb, lst[i][1])):
            if os.path.getsize(file) > 0:
                curr_data.append((file, '{}\\B_{}'.format(dest_dir, os.path.basename(file))))

        if len(curr_data) >= 30:
            shuffle(curr_data)
            data.append(curr_data[:30])
            idx += 1
        else:
            keves += 1
    print(d, 'done:', len(data), 'observation total')

print('not enough frames', keves, 'from a total of', len(data), 'observations')

if not os.path.exists('CVPR2016'):
    os.makedirs('CVPR2016')

shuffle(data)
val_num = int(len(data)/10)
train = data[val_num:]
val = data[:val_num]

for pairs in train:
    for (fromm, to) in pairs:
        if not os.path.exists(os.path.dirname('CVPR2016\\train\\{}'.format(to))):
            os.makedirs(os.path.dirname('CVPR2016\\train\\{}'.format(to)))
        shutil.copy(fromm, 'CVPR2016\\train\\{}'.format(to))

print('train done')

for pairs in val:
    for (fromm, to) in pairs:
        if not os.path.exists(os.path.dirname('CVPR2016\\eval\\{}'.format(to))):
            os.makedirs(os.path.dirname('CVPR2016\\eval\\{}'.format(to)))
        shutil.copy(fromm, 'CVPR2016\\eval\\{}'.format(to))
    #print()

print('eval done')