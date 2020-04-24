from keras.models import load_model
from embedding_net.model import EmbeddingNet
from embedding_net.utils import parse_net_params
from embedding_net.data_loader import EmbeddingNetImageLoader
from embedding_net.utils import plot_tsne_interactive, plot_tsne
import albumentations as A
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import time
import numpy as np
import os

N_way = 10
tests = 1000
configs = []
results = {}

for file in os.listdir("configs"):
    #if file.endswith(".yml") and "nist" in filename:       #example on filtering
    if file.endswith(".yml"):
        configs.append(file)

configs.sort()  #alphabetical order

for file in configs:
    print('going to test:', file)

for filename in configs:
    config_name = filename[:-4]
    if 'siamese' in config_name:
        mode = 'siamese'
    else:
        mode = 'triplet'

    cfg_params = parse_net_params('configs/{}.yml'.format(config_name), verboseOverride=True, verbose=False)
    model = EmbeddingNet(cfg_params, training = False)
    print('Loading model ...')
    #nasnet should be compiled after loading
    model.load_model('work_dirs/{}/weights/best_model_{}.h5'.format(config_name, config_name), mode=mode, _compile=False)
    print('Loading encodings ...')
    model.load_encodings('work_dirs/{}/encodings/encodings_{}.pkl'.format(config_name, config_name))

    ways = np.arange(1, N_way + 1)
    for subset in ['train', 'val']:
        print('measuring one-shot accuracy of ', filename, 'for set:', subset)
        X = model.prepare_sample_dict(subset)
        oneshotaccs = []
        for N in ways:
            acc = model.test_oneshot(N, tests, X, s=subset)
            oneshotaccs.append(acc)
        results[config_name + '_' + subset + 'acc'] = oneshotaccs

print('finished')
print(results)
for c in results:
    print(c, results[c])