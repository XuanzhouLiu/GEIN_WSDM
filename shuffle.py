import os
import sys
import random

import tempfile
from subprocess import call
import numpy as np
from pathlib import Path

def main(file_path, temporary=False, seed = 0):
    rng = np.random.default_rng(seed = seed)

    if temporary:
        path, filename = os.path.split(os.path.realpath(file_path))
        fd = tempfile.TemporaryFile(prefix=filename + '.shuf' + str(seed), dir=path)
    else:
        if os.path.exists(file_path+ '.shuf' + str(seed)):
            fd = open(file_path + '.shuf' + str(seed), 'r')
            return fd
        else:
            fd = open(file_path + '.shuf' + str(seed), 'w+')

    lines = open(file_path, 'r').readlines()
    rng.shuffle(lines)

    for l in lines:
        s = l.strip("\n")
        print(s, file=fd)

    fd.seek(0)

    return fd


if __name__ == '__main__':
    main(sys.argv[1])
