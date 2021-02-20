import os
import os.path as osp
import tarfile
from sacred import Experiment
from torchvision.datasets.utils import download_url


ex = Experiment('Prepare CIFAR10')


@ex.config 
def config():
    data_dir = osp.join('data')
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


@ex.main
def download_extract_cifar10(data_dir, data_url):
    download_url(data_url, root=data_dir)
    filename = osp.join(data_dir, osp.basename(data_url))
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    os.remove(filename)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    ex.run()
    
