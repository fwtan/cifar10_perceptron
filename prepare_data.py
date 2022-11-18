import os
import os.path as osp
import numpy as np
import tarfile
from sacred import Experiment
from torchvision.datasets.utils import download_url


ex = Experiment('Prepare CIFAR10')


@ex.config 
def config():
    data_dir = osp.join('data', 'cifar-10-batches-py')
    data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    train_file = 'train.txt'
    val_file   = 'val.txt'
    test_file  = 'test.txt'
    num_train_samples = 49000
    num_val_samples   = 1000
    num_test_samples  = 10000



@ex.capture
def download_extract_cifar10(data_dir, data_url):
    download_url(data_url, root=osp.dirname(data_dir))
    filename = osp.join(osp.dirname(data_dir), osp.basename(data_url))
    with tarfile.open(filename, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=osp.dirname(data_dir))
    os.remove(filename)


@ex.capture
def generate_cifar10_splits(data_dir, train_file, val_file, test_file, num_train_samples, num_val_samples, num_test_samples):
    train_file = osp.join(data_dir, train_file)
    val_file   = osp.join(data_dir, val_file)
    test_file  = osp.join(data_dir, test_file)

    rand_inds = np.random.permutation(range(num_train_samples + num_val_samples))
    train_inds = [str(i) for i in sorted(rand_inds[:num_train_samples])]
    val_inds   = [str(i) for i in sorted(rand_inds[num_train_samples:])]
    test_inds  = [str(i) for i in range(num_test_samples)]

    with open(train_file, 'w') as f:
        f.write('\n'.join(train_inds))
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_inds))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_inds))


@ex.main
def prepare_cifar10():
    download_extract_cifar10()
    generate_cifar10_splits()


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    ex.run()
    
