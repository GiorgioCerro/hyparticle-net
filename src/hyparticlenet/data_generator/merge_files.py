from hyparticlenet.hgnn.util import merge_splits

PATH = '/scratch/gc2c20/data/jet_tagging/train_sig/'
output = '/scratch/gc2c20/data/jet_tagging/'
merge_splits(full=True, include_custom=True, split_dir=PATH, output=output+'/train_sig.hdf5',
        process_name='signal')

PATH = '/scratch/gc2c20/data/jet_tagging/valid_sig/'
merge_splits(full=True, include_custom=True, split_dir=PATH, output=output+'/valid_sig.hdf5',
        process_name='signal')

PATH = '/scratch/gc2c20/data/jet_tagging/test_sig/'
merge_splits(full=True, include_custom=True, split_dir=PATH, output=output+'/test_sig.hdf5',
        process_name='signal')


