from hyparticlenet.hgnn.util import merge_splits

PATH = '/scratch/gc2c20/data/jet_tagging/train_bkg/'
output = '/scratch/gc2c20/data/jet_tagging/' 
merge_splits(full=True, include_custom=True, split_dir=PATH, output=output+'/train_bkg.hdf5',
        process_name='background')

PATH = '/scratch/gc2c20/data/jet_tagging/valid_bkg/'
merge_splits(full=True, include_custom=True, split_dir=PATH, output=output+'/valid_bkg.hdf5',
        process_name='background')

PATH = '/scratch/gc2c20/data/jet_tagging/test_bkg/'
merge_splits(full=True, include_custom=True, split_dir=PATH, output=output+'/test_bkg.hdf5',
        process_name='background')

