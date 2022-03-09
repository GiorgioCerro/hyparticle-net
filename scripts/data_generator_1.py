import graphicle as gcl
from heparchy.write.hdf import HdfWriter
from heparchy.read.hepmc import HepMC
import numpy as np

import sys
sys.path.append('/mainfs/home/gc2c20/myproject/hyperTree/')

from embedding import HyperEmbedding
from tqdm import tqdm

def generate(path_hdf,process_name,path_hepmc):
    with HdfWriter(path_hdf) as hep_file:
        with hep_file.new_process(process_name) as process:
            with HepMC(path_hepmc) as raw_file:
                for shower in tqdm(raw_file):
                    with process.new_event() as event:
                        event.set_edges(shower.edges)
                        event.set_pmu(shower.pmu)
                        event.set_pdg(shower.pdg)
                        event.set_mask(name='final',data=shower.final)
                        graph = gcl.Graphicle.from_numpy(
                            edges = shower.edges,
                            pmu = shower.pmu,
                            pdg = shower.pdg,
                            final = shower.final,
                        )
                        graph.adj = gcl.transform.particle_as_node(graph.adj)
                        
                        hyper_coords = np.NaN
                        while np.isnan(hyper_coords).sum() > 0:
                            hyper = HyperEmbedding(graph)
                            hyper.get_embedding()
                            hyper_coords = hyper.embeddings

                        event.set_custom_dataset(
                            name='hyper_coords',data=hyper_coords,
                            dtype='double')


path_hdf = '../data/hz_test.hdf5'
path_hepmc = '../data/hz_test.hepmc'
process_name = 'signal'
generate(path_hdf,process_name,path_hepmc)
