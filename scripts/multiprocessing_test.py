import graphicle as gcl
import numpy as np

def create_event(shower,process):
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
