"""ais_plasticity: AIS plasticity demand across real somatodendritic morphologies.

Modules
-------
swc             : SWC parsing into a compartment tree.
dendritic_load  : frequency-domain passive solution -> somatic input conductance, effective
                  capacitance at spike frequencies, transfer impedance to a virtual AIS.
ais_theory      : resistive-coupling threshold model (Goethals & Brette style), axial resistance,
                  plasticity-demand solver.
neuron_builder  : builds a NEURON model (lazy import) with a synthetic hillock/AIS/axon attached.
"""

__all__ = ["swc", "dendritic_load", "ais_theory", "neuron_builder"]
__version__ = "0.1.0"
