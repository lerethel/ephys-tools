This is a script package I developed to analyze my electrophysiological data. It is tailored to my patch-clamp protocols, but you may find some useful general methods here, which you'll probably have to modify in accordance with your data.

The package includes scripts

- to count action potentials in a series of current steps
- to extract some commonly analyzed properties from the first action potential at the rheobase
- to extract I/V curves from a series of voltage steps (includes a basic filter to remove spikes from the traces)
- to reliably find action potentials
- to calculate the membrane resistance using linear regression
- to calculate the resting membrane potential from a 0-pA current step
- to calculate the sag ratio from a hyperpolarizing current step
- to calculate the membrane capacitance and time constant according to [Tamagnini et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25515596/)
