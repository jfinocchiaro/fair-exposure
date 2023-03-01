# fair exposure under homophily

## Dependencies
- Install dependencies by running pip install -r requirements.txt

***

## src
- plartform_opt: optimization to find theta_policy for fairness-aware and agnostic problems.
- players: has player class and some of the functionality of agents
- sims: File specifies how to run and store simulation results.
- sims_copy: some redundancy here with sims.  same functionality, but different form of outputting datatypes

***

## notebooks
- generatre_runs.ipynb: runs and saves simulations according to given parameters and generates some of the plots for analysis
- gen_plots.ipynb: generates more figures for analysis, comparing inter- and intra-group exposure and click rates.
- probability_sharing_distributions.ipynb: estimates empirical model parameters by MLE for Bakshy et al.
- Replication Exposure.ipynb: estimates empirical model parameters by MLE for Garimella et al.


***


## data
- model parameters for different datasets (listed) are estimated by MLE in notesbooks/probability_sharing_distributions.ipynb







