# %%capture --no-stdout
from uqpce import PCE

samp_count = 200
aleat_cnt = 500000
epist_cnt = 1   #IY: If there are any epistemic variables, set it to n epistemic sample 

pce = PCE(
    order=2, verbose=True, outputs=True, plot=True, aleat_samp_size=aleat_cnt, 
    epist_samp_size=epist_cnt
)

# Add two normal variables
pce.add_variable(distribution='normal', mean=1, stdev=3, name='x0')
pce.add_variable(distribution='normal', mean=1, stdev=7, name='x1')

# Generate samples that correspond to the input variables
Xt = pce.sample(count=samp_count)

# Generate responses from equation; the user's analytical tool will replace this
eq = 'x0**2 + x0*x1 - x1'
yt = pce.generate_responses(Xt, equation=eq)

pce.fit(Xt, yt) # Fit the PCE model
pce.check_variables(Xt) # Check if the samples correspond to the distributions
pce.sobols() # Calculate the Sobol indices
cil, cih = pce.confidence_interval() # Calculate the confidence interval