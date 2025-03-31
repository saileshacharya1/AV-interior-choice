# ---------------------------------------------------------------------------------------------------------------------#
# import modules
from scipy.stats import chi2
import logging

# configure logging
logging.basicConfig(
    filename='../outputs/4-5-lr-test.log',  
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s'  
)
logging.info("Script execution started.")

# ---------------------------------------------------------------------------------------------------------------------#
# likelihood ratio test between constrained (4-4-mxl-IV-final-lr-test) and unconstrained (4-2-mxl-IV-final) models

# number of parameters and log likelihood statistics of 
# constrained (4-4-mxl-IV-final-lr-test) and unconstrained (4-2-mxl-IV-final) models
num_params_constrained = 28
num_params_unconstrained = 30
ll_constrained = -2242.22
ll_unconstrained = -2265.24


# difference in number of parameters between the two models
num_params_difference = num_params_unconstrained - num_params_constrained

# likelihood ratio test statistic
lr_statistic = -2 * ((ll_unconstrained) - (ll_constrained))

# p-value from the Chi-squared distribution
p_value = 1 - chi2.cdf(lr_statistic, num_params_difference)

# print
logging.info(f"Likelihood Ratio Statistic: {lr_statistic}")
logging.info(f"p-value: {p_value}")
# ---------------------------------------------------------------------------------------------------------------------#
