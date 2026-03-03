import copy
from core.monte_carlo import run_monte_carlo

def sensitivity_analysis(projects, employees, strategy, param_name, values):

    results = {}

    for value in values:
        config_override = {param_name: value}
        result = run_monte_carlo(
            projects,
            employees,
            strategy,
            runs=30,
            config_override=config_override
        )
        results[value] = result["average_profit"]

    return results