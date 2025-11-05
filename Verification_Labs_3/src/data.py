#
# Data & Setup Functions
#


# Project Run Setup
#

from datetime import datetime
from pathlib import Path

def run_setup(local_project_folder, run_name=''):

    # Folder paths
    data_folder = local_project_folder.joinpath('data')
    if not data_folder.exists():
        raise FileNotFoundError(f'{data_folder} does not exist')
    results_folder = local_project_folder.joinpath('run_results')
    if not results_folder.exists():
        raise FileNotFoundError(f'{results_folder} does not exist')

    # Run Results
    run_name = f'{run_name.strip()}'
    run_name = f'Run_{run_name}' if (run_name) else f'Run_{datetime.now().strftime("%Y%m%d")}'
    run_results_folder = results_folder.joinpath(f'{run_name}')
    run_results_folder.mkdir(parents=True, exist_ok=True) 

    return data_folder, run_results_folder