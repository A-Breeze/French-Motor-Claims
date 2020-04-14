"""
Project configuration
Central store of variables that can be referenced throughout the project.
"""
#########
# Setup #
#########
# Import modules
import pyprojroot

# Project structure
root_dir_path = pyprojroot.here()
data_dir_path = root_dir_path / 'data'
