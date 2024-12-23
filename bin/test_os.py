# Add at the top of the file
import os
from load_opus import DATA_ROOT
# Define DATA_ROOT relative to the current file location
# DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
print(DATA_ROOT)
# Alternative if you want to specify absolute path:
# DATA_ROOT = r'C:\Users\Mattia\Documents\GitHub\translator\data'