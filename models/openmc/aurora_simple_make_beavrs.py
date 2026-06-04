import os
from beavrs.builder import BEAVRS

try:
    os.mkdir('aurora_kinetics_params')
except OSError:
    pass
os.chdir('aurora_kinetics_params')

model = BEAVRS()

model.write_openmc_geometry()
model.write_openmc_materials()
model.write_openmc_plots()
model.write_openmc_settings()
model.write_openmc_tallies()
model.write_openmc_model()
