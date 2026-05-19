import os
from beavrs.builder import BEAVRS

try:
    os.mkdir('build')
except OSError:
    pass
os.chdir('build')

model = BEAVRS()

model.write_openmc_geometry()
model.write_openmc_materials()
model.write_openmc_plots()
model.write_openmc_settings()
model.write_openmc_tallies()
model.write_openmc_model()
