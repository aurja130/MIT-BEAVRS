# import os
# import math
import numpy as np
import openmc
import openmc.mgxs as mgxs
import pandas as pd

def run_temperature_sweep(temp_array):

    group_bounds = np.array([0.0, 0.625, 20e6])     # eV
    
    # Instantiate a 2-group EnergyGroups object
    two_groups = openmc.mgxs.EnergyGroups(group_edges=group_bounds)

    # # Instantiate a 1-group EnergyGroups object
    # one_group = openmc.mgxs.EnergyGroups(group_edges=np.array([group_bounds[0], group_bounds[-1]]))
    
    # Data storage
    pk_data = {
        'T': [],
        'reactivity': [],
        'LAMBDA': [],
        'total_beta': []
    }

    # Iteratively add beta_1 to beta_8 and lambda_1 to lambda_8
    for i in range(0, 8):
        pk_data[f'beta_{i}'] = []
    
    for j in range(0, 8):
        pk_data[f'lambda_{j}'] = []


    # Start the Solver Sweep Loop
    for T in temp_array:
        print(f"\n--- Running Core State: T_fuel = {T} K ---")

        # Load the model
        geom = openmc.Geometry.from_xml('geometry.xml')
        settings = openmc.Settings.from_xml('settings.xml')
        mats = openmc.Materials.from_xml('materials.xml')

        # Update fuel material temperatures
        for mat in mats:
            if 'Fuel' in mat.name:
                mat.temperature = T
        settings.temperature = {'method': 'interpolation'}

        # # Instantiate a tally mesh (for nodal calculations)                                                                                                                                                                  
        # mesh = openmc.RegularMesh(mesh_id=1)
        # mesh.dimension = [17, 17, 1]
        # mesh.lower_left = [-10.71, -10.71, -10000.]
        # mesh.width = [1.26, 1.26, 20000.]
        
        # 1. Re-initialize a clean MGXS Library for this specific iteration
        mgxs_lib = mgxs.Library(geom)
        mgxs_lib.energy_groups = two_groups
        mgxs_lib.num_delayed_groups = 8
        mgxs_lib.mgxs_types = ['nu-fission', 'inverse-velocity', 'delayed-nu-fission', 'decay-rate']
        
        # apply this library to the domain (for nodal calculations)
        # mgxs_lib.domain_type = 'mesh'
        # mgxs_lib.domains = [mesh]

        # apply this library to the domain
        mgxs_lib.domain_type = 'universe'
        mgxs_lib.domains = [geom.root_universe]
        mgxs_lib.build_library()

        # Create a "tallies.xml" file for the MGXS Library
        tallies = openmc.Tallies()
        mgxs_lib.add_to_tallies(tallies, merge=True)

        # # Instantiate a current tally (for nodal calculations)
        # mesh_filter = openmc.MeshSurfaceFilter(mesh)
        # current_tally = openmc.Tally(name='current tally')
        # current_tally.scores = ['current']
        # current_tally.filters = [mesh_filter]

        # # Add current tally to the tallies file
        # tallies.append(current_tally)
        
        # Run OpenMC
        model = openmc.Model(geometry=geom,
                            materials=mats,
                            settings=settings,
                            tallies=tallies)
        model.run(threads=8)
        
        # Extract Data
        sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
        mgxs_lib.load_from_statepoint(sp)
        k_eff_with_uncertainty = sp.k_combined

        sp.close()

        # =====================================================================
        # CALCULATE PRIMARY AND SECONDARY METRICS (2-GROUP ENERGETIC CONDENSATION)
        # =====================================================================
        
        # 1. Extract raw 2-group cross-sections, vectors, and fluxes
        # Lookups are structured as arrays across our 2 energy groups (0: Thermal, 1: Fast)
        
        flux    = mgxs_lib.get_mgxs(geom.root_universe, 'nu-fission').get_flux(value='mean') # fast, thermal (because default order is decreasing neutron energy)
        nu_fiss = mgxs_lib.get_mgxs(geom.root_universe, 'nu-fission').get_xs(value='mean')  # (2,)
        inv_v   = mgxs_lib.get_mgxs(geom.root_universe, 'inverse-velocity').get_xs(value='mean') # [1/v_fast, 1/v_thermal]
        
        # Pull the absolute delayed neutron production matrix (Shape: 8 precursor groups, 2 energy groups)
        delayed_nu_fiss = mgxs_lib.get_mgxs(geom.root_universe, 'delayed-nu-fission').get_xs(value='mean') # shape: (8 dnp groups, 2 ne groups)
        
        # Precursor decay constants do not depend on free neutron energy groups
        decay_const = mgxs_lib.get_mgxs(geom.root_universe, 'decay-rate').get_xs(value='mean') # Shape: (8 dnp groups,)

        # 2. Extract global eigenvalue parameters
        k_eff = k_eff_with_uncertainty.nominal_value
        reactivity = (k_eff - 1) / k_eff

        # 3. Perform the Flux-Weighting Condensation to 0D Point Kinetics Scalars
        # Total core-wide prompt neutron generation time (Total Density / Total Production)
        mean_generation_time = np.dot(inv_v, flux) / np.dot(nu_fiss, flux)  # scalar

        # Collapse the 2D delayed neutron matrix by summing the absolute energy production rates
        # axis=1 collapses the 2 energy group columns into a single flat vector of 8 elements
        group_wise_dn_production_rate = np.sum(delayed_nu_fiss * flux, axis=1) # Shape: (8,)
        total_neutron_production_rate = np.dot(nu_fiss, flux)
        beta = group_wise_dn_production_rate / total_neutron_production_rate             # Shape: (8,)

        # =====================================================================
        # APPEND TO MASTER TABLE
        # =====================================================================
        pk_data['T'].append(T)
        pk_data['reactivity'].append(reactivity)
        pk_data['LAMBDA'].append(mean_generation_time)
        pk_data['total_beta'].append(np.sum(beta))

        # Iteratively append each individual index scalar to its respective lists
        for i in range(0, 8):
            pk_data[f'beta_{i}'].append(beta[i])
        
        for j in range(0, 8):
            pk_data[f'lambda_{j}'].append(decay_const[j])
        
    return pk_data

if __name__ == "__main__":
    # Test a realistic Doppler sweep
    temperatures = [273.15, 300.0, 400.0, 500.0, 600.0,
                    700.0, 800.0, 900.0, 1000.0, 1100.0,
                    1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
                    1700.0, 1800.0, 1900.0, 2000.0, 2073.15]

    
    
    # temperatures = [273.15, 300.0, 400.0]

    results = run_temperature_sweep(temperatures)

    pd.DataFrame(results).to_csv('./pk_data/pk_params_2neg_8dnpg.csv', index=False)
    
    print("\nExtraction Complete! Data Ready for Curve Fitting.")
