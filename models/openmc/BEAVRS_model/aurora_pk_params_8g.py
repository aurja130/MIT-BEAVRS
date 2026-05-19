# import os
import math
import numpy as np
import openmc
import openmc.mgxs as mgxs
import pandas as pd

def run_temperature_sweep(temp_array):
    # Load base geometry and settings once
    geom = openmc.Geometry.from_xml('geometry.xml')
    settings = openmc.Settings.from_xml('settings.xml')

    group_bounds = np.array([0.0, 6.25e-7, 1e37])
    
    # Instantiate a 2-group EnergyGroups object
    # two_groups = openmc.mgxs.EnergyGroups(group_edges=group_bounds)

    # Instantiate a 1-group EnergyGroups object
    one_group = openmc.mgxs.EnergyGroups(group_edges=np.array([group_bounds[0], group_bounds[-1]]))
    
    # Data storage
    pk_data = {
        'T': [],
        'nu_Sigma_f': [],
        'v': [],
        'LAMBDA': [],
        'k_eff': [],
        'reactivity': [],
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

        # # Instantiate a tally mesh (for nodal calculations)                                                                                                                                                                  
        # mesh = openmc.RegularMesh(mesh_id=1)
        # mesh.dimension = [17, 17, 1]
        # mesh.lower_left = [-10.71, -10.71, -10000.]
        # mesh.width = [1.26, 1.26, 20000.]
        
        # 1. Re-initialize a clean MGXS Library for this specific iteration
        mgxs_lib = mgxs.Library(geom)
        mgxs_lib.energy_groups = one_group
        mgxs_lib.num_delayed_groups = 8
        mgxs_lib.mgxs_types = ['transport', 'absorption', 'scatter', 'nu-fission', 'inverse-velocity',
                               'beta', 'decay-rate']
        
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
        
        # 2. Load and update fuel material temperatures
        mats = openmc.Materials.from_xml('materials.xml')
        for mat in mats:
            if 'Fuel' in mat.name:
                mat.temperature = T
        mats.export_to_xml('materials.xml')
        
        settings.temperature = {'method': 'interpolation'}
        settings.export_to_xml('settings.xml')
        
        # Run OpenMC
        model = openmc.Model(geometry=geom,
                            materials=mats,
                            settings=settings,
                            tallies=tallies)
        model.run(threads=8)
        
        # 4. Extract Data
        sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
        mgxs_lib.load_from_statepoint(sp)
        k_eff_with_uncertainty = sp.k_combined

        sp.close()

        # calculate primary and secondary metrics

        beta = mgxs_lib.get_mgxs(geom.root_universe, 'beta').get_xs().reshape(mgxs_lib.num_delayed_groups)
        decay_const = mgxs_lib.get_mgxs(geom.root_universe, 'decay-rate').get_xs()

        k_eff = k_eff_with_uncertainty.nominal_value

        reactivity = (k_eff - 1) / k_eff
        
        # Standard Cross Sections
        nu_Sigma_f = mgxs_lib.get_mgxs(geom.root_universe, 'nu-fission').get_xs()[0]
        inverse_v = mgxs_lib.get_mgxs(geom.root_universe, 'inverse-velocity').get_xs()[0]
        v = 1 / inverse_v
        mean_generation_time = 1 / v / nu_Sigma_f

        # Append to Master Table
        pk_data['T'].append(T)
        pk_data['nu_Sigma_f'].append(nu_Sigma_f)
        pk_data['v'].append(v)
        pk_data['LAMBDA'].append(mean_generation_time)
        pk_data['k_eff'].append(k_eff)
        pk_data['reactivity'].append(reactivity)

        # Iteratively add beta_1 to beta_8 and lambda_1 to lambda_8
        for i in range(0, 8):
            pk_data[f'beta_{i}'] = beta[i]
        
        for j in range(0, 8):
            pk_data[f'lambda_{j}'] = decay_const[j]
        
        pk_data['total_beta'] = np.sum(beta)
        
    return pk_data

if __name__ == "__main__":
    # Test a realistic Doppler sweep
    temperatures = [273.15, 300.0, 400.0, 500.0, 600.0,
                    700.0, 800.0, 900.0, 1000.0, 1100.0,
                    1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
                    1700.0, 1800.0, 1900.0, 2000.0, 2073.15]

    results = run_temperature_sweep(temperatures)

    pd.DataFrame(results).to_csv('./pk_data/pk_params_8g.csv', index=False)
    
    print("\nExtraction Complete! Data Ready for Curve Fitting.")
