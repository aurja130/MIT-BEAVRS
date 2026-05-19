# import os
# import numpy as np
import openmc
import openmc.mgxs as mgxs
import pandas as pd

def run_temperature_sweep(temp_array):
    # Load base geometry and settings once
    geom = openmc.Geometry.from_xml('geometry.xml')
    settings = openmc.Settings.from_xml('settings.xml')
    
    # Define groups once
    groups = mgxs.EnergyGroups([0.0, 20.0e6]) # 1 group for point-kinetics
    
    # Data storage
    th_data = {
        'T_fuel': [],
        'nu_Sigma_f1': [],
        'v': [],
        'mean_generation_time': [],
        'k_eff': [],
        'reactivity': []
    }

    # Start the Solver Sweep Loop
    for T in temp_array:
        print(f"\n--- Running Core State: T_fuel = {T} K ---")
        
        # 1. Re-initialize a clean MGXS Library for this specific iteration
        mgxs_lib = mgxs.Library(geom)
        mgxs_lib.energy_groups = groups
        mgxs_lib.domain_type = 'universe'
        mgxs_lib.domains = [geom.root_universe]
        mgxs_lib.mgxs_types = ['nu-fission', 'inverse-velocity']
        mgxs_lib.build_library() 
        
        # 2. Load and update fuel material temperatures
        mats = openmc.Materials.from_xml('materials.xml')
        for mat in mats:
            if 'Fuel' in mat.name:
                mat.temperature = T
        mats.export_to_xml('materials.xml')
        
        settings.temperature = {'method': 'interpolation'}
        settings.export_to_xml('settings.xml')
        
        # 3. Build tallies container
        tallies_file = openmc.Tallies()
        mgxs_lib.add_to_tallies(tallies_file)
        tallies_file.export_to_xml('tallies.xml')
        
        # Run OpenMC
        openmc.run(threads=8)
        
        # 4. Extract Data
        sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
        mgxs_lib.load_from_statepoint(sp)
        k_eff_with_uncertainty = sp.k_combined
        k_eff = k_eff_with_uncertainty.nominal_value

        reactivity = (k_eff - 1) / k_eff
        
        # Standard Cross Sections
        nu_Sigma_f1 = mgxs_lib.get_mgxs(geom.root_universe, 'nu-fission').get_xs()[0]
        inverse_v1 = mgxs_lib.get_mgxs(geom.root_universe, 'inverse-velocity').get_xs()[0]
        v1 = 1 / inverse_v1
        mean_generation_time1 = 1 / v1 / nu_Sigma_f1

        # Append to Master Table
        th_data['T_fuel'].append(T)
        th_data['nu_Sigma_f1'].append(nu_Sigma_f1)
        th_data['v'].append(v1)
        th_data['mean_generation_time'] = mean_generation_time1
        th_data['k_eff'].append(round(k_eff, 5))
        th_data['reactivity'].append(round(reactivity, 5))
        sp.close()
        
    return th_data

if __name__ == "__main__":
    # Test a realistic Doppler sweep
    # temperatures = [273.15, 300.0, 400.0, 500.0, 600.0,
    #                 700.0, 800.0, 900.0, 1000.0, 1100.0,
    #                 1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
    #                 1700.0, 1800.0, 1900.0, 2000.0, 2073.15]

    temperatures = [273.15, 700.0, 1200.0]
    results = run_temperature_sweep(temperatures)

    pd.DataFrame(results).to_csv('./pk_data/pk_params.csv', index=False)
    
    print("\nExtraction Complete! Data Ready for Curve Fitting.")
