# import os
# import math
import numpy as np
import openmc
import openmc.mgxs as mgxs
import pandas as pd

def run_temperature_sweep(temp_array):
    
    # Instantiate a 2-group EnergyGroups object
    group_bounds = np.array([0.0, 0.625, 20e6])     # eV
    two_groups = openmc.mgxs.EnergyGroups(group_edges=group_bounds)

    # Master list to store every node's data for every temperature
    nodal_data = []


    # Start the Solver Sweep Loop
    for T in temp_array:
        print(f"\n--- Running Core State: T_fuel = {T} K ---")

        # =====================================================================
        # LOAD MODEL COMPONENTS
        # =====================================================================
        geom = openmc.Geometry.from_xml('geometry.xml')
        settings = openmc.Settings.from_xml('settings.xml')
        mats = openmc.Materials.from_xml('materials.xml')

        # Update fuel material temperatures
        for mat in mats:
            if 'Fuel' in mat.name:
                mat.temperature = T
        settings.temperature = {'method': 'interpolation'}

        # =====================================================================
        # MESH SETUP
        # =====================================================================                                                                                                                                                             
        mesh = openmc.RegularMesh(mesh_id=1)
        mesh.dimension = [3, 3, 3]
        mesh.lower_left = [-170.0, -170.0, -182.88]
        mesh.upper_right = [170.0, 170.0, 182.88]
        
        # =====================================================================
        # MGXS LIBRARY SETUP
        # =====================================================================
        mgxs_lib = mgxs.Library(geom)
        mgxs_lib.energy_groups = two_groups
        mgxs_lib.num_delayed_groups = 8
        mgxs_lib.mgxs_types = [
            'diffusion-coefficient', 
            'absorption',            
            'nu-fission',            
            'kappa-fission',         
            'nu-scatter matrix',     
            'inverse-velocity',      
            'delayed-nu-fission',    
            'decay-rate'             
        ]
        
        # apply this library to the domain (for nodal calculations)
        mgxs_lib.domain_type = 'mesh'
        mgxs_lib.domains = [mesh]

        # # apply this library to the domain (for point kinetics calculations)
        # mgxs_lib.domain_type = 'universe'
        # mgxs_lib.domains = [geom.root_universe]
        
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
        
        # =====================================================================
        # RUN OPENMC
        # =====================================================================
        model = openmc.Model(geometry=geom,
                            materials=mats,
                            settings=settings,
                            tallies=tallies)
        model.run(threads=8)

        # =====================================================================
        # EXTRACT SPATIAL NODE DATA
        # =====================================================================
        sp = openmc.StatePoint(f'statepoint.{settings.batches}.h5')
        mgxs_lib.load_from_statepoint(sp)

        # The .get_xs() function for a mesh returns an array of shape (N_nodes, N_groups)
        diff_coeff      = mgxs_lib.get_mgxs(mesh, 'diffusion-coefficient').get_xs(value='mean') 
        abs_xs       = mgxs_lib.get_mgxs(mesh, 'absorption').get_xs(value='mean') 
        nu_fiss      = mgxs_lib.get_mgxs(mesh, 'nu-fission').get_xs(value='mean') 
        kappa_fiss   = mgxs_lib.get_mgxs(mesh, 'kappa-fission').get_xs(value='mean') 
        scatt_mat    = mgxs_lib.get_mgxs(mesh, 'nu-scatter matrix').get_xs(value='mean') 
        inv_v        = mgxs_lib.get_mgxs(mesh, 'inverse-velocity').get_xs(value='mean') 
        del_nu_fiss  = mgxs_lib.get_mgxs(mesh, 'delayed-nu-fission').get_xs(value='mean') 
        decay        = mgxs_lib.get_mgxs(mesh, 'decay-rate').get_xs(value='mean') 
        
        num_nodes = 3 * 3 * 3
        
        for k in range(num_nodes):
            node_dict = {
                'T': T,
                'node_id': k,
                'D1': diff_coeff[k, 0],
                'D2': diff_coeff[k, 1],
                'Sigma_a1': abs_xs[k, 0],
                'Sigma_a2': abs_xs[k, 1],
                'nu_Sigma_f1': nu_fiss[k, 0],
                'nu_Sigma_f2': nu_fiss[k, 1],
                'kappa_Sigma_f1': kappa_fiss[k, 0],
                'kappa_Sigma_f2': kappa_fiss[k, 1],
                # scatt_mat shape is (N, g_in, g_out). Group 0 is Fast, Group 1 is Thermal.
                'Sigma_s12': scatt_mat[k, 0, 1], 
                'inv_v1': inv_v[k, 0],
                'inv_v2': inv_v[k, 1]
            }
            
            # Calculate Node-Wise Delayed Fractions (beta_i)
            total_prod = nu_fiss[k, 0] + nu_fiss[k, 1]
            
            for i in range(8):
                if total_prod > 0:
                    delayed_prod = del_nu_fiss[k, i, 0] + del_nu_fiss[k, i, 1]
                    node_dict[f'beta_{i+1}'] = delayed_prod / total_prod
                else:
                    node_dict[f'beta_{i+1}'] = 0.0
                    
                node_dict[f'lambda_{i+1}'] = decay[k, i]
                
            nodal_data.append(node_dict)

        sp.close()
        
    return nodal_data

if __name__ == "__main__":
    # Test a realistic Doppler sweep
    # temperatures = [273.15, 300.0, 400.0, 500.0, 600.0,
    #                 700.0, 800.0, 900.0, 1000.0, 1100.0,
    #                 1200.0, 1300.0, 1400.0, 1500.0, 1600.0,
    #                 1700.0, 1800.0, 1900.0, 2000.0, 2073.15]

    temperatures = [600.0, 900.0, 1200.0]

    results = run_temperature_sweep(temperatures)

    pd.DataFrame(results).to_csv('./nk_data/nk_params_2neg_8dnpg.csv', index=False)
    
    print("\nExtraction Complete! Data Ready for Curve Fitting.")
