from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.physics.geothermal import Geothermal
from darts.models.darts_model import DartsModel, sim_params
from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, index_vector

from darts.engines import operator_set_evaluator_iface

import math
import numpy as np
import pandas as pd
from darts.engines import rate_prod_well_control, rate_inj_well_control
from darts.models.opt.opt_module_settings import OptModuleSettings
from darts.engines import value_vector, index_vector

# Example of a reservoir model file. Please note that this file has not been refactored.
# Existing DARTS reservoirs models can be used, as long as they are adapted to take a parameter and parameter_type
# variable as the porosity or permeability model parameters.

class Model(DartsModel, OptModuleSettings):
    def __init__(self, parameter, parameter_type, report_step=120, final_time=360*100, customize_new_operator=False,
                 Peaceman_WI=False):
        # call base class constructor
        super().__init__()
        OptModuleSettings.__init__(self)

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.realization = 0
        self.T = final_time
        #self.file_path = file_path
        self.report_step = report_step
        self.customize_new_operator = customize_new_operator

        # initialize global data to record the well location in vtk output file
        self.global_data = {'well location': 0}  # will be updated later in "run"

        self.nx = 140
        self.ny = 140
        self.nz = 1

        self.dx = 20
        self.dy = 20
        self.dz = 100

        # #
        # if parameter_type == 0:
        #     parameter[parameter >= 0.30] = 0.30
        #     parameter[(0.15 <= parameter) & (parameter < 0.3)] = 0.15
        #     parameter[parameter < 0.15] = 0.15
        # #

        # Check the parameter type and calculate the other parameter based on it.
        if parameter_type == 0:
            self.poro = parameter
            self.kx = 196449 * self.poro ** 4.3762
        elif parameter_type == 1:
            self.kx = parameter
            self.poro = np.power(self.kx * 1 / 196449, 1 / 4.3762)
        else:
            self.poro = parameter
            self.kx = 196449 * self.poro ** 4.3762

        # nx = int(np.sqrt(self.kx.size))
        #self.kx = self.kx.reshape(self.ny, self.nx).T.flatten()
        #self.poro = self.poro.reshape(self.ny, self.nx).T.flatten()
        self.kx = self.kx.reshape(self.nx, self.ny).flatten()
        self.poro = self.poro.reshape(self.nx, self.ny).flatten()
        self.kx[self.kx < 1.E-6] = 1.E-6
        self.poro[self.poro < 1.E-4] = 1.E-4

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dx,
                                         dz=self.dz, permx=self.kx, permy=self.kx, permz=self.kx*0.1, poro=self.poro,
                                         depth=2000)

        # self.reservoir.set_boundary_volume(xy_minus=30*30*400, xy_plus=30*30*400)
        # self.reservoir.set_boundary_volume(xy_minus=30 * 30 * 50, xy_plus=30 * 30 * 50)  # xy plane: overburden, underburden
        self.reservoir.set_boundary_volume(xz_minus=30*30*100000, xz_plus=30*30*100000)    # xz plane:


        # self.inj_BHP = [175, 175, 180, 180, 180]  # bar
        # self.prod_BHP = [163, 163, 150, 150, 165]
        #
        # self.total_rate = 36000*2  # m3/day
        # self.doublet_rate = np.array([3/36, 7/36, 10/36, 10/36, 6/36]) * self.total_rate
        # # self.total_rate = 3600  # m3/day
        # self.temp_reinj_w = 308.15  # the temperature of reinjected water
        # np.save('_temp_reinj_w.npy', np.array(self.temp_reinj_w))  # save it for post_process_Brugge.py
        #
        # self.inj_list = [[27, 26], [47, 26], [67, 26], [87, 26], [107, 26]]
        # self.prod_list = [[27, 32], [47, 32], [67, 32], [87, 32], [107, 32]]


        self.inj_BHP = [300]  # bar
        self.prod_BHP = [50]

        self.total_rate = 5000  # m3/day
        self.doublet_rate = np.array([1]) * self.total_rate

        self.temp_reinj_w = 308.15  # the temperature of reinjected water
        np.save('_temp_reinj_w.npy', np.array(self.temp_reinj_w))  # save it for post process

        self.init_tempr = 348.15
        np.save('_temp_init.npy', np.array(self.init_tempr))  # save it for post process

        # self.inj_list = [[30, 14]]
        # self.prod_list = [[30, 46]]
        self.inj_list = [[40, 70]]
        self.prod_list = [[100, 70]]

        # well index setting
        if Peaceman_WI:
            WI = -1  # use Peaceman function; check the function "add_perforation" for more details
        else:
            WI = 200

        n_perf = self.reservoir.nz
        perf_list = list(range(n_perf))
        # perf_list = np.array([3]).astype('int')
        for i, inj in enumerate(self.inj_list):
            self.reservoir.add_well('I' + str(i+1))
            for n in perf_list:
                self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=inj[0], j=inj[1], k=n+1, well_radius=0.1,
                                               well_index=WI, multi_segment=False, verbose=True)

        for p, prod in enumerate(self.prod_list):
            self.reservoir.add_well('P' + str(p+1))
            for n in perf_list:
                self.reservoir.add_perforation(self.reservoir.wells[-1], i=prod[0], j=prod[1], k=n+1, well_radius=0.1,
                                               well_index=WI, multi_segment=False, verbose=True)


        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)  # volumetric heat capacity: kJ/m3/K
        rcond.fill(181.44)  # kJ/m/day/K

        self.min_p = 1
        # self.max_p = 351
        self.max_p = 900
        self.physics = Geothermal(self.timer, n_points=128, min_p=self.min_p, max_p=self.max_p, min_e=1000, max_e=10000, cache=False)
        # temp_rock_pro = np.array(self.physics.property_data.rock, copy=False)
        # temp_rock_pro[0][1] = 1e-5

        self.params.first_ts = 1e-3
        self.params.mult_ts = 8
        self.params.max_ts = 365

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 20
        self.params.max_i_linear = 40

        self.params.newton_type = sim_params.newton_global_chop
        self.params.newton_params = value_vector([1])

        # self.physics.engine.silent_mode = 0
        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                    uniform_temperature=348.15)

    def set_boundary_conditions(self):
        # for i, w in enumerate(self.reservoir.wells):
        #     if 'I' in w.name:
        #         w.control = self.physics.new_bhp_water_inj(300, 308.15)
        #     else:
        #         w.control = self.physics.new_bhp_prod(50)


        idx_inj = 0
        idx_prod = 0
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                w.control = self.physics.new_rate_water_inj(self.doublet_rate[idx_inj], self.temp_reinj_w)
                # w.constraint = self.physics.new_bhp_water_inj(self.max_p - 100, self.temp_reinj_w)
                idx_inj += 1
            else:
                w.control = self.physics.new_rate_water_prod(self.doublet_rate[idx_prod])
                # w.constraint = self.physics.new_bhp_prod(self.min_p + 100)
                idx_prod += 1

# -----------------------------------Adjoint method---------------------------------------------
    def run(self, export_to_vtk=False, file_name='data'):
        import random
        # use seed to generate the same random values every run
        random.seed(3)
        np.random.seed(3)
        if export_to_vtk:

            well_loc = np.zeros(self.reservoir.n)
            for inj in self.inj_list:
                well_loc[(inj[1] - 1) * self.reservoir.nx + inj[0] - 1] = -1

            for prod in self.prod_list:
                well_loc[(prod[1] - 1) * self.reservoir.nx + prod[0] - 1] = 1

            self.global_data = {'well location': well_loc}

            self.export_pro_vtk(global_cell_data=self.global_data, file_name=file_name)


        # now we start to run for the time report--------------------------------------------------------------
        time_step = self.report_step
        even_end = int(self.T / time_step) * time_step
        time_step_arr = np.ones(int(self.T / time_step)) * time_step
        if self.T - even_end > 0:
            time_step_arr = np.append(time_step_arr, self.T - even_end)

        for ts in time_step_arr:
            # for i, w in enumerate(self.reservoir.wells):
            #     if 'I' in w.name:
            #         w.control = self.physics.new_bhp_water_inj(300, 308.15)
            #     else:
            #         w.control = self.physics.new_bhp_prod(50)


            # print("Running time: %d days" % ts)
            idx_inj = 0
            idx_prod = 0
            # perturbation = random.uniform(1, self.total_rate / 4)
            perturbation = np.random.uniform(-1, 1) * 0.1 * self.doublet_rate[0]
            for i, w in enumerate(self.reservoir.wells):
                if 'I' in w.name:
                    w.control = self.physics.new_rate_water_inj(self.doublet_rate[idx_inj] + perturbation,
                                                                self.temp_reinj_w)
                    # w.constraint = self.physics.new_bhp_water_inj(self.max_p - 100, self.temp_reinj_w)
                    idx_inj += 1
                else:
                    w.control = self.physics.new_rate_water_prod(self.doublet_rate[idx_prod] + perturbation)
                    # w.constraint = self.physics.new_bhp_prod(self.min_p + 100)
                    idx_prod += 1

            self.physics.engine.run(ts)
            self.physics.engine.report()

            if export_to_vtk:
                self.export_pro_vtk(global_cell_data=self.global_data, file_name=file_name)

    def set_op_list(self):
        if self.customize_new_operator:
            # customize your own operator, e.g. the Temparature
            temperature_etor = geothermal_customized_etor(self.physics.property_data)

            temperature_itor = self.physics.create_interpolator(temperature_etor, self.physics.n_vars, 1,
                                                                self.physics.n_axes_points, self.physics.n_axes_min,
                                                                self.physics.n_axes_max,
                                                                platform='cpu', algorithm='multilinear',
                                                                mode='adaptive', precision='d')
            self.physics.create_itor_timers(temperature_itor, "customized operator interpolation")

            self.physics.engine.customize_operator = self.customize_new_operator

            # set operator list
            self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_itor_well, temperature_itor]

            # specify the index of blocks of well operator
            op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            op_num[self.reservoir.mesh.n_res_blocks:] = 1

            # specify the index of blocks of customized operator
            op_num_new = np.array(self.reservoir.mesh.op_num, copy=True)
            op_num_new[:] = 2  # set the third interpolator (i.e. "temperature_itor") from "self.op_list" to all blocks
            self.physics.engine.idx_customized_operator = 2
            self.physics.engine.customize_op_num = index_vector(op_num_new)
        else:
            self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_itor_well]
            op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            op_num[self.reservoir.mesh.n_res_blocks:] = 1

    def compute_temperature(self, X):
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def export_pro_vtk(self, file_name='Results', global_cell_data={}):
        nb = self.reservoir.mesh.n_res_blocks
        nv = self.physics.n_vars
        X = np.array(self.physics.engine.X, copy=False)
        tempr = _Backward1_T_Ph_vec(X[0:nb * nv:nv] / 10, X[1:nb * nv:nv] / 18.015)
        self.temp = tempr
        local_cell_data = {'Temperature': tempr,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data, global_cell_data=global_cell_data)


    def store_moments(self):
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        #
        X = np.array(self.physics.engine.X, copy=False)
        # nxny = self.reservoir.nx * self.reservoir.ny
        used_data = X[0 : 2 * self.nx * self.ny]
        T = _Backward1_T_Ph_vec(used_data[0::2] / 10, used_data[1::2] / 18.015)
        z_pres =self.physics.engine.X[:-1:2]
        z_temp = T

        pres_name = "results/pres_realization" + str(self.realization) + "_years_" + str(int(self.T/365)) + ".txt"
        temp_name = "results/temp_realization" + str(self.realization) + "_years_" + str(int(self.T/365)) + ".txt"
        np.savetxt(pres_name, z_pres, delimiter='\n', newline='\n', header="PRESSURE", comments='')
        np.savetxt(temp_name, z_temp, delimiter='\n', newline='\n', header="TEMPERATURE", comments='')

class geothermal_customized_etor(operator_set_evaluator_iface):
    def __init__(self, property_data):
        super().__init__()

        self.temperature        = property_data.temperature
        # self.water_enthalpy     = property_data.water_enthalpy
        # self.steam_enthalpy     = property_data.steam_enthalpy
        # self.water_saturation   = property_data.water_saturation
        # self.steam_saturation   = property_data.steam_saturation
        # self.water_relperm      = property_data.water_relperm
        # self.steam_relperm      = property_data.steam_relperm
        # self.water_density      = property_data.water_density
        # self.steam_density      = property_data.steam_density
        # self.water_viscosity    = property_data.water_viscosity
        # self.steam_viscosity    = property_data.steam_viscosity
        # self.rock_compaction    = property_data.rock_compaction
        # self.rock_energy        = property_data.rock_energy

    def evaluate(self, state, values):
        # water_enth = self.water_enthalpy.evaluate(state)
        # steam_enth = self.steam_enthalpy.evaluate(state)
        # water_den  = self.water_density.evaluate(state)
        # steam_den  = self.steam_density.evaluate(state)
        # water_sat  = self.water_saturation.evaluate(state)
        # steam_sat  = self.steam_saturation.evaluate(state)
        temp       = self.temperature.evaluate(state)
        # water_rp   = self.water_relperm.evaluate(state)
        # steam_rp   = self.steam_relperm.evaluate(state)
        # water_vis  = self.water_viscosity.evaluate(state)
        # steam_vis  = self.steam_viscosity.evaluate(state)
        # pore_volume_factor = self.rock_compaction.evaluate(state)
        # rock_int_energy    = self.rock_energy.evaluate(state)
        # pressure = state[0]


        values[0] = temp

        return 0