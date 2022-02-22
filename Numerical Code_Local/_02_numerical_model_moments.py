from _02_numerical_model_moment_z_deflection_y import MomentZDeflectionY
from _02_numerical_model_moment_y_deflection_z import MomentYDeflectionZ
from _02_numerical_model_moment_x_twist import MomentXTwist
import matplotlib.pyplot as plt
import numpy as np


class Moments:
    def __init__(self, data_dict):
        self.initial_data_dict = data_dict
        self.updated_data_dict = None

        self.m_x = MomentXTwist(data_dict=data_dict)
        self.m_y = MomentYDeflectionZ(data_dict=data_dict)
        self.m_z = MomentZDeflectionY(data_dict=data_dict)

        self.l_a = data_dict['l_a']
        self.d_1 = data_dict['d_1']
        self.d_2 = data_dict['d_2']
        self.d_3 = data_dict['d_3']

        self.x_1 = data_dict['x_1']
        self.x_2 = data_dict['x_2']
        self.x_3 = data_dict['x_3']
        self.x_a = data_dict['x_a']

        self.P = data_dict['P']
        self.dx = data_dict['dx']

        self.theta = data_dict['theta']

        self.x_arr = np.arange(0, self.l_a, self.dx)
        self._contrib_dict = {"R_1_y": 0, "R_1_z": 0,
                              "R_2_y": 0, "R_2_z": 0,
                              "R_3_y": 0, "R_3_z": 0,
                              "C_1_z": 0, "C_2_z": 0,
                              "C_1_y": 0, "C_2_y": 0,
                              "R_I": 0, "C_1_x": 0}

    def solve_for_reaction_forces(self):
        # In this function we will be going from our boundary conditions to the calculation of the reaction forces.
        boundary_conditions = [
            # 1. v_y(x_1) = d_1*cos(theta)
            {"ls": self.m_z.contribution_v_y(self.x_1), "rs": self.d_1 * np.cos(self.theta)},
            # 2. v_y(x_2) = d_2*cos(theta) -- We usually set d_2 to zero though
            {"ls": self.m_z.contribution_v_y(self.x_2), "rs": self.d_2 * np.cos(self.theta)},
            # 3. v_y(x_3) = d_3*cos(theta)
            {"ls": self.m_z.contribution_v_y(self.x_3), "rs": self.d_3 * np.cos(self.theta)},

            # 4. v_z(x_1) = -d_1*sin(theta)
            {"ls": self.m_y.contribution_v_z(self.x_1), "rs": -self.d_1 * np.sin(self.theta)},
            # 5. v_z(x_2) = -d_1*sin(theta) -- We usually set d_2 to zero though
            {"ls": self.m_y.contribution_v_z(self.x_2), "rs": -self.d_2 * np.sin(self.theta)},
            # 6. v_z(x_3) = -d_3*sin(theta)
            {"ls": self.m_y.contribution_v_z(self.x_3), "rs": -self.d_3 * np.sin(self.theta)},

            # 7. theta_x(x_2) = 0
            {"ls": self.m_x.contribution_theta(self.x_2), "rs": 0},

            # 8. M_x(l) = 0
            {"ls": self.m_x.contribution_m_x(self.l_a), "rs": 0},
            # 9. M_z(l) = 0
            {"ls": self.m_z.contribution_m_z(self.l_a), "rs": 0},
            # 10. M_y(l) = 0
            {"ls": self.m_y.contribution_m_y(self.l_a), "rs": 0},

            # 11. S_y(l) = 0
            {"ls": self.m_y.contribution_s_z(self.l_a), "rs": 0},
            # 12. S_z(l) = 0
            {"ls": self.m_z.contribution_s_y(self.l_a), "rs": 0}]

        # now that we have our twelve boundary conditions in a neat format, we need to transform matrix format.
        A = np.zeros((12, 12))
        b = np.zeros((12, 1))

        for row, bc in enumerate(boundary_conditions):
            # Building up the b vector
            b[row] = bc['rs'] - self.P * bc['ls']['P'] - bc['ls']['aero']

            # Building up the A matrix
            for column, key in enumerate(self._contrib_dict):
                A[row, column] = bc['ls'][key]

        # The magic sauce:
        x = np.matmul(np.linalg.inv(A), b)

        self.updated_data_dict = self.initial_data_dict.copy()
        for x_entry, key in enumerate(self._contrib_dict):
            self.updated_data_dict[key] = x[x_entry][0]

        return self.updated_data_dict

    def get_moment_functions(self, data_dict, plot=False, save_plot=True, return_new_dict=False):
        if self.updated_data_dict is None:
            self.solve_for_reaction_forces()

        m_x = self.m_x.get_moment_around_x(self.updated_data_dict)
        m_z = self.m_z.get_moment_around_z(self.updated_data_dict)
        m_y = self.m_y.get_moment_around_y(self.updated_data_dict)

        if plot:
            m_x_values = m_x(self.x_arr)
            m_y_values = m_y(self.x_arr)
            m_z_values = m_z(self.x_arr)

            plt.plot(self.x_arr, m_x_values, label="$M_x(x)$")
            plt.plot(self.x_arr, m_y_values, label="$M_y(x)$")
            plt.plot(self.x_arr, m_z_values, label="$M_z(x)$")

            max_value = np.max([np.max(m_x_values), np.max(m_z_values), np.max(m_y_values)])
            min_value = np.min([np.min(m_x_values), np.min(m_y_values), np.min(m_z_values)])

            p5 = (max_value - min_value)*0.05
            max_value += p5
            min_value -= p5

            plt.plot([self.x_1]*2, [min_value, max_value], dashes=[1, 2], label="$x_1$")
            plt.plot([self.x_2 - self.x_a/2]*2, [min_value, max_value], dashes=[1, 2], label="$x_{R_I}$")
            plt.plot([self.x_2]*2, [min_value, max_value], dashes=[1, 2], label="$x_2$")
            plt.plot([self.x_2 + self.x_a/2]*2, [min_value, max_value], dashes=[1, 2], label="$x_P$")
            plt.plot([self.x_3]*2, [min_value, max_value], dashes=[1, 2], label="$x_3$")

            plt.ylim([min_value, max_value])

            plt.legend()
            plt.title("Moments")
            plt.xlabel("x-position [m]")
            plt.ylabel("Moment [Nm]")
            plt.tight_layout()
            if save_plot:
                plt.savefig("images/numerical_model_moments.pdf")
            plt.show()
        if return_new_dict:
            data_dict['M_x'] = m_x
            data_dict['M_y'] = m_y
            data_dict['M_z'] = m_z
            return data_dict

        return  m_x, m_y, m_z

    def get_displacement_functions(self, data_dict, plot=False, save_plot=True, return_new_dict=False):
        if self.updated_data_dict is None:
            self.solve_for_reaction_forces()

        v_y = self.m_z.get_deflection_in_y(self.updated_data_dict)
        v_z = self.m_y.get_deflection_in_z(self.updated_data_dict)
        theta = self.m_x.get_twist_around_x(self.updated_data_dict)

        if plot:
            v_y_values = v_y(self.x_arr)*1000
            v_z_values = v_z(self.x_arr)*1000
            theta_values = theta(self.x_arr)*180/np.pi

            plt.plot(self.x_arr, v_y_values, label="$v_y(x)$  $[mm]$")
            plt.plot(self.x_arr, v_z_values, label="$v_z(x)$  $[mm]$")
            plt.plot(self.x_arr, theta_values, label=r"$\theta_x(x)$  $[deg]$")

            # now adding some lines to make it easier to analyze
            max_value = np.max([np.max(v_y_values), np.max(v_z_values), np.max(theta_values)])
            min_value = np.min([np.min(v_y_values), np.min(v_z_values), np.min(theta_values)])

            p5 = (max_value - min_value)*0.05
            max_value += p5
            min_value -= p5

            plt.plot([self.x_1]*2, [min_value, max_value], dashes=[1, 2], label="$x_1$")
            plt.plot([self.x_2 - self.x_a/2]*2, [min_value, max_value], dashes=[1, 2], label="$x_{R_I}$")
            plt.plot([self.x_2]*2, [min_value, max_value], dashes=[1, 2], label="$x_2$")
            plt.plot([self.x_2 + self.x_a/2]*2, [min_value, max_value], dashes=[1, 2], label="$x_P$")
            plt.plot([self.x_3]*2, [min_value, max_value], dashes=[1, 2], label="$x_3$")

            plt.ylim([min_value, max_value])

            plt.title("Displacements")
            plt.xlabel("x-position [m]")
            plt.ylabel("Displacement [mm] or twist [deg]")
            plt.legend()
            plt.tight_layout()
            if save_plot:
                plt.savefig("images/numerical_model_displacements.pdf")
            plt.show()

        if return_new_dict:
            data_dict['v_y'] = v_y
            data_dict['v_z'] = v_z
            data_dict['theta_x'] = theta
            return data_dict

        return {"v_y": v_y, "v_z": v_z, "theta_x": theta}

    def get_shear_functions(self, data_dict, plot=False, save_plot=True, return_new_dict=False):
        if self.updated_data_dict is None:
            self.solve_for_reaction_forces()

        s_y = self.m_z.get_shear_force_in_y(self.updated_data_dict)
        s_z = self.m_y.get_shear_force_in_z(self.updated_data_dict)

        if plot:
            s_y_values = s_y(self.x_arr)*1e-3
            s_z_values = s_z(self.x_arr)*1e-3

            plt.plot(self.x_arr, s_y_values, label="$S_y$")
            plt.plot(self.x_arr, s_z_values, label="$S_z$")

            # now adding some lines to make it easier to analyze
            max_value = np.max([np.max(s_y_values), np.max(s_z_values)])
            min_value = np.min([np.min(s_z_values), np.min(s_y_values)])

            p5 = (max_value - min_value) * 0.05
            max_value += p5
            min_value -= p5

            plt.plot([self.x_1] * 2, [min_value, max_value], dashes=[1, 2], label="$x_1$")
            plt.plot([self.x_2 - self.x_a / 2] * 2, [min_value, max_value], dashes=[1, 2], label="$x_{R_I}$")
            plt.plot([self.x_2] * 2, [min_value, max_value], dashes=[1, 2], label="$x_2$")
            plt.plot([self.x_2 + self.x_a / 2] * 2, [min_value, max_value], dashes=[1, 2], label="$x_P$")
            plt.plot([self.x_3] * 2, [min_value, max_value], dashes=[1, 2], label="$x_3$")

            plt.ylim([min_value, max_value])
            plt.grid()

            plt.legend()
            plt.xlabel("x-position [m]")
            plt.ylabel("Force [KN]")
            plt.title("Shear forces")
            plt.tight_layout()
            if save_plot:
                plt.savefig("images/numerical_model_shear_forces.pdf")
            plt.show()

        if return_new_dict:
            data_dict['S_y'] = s_y
            data_dict['S_z'] = s_z
            return data_dict

        return s_y,  s_z
