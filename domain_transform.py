import cv2
import math
import numpy as np
import numpy.matlib as mat
from matplotlib import pyplot as plt

class DomainTransform:

    def __init__(self):
        pass


    def normalized_convolution(self, image_, sigma_s, sigma_r, number_iterations=3, joint_image=None):
        image_to_filter = image_
        image_to_filter = image_to_filter.reshape(image_.shape)
        if joint_image is not None:
            parameter_image = joint_image
        else:
            parameter_image = image_to_filter
        #height, width, channels = parameter_image.shape
        gray_scale_parameter = parameter_image
        dIcdx = cv2.Sobel(gray_scale_parameter, cv2.CV_64F, 1, 0, ksize=3)
        dIcdx = dIcdx.reshape(gray_scale_parameter.shape)
        dIcdy = cv2.Sobel(gray_scale_parameter, cv2.CV_64F, 0, 1, ksize=3)
        dIcdy = dIcdy.reshape(gray_scale_parameter.shape)
        dIdx = np.absolute(dIcdx)
        dIdy = np.absolute(dIcdy)
        axis_dx = 1
        axis_dy = 1
        if len(dIdx.shape) > 2 and dIdx.shape[2] - 1 > 0:
            axis_dx = dIdx.shape[2] - 1
            dIdx = np.sum(dIdx, axis = axis_dx)

        if len(dIdy.shape) > 2 and dIdy.shape[2] - 1 > 0:
            axis_dy = dIdy.shape[2] - 1
            dIdy = np.sum(dIdy, axis = axis_dy)


        dHdx = (1 + sigma_s/sigma_r * dIdx)
        dVdy = (1 + sigma_s/sigma_r * dIdy)

        ct_H = np.cumsum(dHdx, axis=1)
        ct_V = np.cumsum(dVdy, axis=0)

        ct_V = self.transpose(ct_V)

        N = number_iterations
        current_image = image_to_filter

        sigma_H = sigma_s

        for i in range(0, number_iterations):
            exponent = (N - (i + 1))/math.sqrt(math.pow(4.0, N) - 1)
            sigma_H_i = sigma_H * math.sqrt(3) * math.pow(2.0, exponent)
            box_radius = math.sqrt(3) * sigma_H_i
            current_image = self.transformed_domain_box_filter_horizontal(current_image, ct_H, box_radius)
            current_image = self.transpose(current_image)

            current_image = self.transformed_domain_box_filter_horizontal(current_image, ct_V, box_radius)
            current_image = self.transpose(current_image)

        return current_image

    def transformed_domain_box_filter_row(self, xform_domain_position_row, l_pos_row, u_pos_row, width):
        local_l_idx = np.zeros((1, width))
        local_u_idx = np.zeros((1, width))

        aux_find = np.where(xform_domain_position_row > l_pos_row[0])
        local_l_idx[0][0] = aux_find[0][0]

        aux_find_1 = np.where(xform_domain_position_row > u_pos_row[0])
        local_u_idx[0][0] = aux_find_1[0][0]

        for col in range(1, width):
            current_index = int(local_l_idx[0][col - 1])
            aux_find = np.where(xform_domain_position_row[current_index:] > l_pos_row[col])
            local_l_idx[0][col] = local_l_idx[0][col - 1] + aux_find[0][0]

            current_index = int(local_u_idx[0][col - 1])
            aux_find = np.where(xform_domain_position_row[current_index:] > u_pos_row[col])
            local_u_idx[0][col] = local_u_idx[0][col - 1] + aux_find[0][0]
        return local_l_idx, local_u_idx

    def transformed_domain_box_filter_horizontal(self, image_, xform_domain_position, box_radius):
        if len(image_.shape) > 2:
            height, width, channels = image_.shape
        else:
            height, width = image_.shape
            channels = 1

        l_pos = xform_domain_position - box_radius
        u_pos = xform_domain_position + box_radius

        l_idx = np.zeros(xform_domain_position.shape)
        u_idx = np.zeros(xform_domain_position.shape)

        for row in range(0, height):
            xform_domain_position_row = np.append(xform_domain_position[row, :], np.inf)
            l_pos_row = l_pos[row, :]
            u_pos_row = u_pos[row, :]
            local_l_idx = np.zeros((1, width))
            local_u_idx = np.zeros((1, width))

            aux_find = np.where(xform_domain_position_row > l_pos_row[0])
            local_l_idx[0][0] = aux_find[0][0]

            aux_find_1 = np.where(xform_domain_position_row > u_pos_row[0])
            local_u_idx[0][0] = aux_find_1[0][0]

            for col in range(1, width):
                current_index = int(local_l_idx[0][col - 1])
                aux_find = np.where(xform_domain_position_row[current_index:] >= l_pos_row[col])
                local_l_idx[0][col] = local_l_idx[0][col - 1] + aux_find[0][0]

                current_index = int(local_u_idx[0][col - 1])
                aux_find = np.where(xform_domain_position_row[current_index:] >= u_pos_row[col])
                local_u_idx[0][col] = local_u_idx[0][col - 1] + aux_find[0][0]

            l_idx[row, :] = local_l_idx[:,:]
            u_idx[row, :] = local_u_idx[:,:]

        if channels > 1:
            SAT = np.zeros((height, width + 1, channels))
            SAT[:, 1:, :] = np.cumsum(image_, axis=1)
        else:
            SAT = np.zeros((height, width + 1))
            SAT[:, 1:] = np.cumsum(image_, axis=1)
        F = np.zeros(image_.shape)
        self.calculate_filtered(l_idx, u_idx, channels, SAT, F)
        return F

    def calculate_filtered(self, col_l_indices, col_u_indices, num_channel, acc, F):
        for i in range(0, acc.shape[0]):
            for j in range(0, acc.shape[1] - 1):
                row_result = i#int(row_indices[i, j])
                l_idx = int(col_l_indices[i, j]) #This could be any index
                u_idx = int(col_u_indices[i, j])
                if num_channel > 1:
                    for c in range(0, num_channel):
                        result_SAP = acc[row_result, u_idx, c] - acc[row_result, l_idx, c]
                        F[i, j, c] = result_SAP/(u_idx - l_idx)
                else:
                    result_SAP = acc[row_result, u_idx] - acc[row_result, l_idx]
                    F[i, j] = result_SAP / (u_idx - l_idx)
        return F



    @staticmethod
    def transpose(_image):
        return cv2.transpose(_image)
