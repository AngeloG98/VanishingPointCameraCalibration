import os
import sys
import copy
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import io, feature, color, transform
from PIL import Image
from PIL import ImageEnhance

"""
Image Processing.
"""
def read_image(path):
    image = io.imread(path)
    return image

def image_undistort(image):
    f = 3000
    k1 = 0.2
    k2 = -0.07
    cx = image.shape[1] / 2
    cy = image.shape[0] / 2
    rows = image.shape[0]
    cols = image.shape[1]
    image_undistort = copy.deepcopy(image)
    for v in range(rows):
        for u in range(cols):
            x = (u - cx) / f
            y = (v - cy) / f
            r = math.sqrt(x * x + y * y)
            x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r)
            y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) 
            u_distorted = f * x_distorted + cx
            v_distorted = f * y_distorted + cy

            if u_distorted >= 0 and v_distorted >= 0 and u_distorted < cols and v_distorted < rows:
                image_undistort[v][u] = image[int(v_distorted)][int(u_distorted)]
            else:
                image_undistort[v][u] = 0
    return image_undistort

def image_enhance(image, contrast, sharpness):
    img_contrast = ImageEnhance.Contrast(image)
    image_contrasted = img_contrast.enhance(contrast)
    # image_contrasted.show()
    img_sharp = ImageEnhance.Sharpness(image_contrasted)
    image_sharped = img_sharp.enhance(sharpness)
    # image_sharped.show()
    return image_sharped

def get_canny_edges(image, sigma):
    edges = feature.canny(color.rgb2gray(image), sigma=sigma)
    return edges

def get_hough_lines(edges, line_length, line_gap):
    lines = transform.probabilistic_hough_line(edges, line_length=line_length, line_gap=line_gap)
    return np.asarray(lines)

"""
Result Saving.
"""
def visualize_inliers(image, enhanced, edges, lines, inlier_lines_list, colors, fig_name='detected_lines.png'):
    subplot_count = len(inlier_lines_list) + 3

    fig, axes = plt.subplots(3, subplot_count-3, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title('Input image')
    
    ax[2].imshow(enhanced)
    ax[2].set_title('Enhanced image')
    
    ax[3].imshow(edges)
    ax[3].set_title('Canny edges')

    ax[5].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[5].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[5].set_xlim((0, image.shape[1]))
    ax[5].set_ylim((image.shape[0], 0))
    ax[5].set_title('Probabilistic Hough')
    
    for i in range(len(inlier_lines_list)):
        ax[i+6].imshow(edges * 0)
        for line in lines[inlier_lines_list[i]]:
            p0, p1 = line
            ax[i+6].plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])
        ax[i+6].set_xlim((0, image.shape[1]))
        ax[i+6].set_ylim((image.shape[0], 0))
        ax[i+6].set_title('RANSAC {} Inliers'.format(str(i)))

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()

def visualize_vanishing_points(vp1, vp2, vp3, image, lines, edges, inlier_lines_list, colors, fig_name):
    vps = [vp1, vp2, vp3]
    for i in range(len(inlier_lines_list)):
        plt.imshow(image)
        for line in lines[inlier_lines_list[i]]:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])

        plt.plot([vps[i][0]], [vps[i][1]], colors[i]+'X', markersize=5)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("." + fig_name.split('.')[1] + "_vp" + str(i) + '.png') 
        plt.close()

    plt.imshow(image)
    for i in range(len(inlier_lines_list)):
        for line in lines[inlier_lines_list[i]]:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])

    plt.plot([vps[0][0]], [vps[0][1]], colors[0]+'X', markersize=5)
    plt.plot([vps[1][0]], [vps[1][1]], colors[1]+'X', markersize=5)
    plt.plot([vps[2][0]], [vps[2][1]], colors[2]+'X', markersize=5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fig_name) 
    plt.close() 

def storage_calibration_result(vp1, vp2, vp3, forcal, M_r_o2c, V_t_o2c, img_name):
    vps = [vp1, vp2, vp3]
    with open("./RunResult/" + str(img_name) + 'CalibrationResult.txt', "w") as f:
        f.write("-----------Vanishing Points (in pixel)-----------")
        for i in range(len(vps)):
            f.write("\n")
            f.write("vp" + str(i) + ":\r")
            f.write("x: " + str(vps[i][0]))
            f.write("\r")
            f.write("y: " + str(vps[i][1]))
            # f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("-----------Forcal (in pixel)-----------")
        for i in range(len(forcal)):
            f.write("\n")
            f.write("f" + str(i) + ":\r" + str(forcal[i]))
            f.write("\n")
        f.write("\n")
        f.write("-----------Rotation Matrix (from o to c)-----------")
        for i in range(len(M_r_o2c)):
            f.write("\n")
            f.write("r" + str(i) + ":\n" + str(M_r_o2c[i]))
            f.write("\n")
            euler = rotationmatrix_2_eulerangles(M_r_o2c[i])
            f.write("euler" + str(i) + ":\n" + str(euler))
            f.write("\n")
        f.write("\n")    
        f.write("-----------Translation Vector (in meter)-----------")
        for i in range(len(V_t_o2c)):
            f.write("\n")
            f.write("t" + str(i) + ":\n" + str(V_t_o2c[i]))
            f.write("\n")
        f.close()
    print("Finish Calibration!")

"""
Camera Calibration.
"""
def choose_vanishing_points(vp1, vp2, vp3, image):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    vps = [vp1, vp2, vp3]
    lenth_pp = []
    lenth_r = []
    for i in range(len(vps)):
        if i == 2:
            j = 0
        else:
            j = i + 1  
        lenth_r.append(math.sqrt((vps[i][1] - vps[j][1])** 2 + (vps[i][0] - vps[j][0])** 2))
    if lenth_r[0] <= (image.shape[1] / 4):
        del vps[0]
    elif lenth_r[1] <= (image.shape[1] / 4):
        del vps[1]
    elif lenth_r[2] <= (image.shape[1] / 4):
        del vps[2]
    else:    
        for i in range(len(vps)):
            lenth_pp.append(math.sqrt((vps[i][1] - pp[1])** 2 + (vps[i][0] - pp[0])** 2))
        k = lenth_pp.index(max(lenth_pp))
        # print("del point" + str(k))
        del vps[k]
    return vps

def determine_focal_lenth(vps, image):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    forcal = []
    if len(vps) != 2:
        for i in range(len(vps)):
            if i == 2:
                j = 0
            else:
                j = i + 1  
            if vps[i][0] - vps[j][0] == 0:
                return math.fabs(pp[0] - vps[j][0])
            if vps[i][1] - vps[j][1] == 0:
                return math.fabs(pp[1]- vps[j][1])
            k_uv = (vps[i][1] - vps[j][1]) / (vps[i][0] - vps[j][0])
            b_uv = vps[j][1]- k_uv * vps[j][0]
            pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(k_uv * k_uv + 1, 0.5)
            lenth_uv = math.sqrt((vps[i][1] - vps[j][1])** 2 + (vps[i][0] - vps[j][0])** 2)
            lenth_pu = math.sqrt((vps[i][1] - pp[1])** 2 + (vps[i][0] - pp[0])** 2)
            up_uv = math.sqrt(lenth_pu ** 2 - pp_uv ** 2)
            vp_uv = abs(lenth_uv - up_uv)
            forcal.append( math.sqrt(abs(up_uv * vp_uv - (pp_uv)** 2) ) )
    else:
        if vps[0][0] - vps[1][0] == 0:
            return math.fabs(pp[0] - vps[j][0])
        if vps[0][1] - vps[1][1] == 0:
            return math.fabs(pp[1]- vps[j][1])
        k_uv = (vps[0][1] - vps[1][1]) / (vps[0][0] - vps[1][0])
        b_uv = vps[1][1]- k_uv * vps[1][0]
        pp_uv = math.fabs(k_uv * pp[0] - pp[1] + b_uv) / math.pow(k_uv * k_uv + 1, 0.5)
        lenth_uv = math.sqrt((vps[0][1] - vps[1][1])** 2 + (vps[0][0] - vps[1][0])** 2)
        lenth_pu = math.sqrt((vps[0][1] - pp[1])** 2 + (vps[0][0] - pp[0])** 2)
        up_uv = math.sqrt(lenth_pu ** 2 - pp_uv ** 2)
        vp_uv = abs(lenth_uv - up_uv)
        forcal.append(math.sqrt((up_uv * vp_uv) - ((pp_uv)** 2)))
    return forcal

def calculate_rotation_matrix(vps, image, f):
    pp = [image.shape[1] / 2, image.shape[0] / 2]
    M_r_o2c = []
    u = np.array([vps[0][0] - pp[0], vps[0][1] - pp[1], f[0]])
    u_norm = u / np.sqrt((u * u).sum())
    v = np.array([vps[1][0] - pp[0], vps[1][1] - pp[1], f[0]])
    v_norm = v / np.sqrt((v*v).sum())
    w_norm = np.cross(u_norm, v_norm)
    M_r_o2c.append(np.c_[u_norm, v_norm, w_norm])
    return M_r_o2c

def rotationmatrix_2_eulerangles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z]) * 180 / np.pi

def calculate_translation_vector(image, f , M_r_o2c):
    # def calculate_translation_vector(vps, image, f, lenth_AB, a, b):
    # dpi = 2834.6  # px/m  72dpi
    dpi = 3779.5  # px/m  96dpi
    pp = np.array([image.shape[1] / 2, image.shape[0] / 2, 0]) / dpi
    # np.array([a[0], a[1], f]) / dpi
    # np.array([b[0], b[1], f]) / dpi

    # rt1.jpg test example AB in real world along with axi u
    lenth_AB = 2.44  # m
    u = [1, 0, 0]
    # rt1.jpg test example ab in image
    a_img = np.array([1209, 2382, f[0]]) / dpi   
    b_img = np.array([2402, 2017, f[0]]) / dpi
    
    V_t_o2c = [] 
    oa_c = a_img - pp  # a point 
    ob_c = b_img - pp
    AB_c = lenth_AB * np.dot(M_r_o2c[0], u)
    AB_c_norm = AB_c / np.sqrt((AB_c * AB_c).sum())  # ab' vector
    o_c = [0, 0, 0]   # o point
    ob_c_norm = ob_c / np.sqrt((ob_c * ob_c).sum())  # ob vector
    b_cross = cross_point(oa_c, AB_c_norm, o_c, ob_c_norm)
    if (b_cross == np.array([0, 0, 0])).all():
        print("wrong")
    ab_cross = b_cross - oa_c
    oA_c = oa_c * lenth_AB / np.sqrt((ab_cross * ab_cross).sum())
    V_t_o2c.append(np.dot(np.transpose(M_r_o2c[0]), oA_c))
    return V_t_o2c

def calculate_translation_vector_h(image, f, M_r_o2c, px_x, px_y, h):
    V_t_o2c = []
    dpi = 3779.5  # px/m  96dpi
    pp = np.array([image.shape[1] / 2, image.shape[0] / 2, 0]) / dpi
    xy_img = np.array([1209, 2382, f[0]]) / dpi
    xy_c = xy_img - pp
    lenth_xy_c = np.sqrt((xy_c * xy_c).sum())
    xy_c_norm = xy_c / np.sqrt((xy_c * xy_c).sum())
    # h_c = np.dot(M_r_o2c, np.array([0, 0, -h]))
    h_c = np.dot(M_r_o2c, np.array([0, 0, h]))
    alpha = math.acos(np.dot(h_c, xy_c) / (lenth_xy_c * h))
    oxy_c = (h / math.cos(alpha)) * xy_c_norm
    print(math.cos(alpha))
    V_t_o2c.append(np.dot(np.transpose(M_r_o2c[0]), oxy_c))
    return V_t_o2c


def cross_point(p1, v1, p2, v2):
    if (np.dot(v1, v2) == 1):
        crosspoint = np.array([0, 0, 0])
        return crosspoint
    startseg = p2 - p1
    S1 = np.cross(v1, v2)
    S2 = np.cross(startseg, v2)
    num1 = np.dot(startseg, S1)
    if (num1 >= 0.1 or num1 <= -0.1):
        crosspoint = np.array([0, 0, 0])
        return crosspoint
    num2 = np.dot(S1, S2) / (S1 * S1).sum()
    crosspoint = p1 + v1 * num2
    return crosspoint

"""
Vanishing Point Detection.
"""
def calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh):
    current_hypothesis = current_hypothesis / current_hypothesis[-1]
    hypothesis_vp_direction = current_hypothesis[:2] - lines[:,0]
    lines_vp_direction = lines[:,1] - lines[:,0]
    magnitude = np.linalg.norm(hypothesis_vp_direction, axis=1) * np.linalg.norm(lines_vp_direction, axis=1)
    magnitude[magnitude == 0] = 1e-5
    cos_theta = (hypothesis_vp_direction*lines_vp_direction).sum(axis=-1) / magnitude
    theta = np.arccos(np.abs(cos_theta))
    inliers = (theta < ransac_angle_thresh * np.pi / 180)
    inliers[ignore_pts] = False
    return inliers, inliers.sum()

def run_line_ransac(lines, ransac_iter, ransac_angle_thresh, ignore_pts=None):
    best_vote_count = 0
    best_inliers = None
    best_hypothesis = None
    if ignore_pts is None:
        ignore_pts = np.zeros((lines.shape[0])).astype('bool')
        lines_to_chose = np.arange(lines.shape[0])
    else:
        lines_to_chose = np.where(ignore_pts==0)[0]
    for iter_count in range(ransac_iter):
        idx1, idx2 = np.random.choice(lines_to_chose, 2, replace=False)
        l1 = np.cross(np.append(lines[idx1][1], 1), np.append(lines[idx1][0], 1))
        l2 = np.cross(np.append(lines[idx2][1], 1), np.append(lines[idx2][0], 1))

        current_hypothesis = np.cross(l1, l2)
        if current_hypothesis[-1] == 0:
            continue
        inliers, vote_count = calculate_metric_angle(current_hypothesis, lines, ignore_pts, ransac_angle_thresh)
        if vote_count > best_vote_count:
            best_vote_count = vote_count
            best_hypothesis = current_hypothesis
            best_inliers = inliers
    return best_hypothesis, best_inliers

def get_vp_inliers(image_path, contrast, sharpness, sigma, iterations, line_len, line_gap, threshold):
    image = read_image(image_path)
    image_pil = Image.fromarray(image)
    enhanced_pil = image_enhance(image_pil, contrast=contrast, sharpness=sharpness)
    enhanced_ski = np.array(enhanced_pil)
    edges = get_canny_edges(enhanced_ski, sigma=sigma)
    lines = get_hough_lines(edges, line_length=line_len, line_gap=line_gap)

    best_hypothesis_1, best_inliers_1 = run_line_ransac(lines, iterations, threshold)
    ignore_pts = best_inliers_1
    best_hypothesis_2, best_inliers_2 = run_line_ransac(lines, iterations, threshold, ignore_pts=ignore_pts)
    ignore_pts = np.logical_or(best_inliers_1, best_inliers_2)
    best_hypothesis_3, best_inliers_3 = run_line_ransac(lines, iterations, threshold, ignore_pts=ignore_pts)
    inlier_lines_list = [best_inliers_1, best_inliers_2, best_inliers_3]
    best_hypothesis_1 = best_hypothesis_1 / best_hypothesis_1[-1]
    best_hypothesis_2 = best_hypothesis_2 / best_hypothesis_2[-1]
    best_hypothesis_3 = best_hypothesis_3 / best_hypothesis_3[-1]
    hypothesis_list = [best_hypothesis_1, best_hypothesis_2, best_hypothesis_3]
    viz_stuff = [image, enhanced_ski, edges, lines]
    return inlier_lines_list, hypothesis_list, viz_stuff

"""
Main.
"""
def main(image_path, px_x, px_y, h, contrast=5, sharpness=10, sigma=5, iterations=3000, line_len=11, line_gap=7, threshold: 'float' = 2):
    img_name = os.path.basename(image_path).split('.')[0]
    inlier_lines_list, hypothesis_list, viz_stuff = get_vp_inliers(image_path, contrast, sharpness, sigma, iterations, line_len, line_gap, threshold)
    image, enhanced_ski, edges, lines = viz_stuff
    best_hypothesis_1, best_hypothesis_2, best_hypothesis_3 = hypothesis_list
    fig_name = "./RunResult/"+'{}_inliers_iter{}_thresh{}_sigma{}_hlen{}_hgap{}.png'\
                .format(img_name, iterations, threshold, sigma, line_len, line_gap)
    colors = ['r', 'g', 'b']
    visualize_inliers(image, enhanced_ski, edges, lines, inlier_lines_list, colors, fig_name=fig_name)
    fig_name = "./RunResult/"+'{}_vanishing_point_iter{}_thresh{}_sigma{}_hlen{}_hgap{}.png'\
                .format(img_name, iterations, threshold, sigma, line_len, line_gap)
    visualize_vanishing_points(best_hypothesis_1, best_hypothesis_2, best_hypothesis_3,
                               image, lines, edges, inlier_lines_list, colors, fig_name)
    vps = choose_vanishing_points(best_hypothesis_1, best_hypothesis_2, best_hypothesis_3, image)
    forcal = determine_focal_lenth(vps, image)
    # forcal = determine_focal_lenth([best_hypothesis_1, best_hypothesis_2, best_hypothesis_3], image)
    M_r_o2c = calculate_rotation_matrix(vps, image, forcal)
    V_t_o2c = calculate_translation_vector_h(image, forcal, M_r_o2c, px_x, px_y, h)
    # V_t_o2c = calculate_translation_vector(image, forcal , M_r_o2c)
    storage_calibration_result(best_hypothesis_1, best_hypothesis_2, best_hypothesis_3, forcal, M_r_o2c, V_t_o2c, img_name)
    return forcal, M_r_o2c, V_t_o2c
    

