import biorbd
import numpy as np
from IPython import embed

def rotate_pelvis_to_initial_orientation(num_joints, move_orientation, Xsens_position, Xsens_orientation, pelvis_resting_frames):
    """
    This function realigns the pelvis to be faced to the front wall at the beginig of the trial.
    """

    vect_hips = Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] - Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3]
    vect_vert = Xsens_position[pelvis_resting_frames, 0*3:(0+1)*3] - (Xsens_position[pelvis_resting_frames, 15*3:(15+1)*3] + Xsens_position[pelvis_resting_frames, 19*3:(19+1)*3])/2
    vect_hips_mean = np.mean(vect_hips, axis=0)
    vect_vert_mean = np.mean(vect_vert, axis=0)
    vect_front_mean = np.cross(vect_hips_mean, vect_vert_mean)
    vect_hips_mean[2] = 0
    if move_orientation[0] == 1:
        desired_vector = np.array([0, -1, 0])
    elif move_orientation[0] == -1:
        desired_vector = np.array([0, 1, 0])
    else:
        print('Error: move_orientation[0] should be 1 or -1')
        embed()

    angle_needed = np.arccos(np.dot(vect_hips_mean, desired_vector) / (np.linalg.norm(vect_hips_mean) * np.linalg.norm(desired_vector)))
    if vect_front_mean[0] < 0:
        angle_needed = -angle_needed
    rotation_matrix = biorbd.Rotation.fromEulerAngles(np.array([0, 0, angle_needed]), 'xyz').to_array()
    print('\n****angle_needed : ', angle_needed*180/np.pi, '****')
    print('Hips vector : ', vect_hips_mean)
    print('Desired vector : ', desired_vector)
    print('\n')

    # rotate the xsens positions around the hip
    Xsens_position_rotated = np.zeros(np.shape(Xsens_position))
    for i in range(np.shape(Xsens_position)[0]):
        for k in range(num_joints):
            Xsens_position_rotated[i, 3*k : 3*(k+1)] = rotation_matrix @ Xsens_position[i, 3*k : 3*(k+1)]

    Xsens_orientation_rotated = np.zeros(np.shape(Xsens_orientation))
    Xsens_orientation_rotated[:, :] = Xsens_orientation[:, :]
    for i in range(np.shape(Xsens_orientation)[0]):
        Quat_normalized_head = Xsens_orientation[i, 24:28] / np.linalg.norm(
            Xsens_orientation[i, 24:28]
        )
        Quat_normalized_thorax = Xsens_orientation[i, 16:20] / np.linalg.norm(
            Xsens_orientation[i, 16:20]
        )
        Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
                                      Quat_normalized_head[3])
        Quat_thorax = biorbd.Quaternion(Quat_normalized_thorax[0], Quat_normalized_thorax[1], Quat_normalized_thorax[2],
                                        Quat_normalized_thorax[3])
        RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()
        RotMat_thorax = biorbd.Quaternion.toMatrix(Quat_thorax).to_array()
        RotMat_head_rotated = rotation_matrix @ RotMat_head
        RotMat_thorax_rotated = rotation_matrix @ RotMat_thorax

        Quat_head_rotated = biorbd.Quaternion.fromMatrix(biorbd.Rotation(RotMat_head_rotated[0, 0], RotMat_head_rotated[0, 1], RotMat_head_rotated[0, 2],
                                                                         RotMat_head_rotated[1, 0], RotMat_head_rotated[1, 1], RotMat_head_rotated[1, 2],
                                                                         RotMat_head_rotated[2, 0], RotMat_head_rotated[2, 1], RotMat_head_rotated[2, 2])).to_array()
        Quat_thorax_rotated = biorbd.Quaternion.fromMatrix(biorbd.Rotation(RotMat_thorax_rotated[0, 0], RotMat_thorax_rotated[0, 1], RotMat_thorax_rotated[0, 2],
                                                                            RotMat_thorax_rotated[1, 0], RotMat_thorax_rotated[1, 1], RotMat_thorax_rotated[1, 2],
                                                                            RotMat_thorax_rotated[2, 0], RotMat_thorax_rotated[2, 1], RotMat_thorax_rotated[2, 2])).to_array()
        Xsens_orientation_rotated[i, 24:28] = Quat_head_rotated
        Xsens_orientation_rotated[i, 16:20] = Quat_thorax_rotated


    # embed()
    # import matplotlib.pyplot as plt
    # from animate_JCS import plot_gymnasium
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plot_gymnasium(3 + 121 * 0.0254 / 2, ax)
    # for i in range(num_joints):
    #     ax.plot(Xsens_position_rotated[pelvis_resting_frames[0], 3*i], Xsens_position_rotated[pelvis_resting_frames[0], 3*i+1], Xsens_position_rotated[pelvis_resting_frames[0], 3*i+2], '.b')
    #     ax.plot(Xsens_position[pelvis_resting_frames[0], 3*i], Xsens_position[pelvis_resting_frames[0], 3*i+1], Xsens_position[pelvis_resting_frames[0], 3*i+2], '.k')
    #
    #
    # Quat_normalized_head=Xsens_orientation[pelvis_resting_frames[0], 24:28] / np.linalg.norm(
    #         Xsens_orientation[pelvis_resting_frames[0], 24:28]
    #     )
    # Quat_head = biorbd.Quaternion(Quat_normalized_head[0], Quat_normalized_head[1], Quat_normalized_head[2],
    #                               Quat_normalized_head[3])
    # RotMat_head = biorbd.Quaternion.toMatrix(Quat_head).to_array()
    # RotMat_head_rotated = rotation_matrix @ RotMat_head
    #
    # fake_head_orientation_vector_rotated = RotMat_head_rotated @ np.array([1, 0, 0])
    # fake_head_orientation_vector = RotMat_head @ np.array([1, 0, 0])
    #
    # ax.plot(np.array([0, fake_head_orientation_vector_rotated[0]]), np.array([0, fake_head_orientation_vector_rotated[1]]), np.array([0, fake_head_orientation_vector_rotated[2]]), '-b')
    # ax.plot(np.array([0, fake_head_orientation_vector[0]]), np.array([0, fake_head_orientation_vector[1]]), np.array([0, fake_head_orientation_vector[2]]), '-k')
    # plt.show()

    return Xsens_position_rotated, Xsens_orientation_rotated


def get_initial_gaze_orientation(eye_resting_frames, azimuth, elevation):
    """
    This function identify the resting gaze orientation since the zero from Pupil is depandant on the face geometry of the subjects.
    """
    azimuth_zero = np.mean(azimuth[eye_resting_frames])
    elevation_zero = np.mean(elevation[eye_resting_frames])
    azimuth_zero_mean = np.mean(azimuth_zero)
    elevation_zero_mean = np.mean(elevation_zero)

    return azimuth_zero_mean, elevation_zero_mean







