from smpl_measure import *
import numpy as np
import os
import pickle
import csv
from os.path import join, abspath, dirname
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt

folder_dir = join(abspath(dirname(__file__)), sys.argv[1])
pkl_paths = glob(join(folder_dir, "*.pkl"))
measure_result_dir = "measure_result.csv"
measure_avr_dir = "measure_avr.csv"
name_error_dir = "name_error.csv"
pose_error_dir = "pose_error.csv"
angle_error_dir = "angle_error.csv"
error_result_dir = "error_result.csv"
ground_truth_dir = "female(copy).csv"

pose_index = ['차렷','A pose', 'T pose', '항복','앞으로나란히', '걷는']
angle_index = ['(머리위)','(얼굴)','(명치)','(하복부)']

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python %s smpl_param.pkl")  # @TODO extension to the Vertex model case
        exit()

    ground_truth_df = pd.read_csv(ground_truth_dir)

    smpl = loadSMPL(gender='female')

    if os.path.isfile(measure_result_dir):
        os.remove(measure_result_dir)

    for ind, pkl_path in enumerate(pkl_paths):
        #print(pkl_path)
        result = {}
        pkl_file = pkl_path.split('/')[-1].split('.')[0]

        result['Pose'] = pose_index[int(pkl_file.split('_')[0])-1]
        result['Cloth'] = int(pkl_file.split('_')[1])
        result['Name'] = pkl_file.split('_')[2]
        result['Angle'] =str(45*(int(pkl_file.split('_')[3])//4))+'도'+angle_index[int(pkl_file.split('_')[3])%4-1]
        height = ground_truth_df[ground_truth_df.Folder == pkl_file.split('_')[2]]['Stature_Height'].values[0]


        try:
            #measure_len = measure_correction(smpl,pkl_path,height)
            measure_len = measure(smpl,pkl_path)
        except: continue
        result.update(measure_len)
        #print(result)
        with open(measure_result_dir, 'a') as f:
            w = csv.writer(f)
            if ind == 0: w.writerow(result.keys())
            w.writerow(result.values())

    #######################################################ERROR계산########################################################################################333

    measure_df = pd.read_csv(measure_result_dir)


    import copy
    error_df = copy.deepcopy(measure_df)
    error_result = []
    for i in range(error_df.shape[0]):
        error = error_df.iloc[i].values[:]
        ground = ground_truth_df.loc[ground_truth_df.Folder == error_df.iloc[i].values[2]].values[0, 1:]
        error[4:] = np.round(((error[4:]-ground)/ground*100).astype(np.double),2)
        error_df.iloc[i] = error

    #measure_df.to_csv(error_result_dir, mode='w')
    ##################################################################################
    name_error = []
    for name in np.unique(error_df['Name']):
        error = []
        error.append(name)
        error.extend(np.round(np.average(np.abs(error_df.loc[error_df.Name == name].values[:,4:]),axis=0).astype(np.double),2))
        name_error.append(error)

    name_key = ['Name']
    name_key.extend(error_df.keys()[4:])

    name_error_df = pd.DataFrame(name_error, columns=name_key)
    #name_error_df.to_csv(name_error_dir, mode='w')

    ##################################################################################
    pose_error = []
    for pose in np.unique(error_df['Pose']):
        error = []
        error.append(pose)
        error.extend(
            np.round(np.average(np.abs(error_df.loc[error_df.Pose == pose].values[:, 4:]), axis=0).astype(np.double), 2))
        pose_error.append(error)

    pose_key = ['Pose']
    pose_key.extend(error_df.keys()[4:])

    pose_error_df = pd.DataFrame(pose_error, columns=pose_key)
    #pose_error_df.to_csv(pose_error_dir, mode='w')
    ##################################################################################
    angle_error = []
    for angle in np.unique(error_df['Angle']):
        error = []
        error.append(angle)
        error.extend(
            np.round(np.average(np.abs(error_df.loc[error_df.Angle == angle].values[:, 4:]), axis=0).astype(np.double), 2))
        angle_error.append(error)

    angle_key = ['Angle']
    angle_key.extend(error_df.keys()[4:])

    angle_error_df = pd.DataFrame(angle_error, columns=angle_key)
    #angle_error_df.to_csv(angle_error_dir, mode='w')
    try:
        output_name = sys.argv[1].split('/')[0]
    except:
        output_name = sys.argv[1]
    writer = pd.ExcelWriter(output_name+"_final_error.xlsx")
    error_df.to_excel(writer, sheet_name='raw error')
    name_error_df.to_excel(writer, sheet_name='abs name error')
    pose_error_df.to_excel(writer, sheet_name='abs pose error')
    angle_error_df.to_excel(writer, sheet_name='abs angle error')


    '''##########################################################################'''
    ##################################################################################
    name_error = []
    name_std =[]
    for name in np.unique(error_df['Name']):
        error = []
        error.append(name)
        error.extend(np.round(
            np.average(error_df.loc[error_df.Name == name].values[:, 4:], axis=0).astype(np.double), 2))
        name_error.append(error)

        std = []
        std.append(name)
        std.extend(np.round(np.std(error_df.loc[error_df.Name == name].values[:, 4:].astype(np.double), axis=0),2))
        name_std.append(std)

    name_key = ['Name']
    name_key.extend(error_df.keys()[4:])

    name_error_df = pd.DataFrame(name_error, columns=name_key)
    name_std_df = pd.DataFrame(name_std,columns=name_key)

    #name_error_df.to_csv(name_error_dir, mode='w')

    ##################################################################################
    pose_error = []
    for pose in np.unique(error_df['Pose']):
        error = []
        error.append(pose)
        error.extend(
            np.round(
                np.average(error_df.loc[error_df.Pose == pose].values[:, 4:], axis=0).astype(np.double), 2))
        pose_error.append(error)

    pose_key = ['Pose']
    pose_key.extend(error_df.keys()[4:])

    pose_error_df = pd.DataFrame(pose_error, columns=pose_key)
    #pose_error_df.to_csv(pose_error_dir, mode='w')
    ##################################################################################
    angle_error = []
    for angle in np.unique(error_df['Angle']):
        error = []
        error.append(angle)
        error.extend(
            np.round(
                np.average(error_df.loc[error_df.Angle == angle].values[:, 4:], axis=0).astype(np.double),
                2))
        angle_error.append(error)

    angle_key = ['Angle']
    angle_key.extend(error_df.keys()[4:])

    angle_error_df = pd.DataFrame(angle_error, columns=angle_key)
    #angle_error_df.to_csv(angle_error_dir, mode='w')


    name_error_df.to_excel(writer, sheet_name='name error')
    pose_error_df.to_excel(writer, sheet_name='pose error')
    angle_error_df.to_excel(writer, sheet_name='angle error')
    writer.save()

    input_name = output_name+"_final_error.xlsx"
    output_name = input_name.split('.')[0]+"_bmi.xlsx"
    error_df = pd.read_excel(input_name, sheet_name=0)
    '''plt.figure(figsize=(72, 48))
    fig, axes = plt.subplots(3, 5)
    import seaborn as sns
    for idx, key in enumerate(list(landmark_female.keys())[1:]):
        sns.histplot(data=error_df[key], ax=axes[idx // 5, idx % 5])  # ;axes[idx//4,idx%4].set_title(key)
    plt.suptitle(input_name)
    plt.savefig(input_name.split('.')[0]+'_histogram.png',dpi=300)'''

    from operator import add
    #########################################################################################################################
    '''BMI'''
    bmi_arr = [['F153', 'F218', 'F098'], ['F089', 'F236', 'F093'], ['F227', 'F095', 'F016'], ['F201', 'F109', 'F117'],
               ['F013', 'F053', 'F071']]
    bmi_error = []
    bmi_abs_error = []
    bmi_std = []
    for idx, bmi_list in enumerate(bmi_arr):
        temp = []
        temp.append(idx)
        temp.extend(np.round(np.average(error_df[((error_df.Name == bmi_list[0]) + (error_df.Name == bmi_list[1]) + (
                    error_df.Name == bmi_list[2]))].iloc[:].values[:, 5:], axis=0).astype(np.double), 2).tolist())
        bmi_error.append(temp)

        temp = []
        temp.append(idx)
        temp.extend(np.round(np.average(np.abs(error_df[(
                (error_df.Name == bmi_list[0]) + (error_df.Name == bmi_list[1]) + (error_df.Name == bmi_list[2]))].iloc[
                                               :].values[:, 5:]), axis=0).astype(np.double), 2).tolist())
        bmi_abs_error.append(temp)

        temp = []
        temp.append(idx)
        temp.extend(np.round(np.std(error_df[(
                (error_df.Name == bmi_list[0]) + (error_df.Name == bmi_list[1]) + (error_df.Name == bmi_list[2]))].iloc[
                                    :].values[:, 5:].astype(np.double), axis=0), 2).tolist())
        bmi_std.append(temp)

    slash = [['/' for j in range(len(bmi_error[0]))] for i in range(len(bmi_error))]
    bmi_error = [list(map(str, i)) for i in bmi_error]
    bmi_abs_error = [list(map(str, i)) for i in bmi_abs_error]
    bmi_std = [list(map(str, i)) for i in bmi_std]

    a = [list(map(add, bmi_error[i], slash[i])) for i in range(len(bmi_error))]
    a = [list(map(add, a[i], bmi_abs_error[i])) for i in range(len(a))]
    a = [list(map(add, a[i], slash[i])) for i in range(len(a))]
    a = [list(map(add, a[i], bmi_std[i])) for i in range(len(a))]

    bmi_key = ['BMI']
    bmi_key.extend(error_df.keys()[5:])

    bmi_error_df = pd.DataFrame(bmi_error, columns=bmi_key)
    bmi_abs_error_df = pd.DataFrame(bmi_abs_error, columns=bmi_key)
    bmi_std_df = pd.DataFrame(bmi_std, columns=bmi_key)
    bmi_df = pd.DataFrame(a, columns=bmi_key)
    # print(bmi_error_df)

    #########################################################################################################################
    '''POSE'''
    pose_arr = ['차렷', 'A pose', 'T pose']
    pose_error = []
    pose_abs_error = []
    pose_std = []
    for pose in pose_arr:
        temp = []
        temp.append(pose)
        temp.extend(
            np.round(np.average(error_df[error_df.Pose == pose].iloc[:].values[:, 5:], axis=0).astype(np.double), 2))
        pose_error.append(temp)

        temp = []
        temp.append(pose)
        temp.extend(
            np.round(
                np.average(np.abs(error_df[error_df.Pose == pose].iloc[:].values[:, 5:]), axis=0).astype(np.double), 2))
        pose_abs_error.append(temp)

        temp = []
        temp.append(pose)
        temp.extend(
            np.round(np.std(error_df[error_df.Pose == pose].iloc[:].values[:, 5:].astype(np.double), axis=0), 2))
        pose_std.append(temp)

    pose_error = [list(map(str, i)) for i in pose_error]
    pose_abs_error = [list(map(str, i)) for i in pose_abs_error]
    pose_std = [list(map(str, i)) for i in pose_std]

    a = [list(map(add, pose_error[i], slash[i])) for i in range(len(pose_error))]
    a = [list(map(add, a[i], pose_abs_error[i])) for i in range(len(a))]
    a = [list(map(add, a[i], slash[i])) for i in range(len(a))]
    a = [list(map(add, a[i], pose_std[i])) for i in range(len(a))]

    pose_key = ['Pose']
    pose_key.extend(error_df.keys()[5:])

    pose_error_df = pd.DataFrame(pose_error, columns=pose_key)
    pose_abs_error_df = pd.DataFrame(pose_abs_error, columns=pose_key)
    pose_std_df = pd.DataFrame(pose_std, columns=pose_key)
    pose_df = pd.DataFrame(a, columns=pose_key)

    # print(pose_error_df)

    #######################################################################################################################
    '''Cloth'''
    # 1
    # 236
    # 45
    cloth_arr = [[1], [2, 3, 6], [4, 5]]
    cloth_error = []
    cloth_abs_error = []
    cloth_std = []
    for cloth in cloth_arr:
        temp = []
        temp.append(cloth)
        if len(cloth) == 1:
            temp.extend(np.round(
                np.average(error_df[((error_df.Cloth == cloth[0]))].iloc[:].values[:, 5:],
                           axis=0).astype(np.double), 2).tolist())
        elif len(cloth) == 2:
            temp.extend(np.round(np.average(
                error_df[((error_df.Cloth == cloth[0]) + (error_df.Cloth == cloth[1]))].iloc[:].values[:, 5:],
                axis=0).astype(np.double), 2).tolist())
        else:
            temp.extend(np.round(np.average(error_df[((error_df.Cloth == cloth[0]) + (error_df.Cloth == cloth[1]) + (
                        error_df.Cloth == cloth[2]))].iloc[:].values[:, 5:], axis=0).astype(np.double), 2).tolist())
        cloth_error.append(temp)

        temp = []
        temp.append(cloth)
        if len(cloth) == 1:
            temp.extend(np.round(
                np.average(np.abs(error_df[((error_df.Cloth == cloth[0]))].iloc[:].values[:, 5:]),
                           axis=0).astype(np.double), 2).tolist())
        elif len(cloth) == 2:
            temp.extend(np.round(
                np.average(np.abs(
                    error_df[((error_df.Cloth == cloth[0]) + (error_df.Cloth == cloth[1]))].iloc[:].values[:, 5:]),
                           axis=0).astype(np.double), 2).tolist())
        else:
            temp.extend(np.round(np.average(np.abs(
                error_df[
                    ((error_df.Cloth == cloth[0]) + (error_df.Cloth == cloth[1]) + (error_df.Cloth == cloth[2]))].iloc[
                :].values[:, 5:]), axis=0).astype(np.double), 2).tolist())
        cloth_abs_error.append(temp)

        temp = []
        temp.append(cloth)
        if len(cloth) == 1:
            temp.extend(np.round(
                np.std(error_df[((error_df.Cloth == cloth[0]))].iloc[:].values[:, 5:].astype(np.double),
                       axis=0), 2).tolist())
        elif len(cloth) == 2:
            temp.extend(np.round(
                np.std(error_df[((error_df.Cloth == cloth[0]) + (error_df.Cloth == cloth[1]))].iloc[:].values[:,
                       5:].astype(np.double),
                       axis=0), 2).tolist())
        else:
            temp.extend(np.round(np.std(
                error_df[
                    ((error_df.Cloth == cloth[0]) + (error_df.Cloth == cloth[1]) + (error_df.Cloth == cloth[2]))].iloc[
                :].values[:, 5:].astype(np.double), axis=0), 2).tolist())
        cloth_std.append(temp)

    cloth_error = [list(map(str, i)) for i in cloth_error]
    cloth_abs_error = [list(map(str, i)) for i in cloth_abs_error]
    cloth_std = [list(map(str, i)) for i in cloth_std]

    a = [list(map(add, cloth_error[i], slash[i])) for i in range(len(cloth_error))]
    a = [list(map(add, a[i], cloth_abs_error[i])) for i in range(len(a))]
    a = [list(map(add, a[i], slash[i])) for i in range(len(a))]
    a = [list(map(add, a[i], cloth_std[i])) for i in range(len(a))]

    cloth_key = ['Cloth']
    cloth_key.extend(error_df.keys()[5:])

    cloth_error_df = pd.DataFrame(cloth_error, columns=cloth_key)
    cloth_abs_error_df = pd.DataFrame(cloth_abs_error, columns=cloth_key)
    cloth_std_df = pd.DataFrame(cloth_std, columns=cloth_key)
    cloth_df = pd.DataFrame(a, columns=cloth_key)

    # print(cloth_error_df)

    #########################################################################################################################
    '''ANGLE'''
    angle_arr = ['0도(명치)', '45도(명치)', '90도(명치)', '135도(명치)', '180도(명치)']
    angle_error = []
    angle_abs_error = []
    angle_std = []
    for angle in angle_arr:
        try:
            temp = []
            temp.append(angle)
            temp.extend(
                np.round(np.average(error_df[error_df.Angle == angle].iloc[:].values[:, 5:], axis=0).astype(np.double),
                         2))
            angle_error.append(temp)
        except:
            continue

        try:
            temp = []
            temp.append(angle)
            temp.extend(np.round(
                np.average(np.abs(error_df[error_df.Angle == angle].iloc[:].values[:, 5:]), axis=0).astype(np.double),
                2))
            angle_abs_error.append(temp)
        except:
            continue

        try:
            temp = []
            temp.append(angle)
            temp.extend(
                np.round(np.std(error_df[error_df.Angle == angle].iloc[:].values[:, 5:].astype(np.double), axis=0), 2))
            angle_std.append(temp)
        except:
            continue

    angle_error = [list(map(str, i)) for i in angle_error]
    angle_abs_error = [list(map(str, i)) for i in angle_abs_error]
    angle_std = [list(map(str, i)) for i in angle_std]

    a = [list(map(add, angle_error[i], slash[i])) for i in range(len(angle_error))]
    a = [list(map(add, a[i], angle_abs_error[i])) for i in range(len(a))]
    a = [list(map(add, a[i], slash[i])) for i in range(len(a))]
    a = [list(map(add, a[i], angle_std[i])) for i in range(len(a))]

    angle_key = ['Angle']
    angle_key.extend(error_df.keys()[5:])

    angle_error_df = pd.DataFrame(angle_error, columns=angle_key)
    angle_abs_error_df = pd.DataFrame(angle_abs_error, columns=angle_key)
    angle_std_df = pd.DataFrame(angle_std, columns=angle_key)
    angle_df = pd.DataFrame(a, columns=angle_key)

    # print(angle_error_df)

    writer = pd.ExcelWriter(output_name)
    '''#bmi_error_df.to_excel(writer, sheet_name='bmi error')
    bmi_abs_error_df.to_excel(writer, sheet_name='bmi abs error')
    #bmi_std_df.to_excel(writer, sheet_name='bmi std')
    #pose_error_df.to_excel(writer, sheet_name='pose error')
    pose_abs_error_df.to_excel(writer, sheet_name='pose abs error')
    #pose_std_df.to_excel(writer, sheet_name='pose std')
    #cloth_error_df.to_excel(writer, sheet_name='cloth error')
    cloth_abs_error_df.to_excel(writer, sheet_name='cloth abs error')
    #cloth_std_df.to_excel(writer, sheet_name='cloth std')
    #angle_error_df.to_excel(writer, sheet_name='angle error')
    angle_abs_error_df.to_excel(writer, sheet_name='angle abs error')
    #angle_std_df.to_excel(writer, sheet_name='angle std')'''
    bmi_df.to_excel(writer, sheet_name="bmi avr,abs_avr,std")
    pose_df.to_excel(writer, sheet_name="pose avr,abs_avr,std")
    cloth_df.to_excel(writer, sheet_name="cloth avr,abs_avr,std")
    angle_df.to_excel(writer, sheet_name="angle avr,abs_avr,std")
    writer.save()



