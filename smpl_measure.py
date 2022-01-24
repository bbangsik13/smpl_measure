from copy import deepcopy
import open3d as o3d
import numpy as np
import sys
import trimesh
from smpl.serialization import load_model
import networkx as nx

landmark_female = {
    "Stature_Height": [411],
    "Cervical_Height": [828],
    "Axilla_Height": [4200],
    "Waist_Height": [3509],
    "Hip_Height": [6539],
    "Crotch_Height": [1209],
    "Knee_Height": [4533],
    "Neck_Cir": [3057, 3809, 210],#[3057, 3809, 3164]
    "Burst_Cir": [4429, 599, 3014],
    "Waist_Cir": [3509, 4118, 797],#[3509, 4118, 3502]
    "Hip_Cir": [3117, 6539, 3510],
    "Thigh_Cir":[4334, 4713, 4646], #[4334, 4419, 4646],#[4334, 4713, 4646],
    "Upper_Arm_Cir": [4917, 5012, 5364],#[4917, 5012, 6442]
    "Upper_Arm_Length": [5324, 5034,5144],# 5218],
    "Arm_Length": [5324, 5569, 5110],
    "Biacromial_Breadth_Length":[5342,1822,1306] #[5342, 683, 1213]
}

def loadSMPL(gender='female', debug=False):
    if gender == 'female':
        path = './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    elif gender == 'male':
        path = './models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    else:
        print("Unsupported Gender:", gender)
        assert (False)

    smpl = load_model(path)
    if debug:
        print("pose & beta size:", smpl.pose.size, ",", smpl.betas.size)
        print("pose :", smpl.pose)  # 24*3 = 72 (in 1-D)
        print("shape:", smpl.betas)  # 10  (check the latent meaning..)

        s, l = np.min(smpl.r[:, 0]), np.max(smpl.r[:, 0])  # x
        print("x-range (width): %.3f (%.3f,%.3f)  meter" % (l - s, s, l))
        s, l = np.min(smpl.r[:, 1]), np.max(smpl.r[:, 1])  # y
        print("y-range (tall) : %.3f (%.3f,%.3f) meter" % (l - s, s, l))
        s, l = np.min(smpl.r[:, 2]), np.max(smpl.r[:, 2])  # z
        print("z-range (thick): %.3f (%.3f,%.3f) meter" % (l - s, s, l))

    return smpl

def applyShapeFromSMPLParam(smpl, smplparam_path):
    '''
        smpl: In/Out
        smplparam_path: pkl file path for SMPL parameters

        apply only shape paramters
    '''

    #  parameter loading and apply
    import pickle
    if not isinstance(smplparam_path, dict):
        ff = pickle.load(open(smplparam_path, 'rb'), encoding="latin1")  # Python3 pickle issue
        # dd = pickle.load(open(fname_or_dict))
    else:
        ff = smplparam_path

    # apply the shape paramters
    smpl.betas[:] = ff['betas']

'''def cir_len():
    landmark = landmark_female[key]
    mesh_arr = []
    human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(T_pose_vertices)),
                                      o3d.utility.Vector3iVector(deepcopy(T_pose_faces)))
    mesh_arr.append(human.paint_uniform_color([1, 1, 1]))
    mesh_arr.append(
        o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            T_pose_vertices[landmark[0]]))
    mesh_arr.append(
        o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            T_pose_vertices[landmark[1]]))
    mesh_arr.append(
        o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            T_pose_vertices[landmark[2]]))
    o3d.visualization.draw(mesh_arr, title=key)
'''
def convex_cir(landmark,mesh,vertices,faces,title,debug=False):
    length = 0
    slice = mesh.section(plane_origin=vertices[landmark[0]],
                         plane_normal=np.cross(vertices[landmark[1]] - vertices[landmark[0]],
                                               vertices[landmark[2]] - vertices[landmark[1]]))
    node = slice.vertex_nodes
    graph = nx.Graph()
    graph.add_edges_from(node)

    length += np.linalg.norm(vertices[landmark[1]] - vertices[landmark[0]])
    #########################################################################################
    start_node = np.where(slice.vertices == vertices[landmark[1]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[2]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path) - 1):
        length += np.linalg.norm(slice.vertices[path[i + 1]] - slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1, 1, 1]))

        for slice_point in slice.vertices:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                    slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr, title=title)
    ##########################################################################################
    start_node = np.where(slice.vertices == vertices[landmark[2]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[0]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path) - 1):
        length += np.linalg.norm(slice.vertices[path[i + 1]] - slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1, 1, 1]))

        for slice_point in slice.vertices:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                    slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr, title=title)
    return length

def concave_cir(landmark,mesh,vertices,faces,title,debug=False):

    length = 0
    slice = mesh.section(plane_origin=vertices[landmark[0]],
                         plane_normal=np.cross(vertices[landmark[1]] - vertices[landmark[0]], vertices[landmark[2]] - vertices[landmark[1]]))
    node = slice.vertex_nodes
    graph = nx.Graph()
    graph.add_edges_from(node)

    start_node = np.where(slice.vertices == vertices[landmark[0]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[1]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path) - 1):
        length += np.linalg.norm(slice.vertices[path[i + 1]] - slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1,1,1]))
        for slice_point in slice.vertices:
            mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr,title=title)
    #########################################################################################
    start_node = np.where(slice.vertices == vertices[landmark[1]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[2]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path) - 1):
        length += np.linalg.norm(slice.vertices[path[i + 1]] - slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1,1,1]))

        for slice_point in slice.vertices:
            mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr,title=title)
    ##########################################################################################
    start_node = np.where(slice.vertices == vertices[landmark[2]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[0]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path) - 1):
        length += np.linalg.norm(slice.vertices[path[i + 1]] - slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1,1,1]))

        for slice_point in slice.vertices:
            mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr,title=title)

    return length

def face_len(landmark,mesh,vertices,faces,title,debug=False):
    length = 0
    slice = mesh.section(plane_origin=vertices[landmark[0]],
                         plane_normal=np.cross(vertices[landmark[1]] - vertices[landmark[0]],
                                               vertices[landmark[2]] - vertices[landmark[1]]))
    node = slice.vertex_nodes
    graph = nx.Graph()
    graph.add_edges_from(node)

    start_node = np.where(slice.vertices == vertices[landmark[0]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[2]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path)-1):
        length += np.linalg.norm(slice.vertices[path[i+1]]-slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1,1,1]))

        for slice_point in slice.vertices:
            mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr,title=title)
    ##############################################################################
    start_node = np.where(slice.vertices == vertices[landmark[2]])[0][0]
    end_node = np.where(slice.vertices == vertices[landmark[1]])[0][0]
    path = nx.shortest_path(graph, start_node, end_node)
    for i in range(len(path) - 1):
        length += np.linalg.norm(slice.vertices[path[i + 1]] - slice.vertices[path[i]])
    if debug:
        mesh_arr = []
        human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(vertices)),
                                          o3d.utility.Vector3iVector(deepcopy(faces)))
        mesh_arr.append(human.paint_uniform_color([1, 1, 1]))

        for slice_point in slice.vertices:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 0, 0]).translate(
                    slice_point))
        for path_ind in path:
            mesh_arr.append(
                o3d.geometry.TriangleMesh.create_sphere(radius=1e-2).paint_uniform_color([0, 1, 1]).translate(
                    slice.vertices[path_ind]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
            vertices[landmark[0]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
            vertices[landmark[1]]))
        mesh_arr.append(o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 0, 1]).translate(
            vertices[landmark[2]]))
        o3d.visualization.draw(mesh_arr, title=title)

    return length

def measure_correction(smpl,file_path,height,debug = False):

    applyShapeFromSMPLParam(smpl, file_path)#apply beta

    T_pose_vertices = smpl.r
    T_pose_faces = smpl.f
    T_pose_mesh = trimesh.base.Trimesh(T_pose_vertices, T_pose_faces)

    smpl.pose[16 * 3 + 2] = - 14 * np.pi / 32
    smpl.pose[17 * 3 + 2] = + 14 * np.pi / 32

    I_pose_vertices = smpl.r
    I_pose_faces = smpl.f
    I_pose_mesh = trimesh.base.Trimesh(I_pose_vertices, I_pose_faces)

    floor_landmark = np.argmin(T_pose_vertices[:,1])
    ratio = 1
    measured_dict = {}
    for key in landmark_female:
        method = key.split('_')[-1]
        landmark = landmark_female[key]
        if method == "Height":
            measure_len = T_pose_vertices[landmark[0]][1] - T_pose_vertices[floor_landmark][1]#fin
            if debug:
                mesh_arr = []
                human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(T_pose_vertices)),
                                                  o3d.utility.Vector3iVector(deepcopy(T_pose_faces)))
                mesh_arr.append(human.paint_uniform_color([1, 1, 1]))
                mesh_arr.append(
                    o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
                        T_pose_vertices[floor_landmark]))
                mesh_arr.append(
                    o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
                        T_pose_vertices[landmark[0]]))
                o3d.visualization.draw(mesh_arr,title=key)
        elif method == "Cir":
            if (key.find('Burst') == -1) and (key.find('Hip') == -1):
                measure_len = concave_cir(landmark,T_pose_mesh,T_pose_vertices,T_pose_faces,key,debug)#fin
            else:
                measure_len = convex_cir(landmark,T_pose_mesh,T_pose_vertices,T_pose_faces,key,debug)#fin
        else:
            if key.find('Arm') == -1:
                measure_len = face_len(landmark,T_pose_mesh,T_pose_vertices,T_pose_faces,key,debug)#fin?
            else:
                measure_len = face_len(landmark, T_pose_mesh, T_pose_vertices, T_pose_faces, key, debug)#fin?
        measure_len *= 100
        if key == "Stature_Height": ratio = height / measure_len
        measured_dict[key] = measure_len * ratio #np.round(measure_len * 100, 2)

    return measured_dict

def measure(smpl,file_path,debug = False):

    applyShapeFromSMPLParam(smpl, file_path)#apply beta

    T_pose_vertices = smpl.r
    T_pose_faces = smpl.f
    T_pose_mesh = trimesh.base.Trimesh(T_pose_vertices, T_pose_faces)

    smpl.pose[16 * 3 + 2] = - 14 * np.pi / 32
    smpl.pose[17 * 3 + 2] = + 14 * np.pi / 32

    I_pose_vertices = smpl.r
    I_pose_faces = smpl.f
    I_pose_mesh = trimesh.base.Trimesh(I_pose_vertices, I_pose_faces)

    floor_landmark = np.argmin(T_pose_vertices[:,1])

    measured_dict = {}
    for key in landmark_female:
        method = key.split('_')[-1]
        landmark = landmark_female[key]
        if method == "Height":
            measure_len = T_pose_vertices[landmark[0]][1] - T_pose_vertices[floor_landmark][1]#fin
            if debug:
                mesh_arr = []
                human = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(deepcopy(T_pose_vertices)),
                                                  o3d.utility.Vector3iVector(deepcopy(T_pose_faces)))
                mesh_arr.append(human.paint_uniform_color([1, 1, 1]))
                mesh_arr.append(
                    o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([1, 0, 0]).translate(
                        T_pose_vertices[floor_landmark]))
                mesh_arr.append(
                    o3d.geometry.TriangleMesh.create_sphere(radius=2e-2).paint_uniform_color([0, 1, 0]).translate(
                        T_pose_vertices[landmark[0]]))
                o3d.visualization.draw(mesh_arr,title=key)
        elif method == "Cir":
            if (key.find('Burst') == -1) and (key.find('Hip') == -1):
                measure_len = concave_cir(landmark,T_pose_mesh,T_pose_vertices,T_pose_faces,key,True)#fin
            else:
                measure_len = convex_cir(landmark,T_pose_mesh,T_pose_vertices,T_pose_faces,key,debug)#fin
        else:
            if key.find('Arm') == -1:
                measure_len = face_len(landmark,T_pose_mesh,T_pose_vertices,T_pose_faces,key,debug)#fin?
            else:
                measure_len = face_len(landmark, T_pose_mesh, T_pose_vertices, T_pose_faces, key, debug)#fin?
        measured_dict[key] = measure_len*100 #np.round(measure_len * 100, 2)

    return measured_dict

if __name__ == "__main__":
    filepath = sys.argv[1]
    debug = False
    smpl = loadSMPL(gender='female')  # load SMPL model
    measured_dict = measure(smpl,filepath,debug)
    print(measured_dict)


