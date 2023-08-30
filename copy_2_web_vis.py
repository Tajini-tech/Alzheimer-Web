import collections
import glob
import numpy as np
import csv
import pandas as pd
import nibabel as nib
#import scipy
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from skimage import measure
from skimage.transform import resize
import plotly.graph_objects as go
from plotly.graph_objs import Layout
import plotly.express as px
from matplotlib import cm
import matplotlib.colors
import matplotlib.cm
from matplotlib.cm import ScalarMappable
from keras.layers import Input, Dense, Flatten, Dropout, merge, Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
import innvestigate
import innvestigate.utils as iutils

####################################################################################
input_img = Input(shape=(105, 105, 1))
def get_model(input_img):
    conv1 = Conv2D(5, (3, 3), padding='same')(input_img)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(5, (3, 3), padding='same')(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(5, (3, 3), padding='same')(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv3 = Flatten()(conv3)
    conv3 = Dropout(rate=0.1)(conv3)
    dense = Dense(64, activation='relu')(conv3)
    dense = Dropout(rate=0.1)(dense)
    dense = Dense(32, activation='relu')(dense)
    dense = Dropout(rate=0.1)(dense)
    dense = Dense(2, activation='softmax')(dense)
    return dense
model = Model(input_img, get_model(input_img))
print(model.summary())

src_path_test = r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\WMModel\testimages"
p = r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\Freitag_weights\_ep  11 in fo  1 eval_acc 0.800818 eval_loss 0.427652.h5"
p_sagital = r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\Freitag_weights\Sagital_ep  19 in fo  1 eval_acc 0.840905 eval_loss 0.353080.h5"
p_axial = r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\Freitag_weights\Axial_ep  19 in fo  1 eval_acc 0.850267 eval_loss 0.336413.h5"
T = ["basal ganglia","cerebellum", "frontal", "hippocampus","insula cingulate","occipital", "parietal" ,"temporal"
    , "brainmask"]
T_vis=["Brain_slicing","Saliency_slicing","3d_Scatter"]
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
external_stylesheets=[dbc.themes.BOOTSTRAP]
app = Dash(__name__,external_stylesheets=external_stylesheets)
def Load_Data(src):
    d_Test = list()
    for files in glob.glob(src + "/*.nii"):
        image = nib.load(files)
        img = image.get_fdata().astype(np.float32)
        img = (img - img.min()) / img.max()
        IMG_PX_SIZE = 105
        img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
        d_Test.append(img)
    return np.asarray(d_Test)
d_Test=Load_Data(src_path_test)
def get_data_axes(arr,k,add_dimension=True):
    arr_ax = list()
    for i in range(arr.shape[0]):
        for j in range(20, 85):
            arr_ax.append(np.take(arr[i], j, k))
    arr_ax = np.asarray(arr_ax)
    if add_dimension:
        arr_ax = np.reshape(arr_ax, (len(arr_ax), arr_ax[0].shape[0], arr_ax[0].shape[1], 1))
    else:
        arr_ax = np.reshape(arr_ax, (len(arr_ax), arr_ax[0].shape[0], arr_ax[0].shape[1]))
    return arr_ax

x_coronal= get_data_axes(d_Test,1)# np.take(arr,slices[20:85],1)
x_sagital = get_data_axes(d_Test,0)
x_axial = get_data_axes(d_Test,2)

print("number of subjects is  :", x_coronal.shape[0] / 65)
####################################################
src_path_mask =r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\Hippocampus_masks"

def Load_Masks(src):
    data_overlay = list()
    for files in glob.glob(src + "/*.nii"):
        image = nib.load(files)
        img = image.get_fdata().astype(np.float32)
        IMG_PX_SIZE = 105
        img = resize(img, (IMG_PX_SIZE, IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True)
        data_overlay.append(img)
    data_overlay = np.asarray(data_overlay)
    data_overlay = np.nan_to_num(data_overlay)
    return data_overlay
s=Load_Masks(src_path_mask)
data_overlay=list()
for i in range (s.shape[0]):
    data_overlay.append(s[i]*d_Test[0])
data_overlay=np.asarray(data_overlay)
x_mask_coronal = get_data_axes(data_overlay,1,add_dimension=False)
x_mask_sagital = get_data_axes(data_overlay,0,False)
x_mask_axial = get_data_axes(data_overlay,2,False)
num_mask = x_mask_coronal.shape[0] // 65

def apply_threshold(arr, t=3):
    arr[arr<t*0.1]=0
    return arr
##################################################################
def print_prediction(path, slices):
    model.load_weights(path)
    pre = (model.predict(slices) > 0.5).astype(int)
    elements_count = collections.Counter(pre[:, 0])
    NC_ratio = elements_count.get(0)
    AD_ratio = elements_count.get(1)
    if AD_ratio is None:
        return("NC probability : 1")
    if NC_ratio is None:
        return("AD probability : 1")
    if ((AD_ratio is not None) and (NC_ratio is not None)):
        if (NC_ratio > AD_ratio):
            return ("NC probability : %f"%( NC_ratio / (NC_ratio + AD_ratio)))
        else:
            return ("AD probability : %f"%( AD_ratio / (AD_ratio + NC_ratio)))
####################################################################################
print_prediction(p,x_coronal)
print_prediction(p_sagital,x_sagital)
print_prediction(p_axial,x_axial)
def ratio_relevance_volume(rel,vol,k):
  rel=rel[k]
  vol=vol[65*k:65*(k+1)]
  vol[vol>0]=1
  sum_rel=rel.sum()
  sum_vol=vol.sum()
  print("ratio",sum_rel/sum_vol)
  return sum_rel / sum_vol
def get_area_volume(vol,k):
    vol = vol[65 * k:65 * (k + 1)]
    vol[vol > 0] = 1
    return vol.sum()
def map_rescaling(map, rescale_threshold=8):
    a = ndi.filters.gaussian_filter(map, sigma=0.8)
    scale = np.quantile(np.absolute(a), 0.99)
    a = (a / scale)
    a[a >= rescale_threshold] = rescale_threshold
    a = a / rescale_threshold
    return a
def get_scores(path, slices, x_mask,text,rt=8):
    src_path_csv = r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\relevance_maps_{}.csv".format(str(text))
    pos_rel = list()
    neg_rel = list()
    mask_map=list()
    model.load_weights(path)
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    methods = [("lrp.sequential_preset_a", {"neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0"), ]
    # methods = [("lrp.alpha_1_beta_0", {"neuron_selection_mode": "index"}, "LRP-alpha1beta0"), ]
    # methods=[("lrp.epsilon",          {"epsilon": 1},          "LRP-epsilon"),]
    analyzers = []
    for method in methods:#if len(method)>1
        analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
        analyzers.append(analyzer)
    for method, analyzer in zip(methods, analyzers):
        a = np.reshape(analyzer.analyze(slices, neuron_selection=1), slices.shape[0:3])
        # a = np.reshape(analyzer.analyze(slices), slices.shape[0:3])
        a_pos = np.copy(a)
        a_neg = np.copy(a)
        a_pos[a_pos<0]=0
        a_neg[a_neg > 0] = 0
        A = map_rescaling(a_pos, rt)
        for i in range(num_mask):
            mask_map.append(A * x_mask[65 * i:65 * (i + 1)])
            neg_rel.append(a_neg * x_mask[65 * i:65 * (i + 1)])
            pos_rel.append((a_pos * x_mask[65 * i:65 * (i + 1)]))
        mask_map.append(A)
        neg_rel.append(a_neg)
        pos_rel.append(a_pos)
        mask_map=np.asarray(mask_map)
        neg_rel = np.asarray(neg_rel)
        pos_rel = np.asarray(pos_rel)
        with open(src_path_csv,"w") as csvfile:
            fieldnames = ["slice", 'basal ganglia', 'cerebellum', "frontal", "hippocampus", "insula cingulate", "occipital", "parietal",
                          "temporal", "brainmask", "all"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(65):
                writer.writerow(
                    {"slice": i, 'basal ganglia': np.sum(pos_rel[0][i]), 'cerebellum': np.sum(pos_rel[1][i]),
                     "frontal": np.sum(pos_rel[2][i]), "hippocampus": np.sum(pos_rel[3][i]),
                     "insula cingulate": np.sum(pos_rel[4][i]),
                     "occipital": np.sum(pos_rel[5][i]), "parietal": np.sum(pos_rel[6][i]),
                     "temporal": np.sum(pos_rel[7][i]), "brainmask": np.sum(pos_rel[8][i]),
                     "all": np.sum(a_pos[i])
                     })
    return neg_rel, a,src_path_csv,pos_rel,mask_map


def get_mean_sum_rel(scores_coronal,scores_sagital,scores_axial,text=""):

    mean_relevance= r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\mean_relevance {}".format(str(text))
    with open(mean_relevance, "w") as csvfile:
        fieldnames = [ 'basal ganglia', 'cerebellum', "frontal", "hippocampus", "insula cingulate", "occipital",
                      "parietal",
                      "temporal", "brainmask"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({ 'basal ganglia': (np.sum(scores_coronal[0])/get_area_volume(x_mask_coronal,0)+np.sum(scores_sagital[0])/get_area_volume(x_mask_coronal,0)+np.sum(scores_axial[0])/get_area_volume(x_mask_coronal,0))/3,
                           'cerebellum': (np.sum(scores_coronal[1]/get_area_volume(x_mask_coronal,1))+np.sum(scores_sagital[1])/get_area_volume(x_mask_coronal,1)+np.sum(scores_axial[1])/get_area_volume(x_mask_coronal,1))/3,
                           "frontal": (np.sum(scores_coronal[2])/get_area_volume(x_mask_coronal,2)+np.sum(scores_sagital[2])/get_area_volume(x_mask_coronal,2)+np.sum(scores_axial[2])/get_area_volume(x_mask_coronal,2))/3,
                           "hippocampus":(np.sum(scores_coronal[3])/get_area_volume(x_mask_coronal,3)+np.sum(scores_sagital[3])/get_area_volume(x_mask_coronal,3)+np.sum(scores_axial[3])/get_area_volume(x_mask_coronal,3))/3,
                           "insula cingulate": (np.sum(scores_coronal[4])/get_area_volume(x_mask_coronal,4)+ np.sum(scores_sagital[4])/get_area_volume(x_mask_coronal,4)+np.sum(scores_axial[4])/get_area_volume(x_mask_coronal,4))/3,
                           "occipital": (np.sum(scores_coronal[5])/get_area_volume(x_mask_coronal,5)+ np.sum(scores_sagital[5])/get_area_volume(x_mask_coronal,5)+
                                                       np.sum(scores_axial[5])/get_area_volume(x_mask_coronal,5))/3,
                           "parietal": (np.sum(scores_coronal[6])/get_area_volume(x_mask_coronal,6)+ np.sum(scores_sagital[6])/get_area_volume(x_mask_coronal,6)+
                                                       np.sum(scores_axial[6])/get_area_volume(x_mask_coronal,6))/3,
                           "temporal": (np.sum(scores_coronal[7])/get_area_volume(x_mask_coronal,7)+ np.sum(scores_sagital[7])/get_area_volume(x_mask_coronal,7)+
                                                       np.sum(scores_axial[7])/get_area_volume(x_mask_coronal,7))/3,
                           "brainmask": (np.sum(scores_coronal[8])/get_area_volume(x_mask_coronal,8)
                                               + np.sum(scores_sagital[8])/get_area_volume(x_mask_coronal,8)+
                                                       np.sum(scores_axial[8])/get_area_volume(x_mask_coronal,8))/3})
    df = pd.read_csv(mean_relevance, delimiter=",")
    layout = Layout(paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                    )
    fig = go.Figure(data=[go.Bar(
        x=fieldnames, y =df.loc[0],
        text=df.values,
        textposition='auto',
    )],layout=layout)
    return fig
x1 = np.ones(105)
y1 = np.linspace(0, 104, 105)
z1 = np.linspace(0, 104, 105)
z_plane_pos = np.ones((105, 105))
def two_d_vis(relevance_map, original_data,t=0 ,k=8,title=""):
    relevance_map=np.copy(relevance_map)
    relevance_map = relevance_map[k]
    fig = go.Figure()
    colorscale=[[0, 'rgba' + "(255,255,255,0)"],
                [0.13, 'rgb' + "(166,217,106)"],
                [0.25, 'rgb' + "(102, 189, 99)"],
                [0.38, 'rgb' + "(26, 152, 80)"],
                [0.50, 'rgb' + "(254, 224, 139)"],
                [0.62, 'rgb' + "(253, 174, 97)"],
                [0.75, 'rgb' + "(244, 109, 67)"],
                [0.88, 'rgb' + "(215, 48, 39)"],
                [1, 'rgb' + "(88, 19, 16)"]
                ]
    pil_img = apply_threshold(relevance_map[40], t)
    fig.add_traces([
        go.Surface(x=x1, y=y1, z=np.array([z1] * 105), surfacecolor=original_data[40],
                   colorscale='Gray', cmin=200, cmax=200, showscale=False),
        go.Surface(x=2*x1, y=y1, z=np.array([z1] * 105),opacity=0.7,surfacecolor=pil_img,
                   colorscale=colorscale,cmin=0,cmax=1
                   ,showscale=True)])
    frames = []
    for row in range(0, 65):
        pil_img = apply_threshold(relevance_map[row], t)
        frames.append(
            go.Frame(
                name=str(row),
                data=[
                    go.Surface(x=row*x1, y=y1, z=np.array([z1] * 105), surfacecolor=original_data[row]
                               , colorscale='Gray', cmin=200, cmax=200, showscale=False),
                    go.Surface(x=(2+row)*x1, y=y1, z=np.array([z1] * 105),
                                opacity=0.7,colorscale=colorscale,surfacecolor=pil_img,cmin=0,cmax=1,
                               showscale=True),
                ],
            )
        )
    figa = go.Figure(data=fig.data, frames=frames, layout=fig.layout)
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "slice="},
            "len": 1,
            "steps": [
                {
                    "args": [
                        [fr.name],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "fromcurrent": True,
                        },
                    ],
                    "label":fr.name,
                    "method": "animate",
                }
                for fr in figa.frames
            ],
        }
    ]
    figa.update_layout(title=title,
        #width=480, height=480,
                       autosize=True,template="plotly_dark",
        scene=dict(xaxis=dict(showticklabels=False, range=[0,67]),#, range=[0, 105]
                   yaxis=dict(showticklabels=False),
                   zaxis=dict(showticklabels=False),
                   ),
        sliders=sliders,
    )
    figa['layout']['sliders'][0]['activebgcolor'] = "darkgoldenrod"
    figa['layout']['sliders'][0]['bgcolor'] = "blue"
    figa['layout']['sliders'][0]["minorticklen"] = 0
    return figa
def three_d_vis(relevance_map, original_data,t=0,k=8):
    fig = go.Figure()
    relevance_map=np.copy(relevance_map)
    relevance_map=relevance_map[k]
    colorscale = [[0, 'rgba' + "(255,255,255,0)"],
                  [0.13, 'rgb' + "(166,217,106)"],
                  [0.25, 'rgb' + "(102, 189, 99)"],
                  [0.38, 'rgb' + "(26, 152, 80)"],
                  [0.50, 'rgb' + "(254, 224, 139)"],
                  [0.62, 'rgb' + "(253, 174, 97)"],
                  [0.75, 'rgb' + "(244, 109, 67)"],
                  [0.88, 'rgb' + "(215, 48, 39)"],
                  [1, 'rgb' + "(88, 19, 16)"]]
    heatmap_threshold = apply_threshold(relevance_map[20], t)
    v0, f0 = marching_cubes(original_data[20:30], 0.09, 1)
    fig.add_traces([
        make_mesh_3d_colorscale(v0, f0,opacity=0.5),
        go.Surface(x=x1,y=y1,z=np.array([z1] * 105),
                   surfacecolor=heatmap_threshold, colorscale=colorscale,cmin=0,cmax=1, opacity=0.7,
                   showscale=True)])
    frames = []
    for row in range(0, 63):
        heatmap_threshold = apply_threshold(relevance_map[row], t)

        if row + 10 < 65:
            v, f = marching_cubes(original_data[row:row + 10], 0.09, 1)
        else:
            v, f = marching_cubes(original_data[row:64], 0.09, 1)
        frames.append(
            go.Frame(
                name=str(row),
                data=[
                    make_mesh_3d_colorscale(apply_trans(translation(row,0,0),v), f,opacity=0.5),
                    go.Surface(x=row+x1,y=y1,z=np.array([z1] * 105),
                               surfacecolor=heatmap_threshold, colorscale=colorscale,cmin=0,cmax=1,opacity=0.7,
                               showscale=True),
                ],
            )
        )
    figa = go.Figure(data=fig.data, frames=frames, layout=fig.layout)
    figa.update_layout(
                   template="plotly_dark",
                   autosize=True,
                       scene=dict(xaxis=dict(range=[0,65],autorange=False),
                                  yaxis=dict(range=[0,105],autorange=False),
                                  zaxis=dict(range=[0,105],autorange=False),
                                  aspectratio=dict(x=1, y=1, z=1),
                                  camera_eye=dict(x=1.45, y=1.45, z=0.5)
                                  ),
                       sliders=[
                           {
                               "active": 0,
                               "currentvalue": {"prefix": "slice="},
                               "len": 1,
                               "steps": [
                                   {
                                       "args": [
                                           [fr.name],
                                           {
                                               "frame": {"duration": 0, "redraw": True},
                                               "mode": "immediate",
                                               "fromcurrent": True,
                                           },
                                       ],
                                       "label": fr.name,
                                       "method": "animate",
                                   }
                                   for fr in figa.frames
                               ],
                           }
                       ], )
    figa['layout']['sliders'][0]['activebgcolor'] = "darkgoldenrod"
    figa['layout']['sliders'][0]['bgcolor'] = "blue"
    figa['layout']['sliders'][0]["minorticklen"] = 0
    return figa
def marching_cubes(arr, threshold=0.09, step_size=1):
    verts, faces, norm, val = measure.marching_cubes(arr, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces
def triangles(verts,faces):
    return verts[faces]
def area_triangles(triangles):
    areas = []
    for t in triangles:
        vectors = np.diff(t, axis=0)
        cross = np.cross(vectors[0], vectors[1])
        area =np.linalg.norm(cross)/2
        areas.append(area)
    return np.array(areas)
def angles_and_edges_lengths(triangles):
    vectors = np.diff(triangles, axis=1)
    lengths = np.linalg.norm(vectors, axis=2)
    dot_products = np.einsum('...i,...i', vectors[:, 0], vectors[:, 1])
    cos_angles = dot_products / (lengths[:, 0] * lengths[:, 1])
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles)
    degenerate_triangles = np.where(np.isclose(lengths, 0, atol=1e-6).any(axis=1))[0]
    angles[degenerate_triangles] = 0
    return np.degrees(angles), lengths
def get_vertex_neighbors(vertices, faces):
    vertex_neighbors = [[] for i in range(vertices.shape[0])]
    for face in faces:
        vertex_neighbors[face[0]].append(face[1])
        vertex_neighbors[face[0]].append(face[2])
        vertex_neighbors[face[1]].append(face[0])
        vertex_neighbors[face[1]].append(face[2])
        vertex_neighbors[face[2]].append(face[0])
        vertex_neighbors[face[2]].append(face[1])
    return vertex_neighbors
def laplacian_calculation(v,f):
    neighbors = get_vertex_neighbors(v,f)
    vertices = v
    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbors)])
    data = np.concatenate([[1.0 / len(n)] * len(n)# avg of positions of neighbors
                               for n in neighbors])
    matrix = coo_matrix((data, (row, col)),
                        shape=[len(vertices)] * 2)
    return matrix

def filter_laplacian_taubin(v,f,
                     lamb=0.5,mu=0.5,
                     iterations=10,filter=""):
    v=np.copy(v)
    f = np.copy(f)

    laplacian_operator = laplacian_calculation(v,f)
    vertices = v
    # Number of passes
    for _index in range(iterations):
        if filter=="taubin":
            dot = laplacian_operator.dot(vertices) - vertices
            if _index % 2 == 0:
                vertices += lamb * dot#shrink
            else:
                vertices -= mu * dot#inflate
        else:
            dot = laplacian_operator.dot(vertices) - vertices
            vertices += lamb * dot
    verts = vertices
    return verts
def draw_wireframe(v,f):
    Xe = []
    Ye = []
    Ze = []
    tri_points = v[f]
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])
    # define the trace for triangle sides
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color='rgb(70,70,70)', width=1))
    return lines

def make_mesh_3d(verts, faces, color='rgb(236, 236, 212)',opacity=0.1):

    vert_x, vert_y, vert_z = verts[:, :3].T
    face_i, face_j, face_k = faces.T
    data = go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,
                      opacity=opacity, name='', showscale=False, hoverinfo='none',
                     color=color,lighting=dict(ambient=1, diffuse=0.1))
    return data
def make_mesh_3d_colorscale(verts, faces,opacity=0.7):
    colorscale = [[0.0, 'rgb(253, 225, 197)'],
                  [0.1, 'rgb(253, 216, 179)'],
                  [0.2, 'rgb(253, 207, 161)'],
                  [0.3, 'rgb(253, 194, 140)'],
                  [0.4, 'rgb(253, 181, 118)'],
                  [0.5, 'rgb(253, 167, 97)'],
                  [0.6, 'rgb(253, 153, 78)'],
                  [0.7, 'rgb(252, 140, 59)'],
                  [0.8, 'rgb(248, 126, 43)'],
                  [0.9, 'rgb(243, 112, 27)'],
                  [1.0, 'rgb(236, 98, 15)']]
    vert_x, vert_y, vert_z = verts[:, :3].T
    face_i, face_j, face_k = faces.T
    data = go.Mesh3d(x=vert_x, y=vert_y, z=vert_z, i=face_i, j=face_j, k=face_k,intensity=vert_z,
                     colorscale=colorscale, opacity=opacity, name='', showscale=False, hoverinfo="none",
                     flatshading=True,
                     lighting=dict(ambient=0.18,
                                   diffuse=1,
                                   fresnel=0.1,
                                   specular=1,
                                   roughness=0.05,
                                   facenormalsepsilon=1e-8,
                                   vertexnormalsepsilon=1e-15),
                     lightposition=dict(x=100,y=200,z=0))
    return data

def translation(x=0, y=0, z=0):
    m = np.array([[1, 0, 0, x],
                  [0, 1, 0, y],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return m
tr=translation(20,0,0)
def apply_trans(trans, pts, move=True):
    out_pts = np.dot(pts, trans[:3, :3].T)
    if move:
        out_pts += trans[:3, 3]
    return out_pts

def three_d_vis_mask(relevance_map, original_data, k=8,scale=3,t="rgb(236, 236, 212)",text="" ):
    relevance_map = np.copy(relevance_map)
    relevance_map = relevance_map[k]
    relevance_map[relevance_map < (scale*0.1)] = 0
    verts, faces = marching_cubes(original_data)
    zero_pad_left = np.zeros((20, 105, 105))
    zero_pad_right = np.zeros((20, 105, 105))
    # Concatenate the original array with the left and right zero padding
    norm_volume_of_saliency = np.concatenate((zero_pad_left, relevance_map, zero_pad_right), axis=0)
    highlight_coordinates = np.argwhere(norm_volume_of_saliency)
    data = make_mesh_3d(verts, faces)
    tree = cKDTree(verts)
    _, nearest_indices = tree.query(highlight_coordinates, k=1)

    # Define the colorscale
    colorscale = [
        [0.0, "#FFFFFF00"],
        [0.13, "#A6D96A"],
        [0.25, "#66BD63"],
        [0.38, "#1A9850"],
        [0.50, "#FEE08B"],
        [0.62, "#FDAE61"],
        [0.75, "#F46D43"],
        [0.88, "#D73027"],
        [1.0, "#581B10"]
    ]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colorscale)
    sm = ScalarMappable(cmap=cmap)
    rgba_colors = sm.to_rgba(
        norm_volume_of_saliency[highlight_coordinates[:, 0], highlight_coordinates[:, 1], highlight_coordinates[:, 2]])

    vertex_colors = np.array([[10, 10, 10, 5]] * len(verts))
    for i, vertex_index in enumerate(nearest_indices):
        vertex_colors[vertex_index] = np.array(rgba_colors[i] * 255, dtype=int)
    highlighted_verts = verts[nearest_indices]
    light_position = np.mean(highlighted_verts, axis=0)
    print("light position",light_position[0], light_position[1], light_position[2])
    #data.update(vertexcolor=vertex_colors.tolist(), opacity=1,
    #            lightposition=dict(x=light_position[0], y=light_position[1], z=light_position[2]),
    #            )
        
    data.update(vertexcolor=vertex_colors.tolist(), opacity=1,
                #lightposition=dict(x=52.257698, y=54.122665, z=42.303246),
                )
    fig = go.Figure(data=[data, make_mesh_3d(verts, faces, opacity=0.02,).update(lighting=dict(#ambient=0.18,
                                   ambient=1,
                                   diffuse=1,
                                   facenormalsepsilon=1e-8,
                                   vertexnormalsepsilon=1e-15),intensity=verts[:,:3].T[2],colorscale="Blues")])#colorscale=colorscale_mesh,#np.linspace(0, 1, len(verts))# # Intensity of each vertex, which will be interpolated and color-coded
   
    frames = []
    for i in range(0, 105, 1):
        frame = dict(layout=dict(scene=dict(xaxis=dict(range=[i, 105 + i], autorange=False),
                                            yaxis=dict(range=[0, 105], autorange=False),
                                            zaxis=dict(range=[0, 105], autorange=False)
                                            , aspectratio=dict(x=1, y=1, z=1))), name=f"frame{i}")
        frames.append(frame)
    fig.frames = frames

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "slice="},
            "len": 1,
            "steps": [
                {
                    "args": [
                        [fr.name],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "fromcurrent": True,
                        },
                    ],
                    "label": fr.name,
                    "method": "animate",
                }
                for fr in fig.frames
            ],
        }
    ]
    fig.update_layout(autosize=True, template="plotly_dark",title=text,
                       # width=480, height=480,

                       sliders=sliders, )
    fig['layout']['sliders'][0]['activebgcolor'] = "darkgoldenrod"
    fig['layout']['sliders'][0]['bgcolor'] = "blue"
    fig['layout']['sliders'][0]["minorticklen"] = 0
    #fig.update_layout(scene=dict(camera=dict(eye=dict(x=light_position[0], y=light_position[1], z=light_position[2]))))

    return fig

def saliency_slicing(relevance_map, original_data, k=8,scale=3,t="rgb(236, 236, 212)",text="" ):
    relevance_map = np.copy(relevance_map)
    relevance_map = relevance_map[k]
    relevance_map[relevance_map < (scale*0.1)] = 0
    verts, faces = marching_cubes(original_data)

    zero_pad_left = np.zeros((20, 105, 105))
    zero_pad_right = np.zeros((20, 105, 105))
    norm_volume_of_saliency = np.concatenate((zero_pad_left, relevance_map, zero_pad_right), axis=0)
    colorscale = [[0, 'rgba' + "(255,255,255,0)"],
                  [0.13, 'rgb' + "(166,217,106)"],
                  [0.25, 'rgb' + "(102, 189, 99)"],
                  [0.38, 'rgb' + "(26, 152, 80)"],
                  [0.50, 'rgb' + "(254, 224, 139)"],
                  [0.62, 'rgb' + "(253, 174, 97)"],
                  [0.75, 'rgb' + "(244, 109, 67)"],
                  [0.88, 'rgb' + "(215, 48, 39)"],
                  [1, 'rgb' + "(88, 19, 16)"]]
    fig = go.Figure(data= make_mesh_3d(verts, faces, opacity=0.09,).update(lighting=dict(ambient=0.18,
                                   diffuse=1,
                                   facenormalsepsilon=1e-8,
                                   vertexnormalsepsilon=1e-15),intensity=verts[:,:3].T[2],colorscale="Blues",
                                   lightposition=dict(x=52.257698, y=54.122665, z=42.303246)))
    for i in range(105):
        fig.add_traces(go.Surface(x=x1+i,y=y1,z=np.array([z1] * 105),surfacecolor=norm_volume_of_saliency[i],
                                  colorscale=colorscale,cmin=0,cmax=1, opacity=0.7,
                   showscale=True,visible=False))
    steps = []
    for i in range(105):
        # Hide all traces
        step = dict(
            method='restyle',
            args=['visible', [False] * len(fig.data)],
        )
        step['args'][1][i] = True
        step['args'][1][0] = True
        steps.append(step)

    sliders = [dict(
        steps=steps,
    )]
    fig.update_layout(autosize=True, template="plotly_dark",title=text,
                       # width=480, height=480,

                       sliders=sliders, )
    fig['layout']['sliders'][0]['activebgcolor'] = "darkgoldenrod"
    fig['layout']['sliders'][0]['bgcolor'] = "blue"
    fig['layout']['sliders'][0]["minorticklen"] = 0
    return fig
def saliency_scatter(relevance_map, original_data, k=8,scale=3,t="rgb(236, 236, 212)",text="" ):
    relevance_map = np.copy(relevance_map)
    relevance_map = relevance_map[k]
    relevance_map[relevance_map < (scale * 0.1)] = 0
    relevance_map[relevance_map<0.3]=0
    verts, faces = marching_cubes(original_data)

    zero_pad_left = np.zeros((20, 105, 105))
    zero_pad_right = np.zeros((20, 105, 105))
    result = np.concatenate((zero_pad_left, relevance_map, zero_pad_right), axis=0)
    highlight_coordinates = np.argwhere(result)
    fig = px.scatter_3d(x=highlight_coordinates[:, 0], y=highlight_coordinates[:, 1], z=highlight_coordinates[:, 2],
                        color=result[highlight_coordinates[:, 0].astype(int),
                                           highlight_coordinates[:, 1].astype(int),
                                           highlight_coordinates[:, 2].astype(int)],
                        color_continuous_scale=["green", "yellow", "red"],
                        color_continuous_midpoint=0.5,
                        )
    #fig.update_traces(opacity=1)
    fig.update_traces(marker=dict(size=1), opacity=1)
    figa = go.Figure(data=fig.data, layout=fig.layout)
    figa.add_traces(data= make_mesh_3d(verts, faces, opacity=0.09,).update(lighting=dict(ambient=0.18,
                                   diffuse=1,
                                   facenormalsepsilon=1e-8,
                                   vertexnormalsepsilon=1e-15),intensity=verts[:,:3].T[2],flatshading=True,
                                   lightposition=dict(x=52.257698, y=54.122665, z=42.303246)))
    frames = []
    for i in range(0, 105, 1):
        frame = dict(layout=dict(scene=dict(xaxis=dict(range=[i, 105 + i], autorange=False),
                                            yaxis=dict(range=[0, 105], autorange=False),
                                            zaxis=dict(range=[0, 105], autorange=False)
                                            , aspectratio=dict(x=1, y=1, z=1))), name=f"frame{i}")

        frames.append(frame)
    figa.frames = frames

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "slice="},
            "len": 1,
            "steps": [
                {
                    "args": [
                        [fr.name],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "fromcurrent": True,
                        },
                    ],
                    "label": fr.name,
                    "method": "animate",
                }
                for fr in figa.frames
            ],
        }
    ]
    figa.update_layout(autosize=True, template="plotly_dark", title=text,
                      # width=480, height=480,
                      sliders=sliders, )
    figa['layout']['sliders'][0]['activebgcolor'] = "darkgoldenrod"
    figa['layout']['sliders'][0]['bgcolor'] = "blue"
    figa['layout']['sliders'][0]["minorticklen"] = 0
    return figa
def plot_dataframe_mask(dataframe_coronal,dataframe_sagital,dataframe_axial,t,scale=3):
    dataframe_coronal_copy=np.copy(dataframe_coronal)
    dataframe_sagital_copy=np.copy(dataframe_sagital)
    dataframe_axial_copy=np.copy(dataframe_axial)
    dataframe_coronal_copy[dataframe_coronal_copy<scale*0.1]=0
    dataframe_sagital_copy[dataframe_sagital_copy<scale*0.1]=0
    dataframe_axial_copy[dataframe_axial_copy<scale*0.1]=0
    df_coronal = list()
    df_sagital = list()
    df_axial=list()
    x = np.arange(65)
    colors = px.colors.qualitative.Plotly
    layout=Layout(paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
    for i in range(65):
        df_coronal.append(np.sum(dataframe_coronal_copy[t][i]))
        df_sagital.append(np.sum(dataframe_sagital_copy[t][i]))
        df_axial.append(np.sum(dataframe_axial_copy[t][i]))
    fig = go.Figure(
        data=[go.Scatter(x=x, y=df_coronal, mode='lines',name="coronal" ,line=dict(color=colors[0])),
              go.Scatter(x=x, y=df_sagital, mode='lines',name="sagital", line=dict(color=colors[1])),
              go.Scatter(x=x, y=df_axial, mode='lines',name="axial", line=dict(color=colors[2]))],
        layout=layout)
    return fig

def mixed_vis(relevance_map, original_data_3d, original_data_2d ,k=8,scale=3,radio="slicing on"):
    relevance_map=np.copy(relevance_map)
    colorscale = [[0, 'rgba' + "(255,255,255,0)"],
                  [0.13, 'rgb' + "(166,217,106)"],
                  [0.25, 'rgb' + "(102, 189, 99)"],
                  [0.38, 'rgb' + "(26, 152, 80)"],
                  [0.50, 'rgb' + "(254, 224, 139)"],
                  [0.62, 'rgb' + "(253, 174, 97)"],
                  [0.75, 'rgb' + "(244, 109, 67)"],
                  [0.88, 'rgb' + "(215, 48, 39)"],
                  [1, 'rgb' + "(88, 19, 16)"]
                  ]
    colorscale_mesh = [[0.0, 'rgb(253, 225, 197)'],
                  [0.1, 'rgb(253, 216, 179)'],
                  [0.2, 'rgb(253, 207, 161)'],
                  [0.3, 'rgb(253, 194, 140)'],
                  [0.4, 'rgb(253, 181, 118)'],
                  [0.5, 'rgb(253, 167, 97)'],
                  [0.6, 'rgb(253, 153, 78)'],
                  [0.7, 'rgb(252, 140, 59)'],
                  [0.8, 'rgb(248, 126, 43)'],
                  [0.9, 'rgb(243, 112, 27)'],
                  [1.0, 'rgb(236, 98, 15)']]
    relevance_map = relevance_map[k]
    fig = go.Figure()
    ve, fa = marching_cubes(original_data_3d)
    relevance_map[relevance_map < 0.2] = 0
    zero_pad_left = np.zeros((20, 105, 105))
    zero_pad_right = np.zeros((20, 105, 105))
    relevance_map = np.concatenate((zero_pad_left, relevance_map, zero_pad_right), axis=0)
    fig.add_traces([
        go.Surface(x=2*x1, y=y1, z=np.array([z1] * 105), surfacecolor=original_data_2d[0],
                   colorscale='Gray', cmin=200, cmax=200, showscale=False),
        go.Surface(x=x1, y=y1, z=np.array([z1] * 105), opacity=0.7, surfacecolor=relevance_map[0],
                   colorscale=colorscale, cmin=0, cmax=1
                   , showscale=True),
        make_mesh_3d(ve, fa, opacity=0.4, ).update(lighting=dict(ambient=0.68,
                                                                        diffuse=0.8,
                                                                        facenormalsepsilon=1e-8,
                                                                        vertexnormalsepsilon=1e-15),colorscale=colorscale_mesh,
                                                          intensity=ve[:, :3].T[2],)])
    frames = []
    if radio=="slicing on":
        for row in range(0, 105):
            frames.append(
                go.Frame(
                    name=str(row),
                    data=[
                        go.Surface(x=(row+2) * x1, y=y1, z=np.array([z1] * 105), surfacecolor=original_data_2d[row]
                                   , colorscale='Gray', cmin=200, cmax=200, showscale=False),
                        go.Surface(x=(row+1) * x1, y=y1, z=np.array([z1] * 105), opacity=0.7, surfacecolor=relevance_map[row],
                                   colorscale=colorscale, cmin=0, cmax=1
                                   , showscale=True)
                    ],layout=dict(scene=dict(xaxis=dict(range=[row, 105 + row], autorange=False),
                                                yaxis=dict(range=[0, 105], autorange=False),
                                                zaxis=dict(range=[0, 105], autorange=False)
                                                , aspectratio=dict(x=1, y=1, z=1)))
                ))
    elif radio=="slicing off":
        for row in range(0, 105):
            frames.append(
                go.Frame(
                    name=str(row),
                    data=[
                        go.Surface(x=(row + 2) * x1, y=y1, z=np.array([z1] * 105), surfacecolor=original_data_2d[row]
                                   , colorscale='Gray', cmin=200, cmax=200, showscale=False),
                        go.Surface(x=(row + 1) * x1, y=y1, z=np.array([z1] * 105), opacity=0.7,
                                   surfacecolor=relevance_map[row],
                                   colorscale=colorscale, cmin=0, cmax=1
                                   , showscale=True)
                    ],
                ))
    figa = go.Figure(data=fig.data, frames=frames)
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "slice="},
            "len": 1,
            "steps": [
                {
                    "args": [
                        [fr.name],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "fromcurrent": True,
                        },
                    ],
                    "label": fr.name,
                    "method": "animate",
                }
                for fr in figa.frames
            ],
        }
    ]
    figa.update_layout(autosize=True,template="plotly_dark",
        #width=480, height=480,
        scene=dict(
            zaxis=dict(range=[0, 106], autorange=False),
            yaxis=dict(range=[0, 106], autorange=False),
            xaxis=dict(range=[0, 106], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        sliders=sliders,)
    figa['layout']['sliders'][0]['activebgcolor'] = "darkgoldenrod"
    figa['layout']['sliders'][0]['bgcolor'] = "blue"
    figa['layout']['sliders'][0]["minorticklen"] = 0
    return figa

subject_name =glob.glob(r"C:\Users\PythonEntw\PycharmProjects\Brain_Vis\WMModel\testimages\*.nii*")
import os
subject_name=os.path.split(subject_name[0])[1]

bin_sagital=np.copy(x_mask_sagital)
bin_coronal=np.copy(x_mask_coronal)
bin_axial=np.copy(x_mask_axial)
bin_sagital[bin_sagital>0]=1
bin_coronal[bin_coronal>0]=1
bin_axial[bin_axial>0]=1

mask_1, a, dataframe_coronal_, pos_rel_1, vis_map_1 = get_scores(p, x_coronal, bin_coronal, "coronal " + subject_name, rt=5)
mask_2, a2, dataframe_sagital_, pos_rel_2, vis_map_2=get_scores(p_sagital, x_sagital, bin_sagital, "sagital " + subject_name, rt=5)
mask_3, a3, dataframe_axial_, pos_rel_3, vis_map_3=get_scores(p_axial, x_axial, bin_axial, "axial " + subject_name, rt=5)

x_coronal_test = np.reshape(x_coronal,(len(x_coronal), x_coronal[0].shape[0], x_coronal[0].shape[1]))
x_sagital_test = np.reshape(x_sagital,(len(x_sagital), x_sagital[0].shape[0], x_sagital[0].shape[1]))
x_axial_test = np.reshape(x_axial,(len(x_axial), x_axial[0].shape[0], x_axial[0].shape[1]))

controls = dbc.Card([
html.Div([
        html.Label('Upload'),
        dcc.Upload(html.Button('Upload File')),
        ]),
        html.Div(
            [
                dbc.Label("choose mask for overlay"),
                dcc.Dropdown(id='dropdown',multi=False, options=[
                    {'label': x, 'value': x} for x in T]
                             ,
                             value=None),
            ]
        ),html.Div(
            [   dbc.Label("choose threshold"),
                dcc.Slider(0, 10, 1, value=3,id="slider_threshold", marks=None,
                           tooltip={"placement": "bottom", "always_visible": True}),
            ]
        ),
            html.Div(
            [
                dbc.Label("choose mask to get scores"),
                dcc.Dropdown(id='dropdown_dataframe',multi=False, options=[
                    {'label': x, 'value': x} for x in T]
                             ,
                             value=None),
            ]
        ),html.Div(
            [
                dbc.Label("choose visualization "),
                dcc.Dropdown(id='dropdown_vis',multi=False, options=[
                    {'label': x, 'value': x} for x in T_vis]
                             ,
                             value=None),
            ]
        ),
        html.Div(
            [
                dbc.Label("Select region to draw"),
                dcc.Dropdown(id='dropdown_region',multi=False, options=[
                    {'label': x, 'value': x} for x in T]
                             ,
                             value=None),
                dcc.RadioItems(
                ['slicing on', 'slicing off'],
                'slicing on',
                id='radio',
                inline=True
            )
            ]
        ),
    ],
    body=True,
)
row_content_1=[dbc.Col(dcc.Graph(id="graph-court"),width={"size": 3, "offset": 2}),
                dbc.Col(dcc.Graph(id="graph_sagital"), width=3),
                dbc.Col(dcc.Graph(id="graph_axial"),width=3)]
row_content_2=[dbc.Col(controls, width=2),
               dbc.Col(dcc.Graph(id="graph-2d_court"),width=3),
               dbc.Col(dcc.Graph(id="graph-2d_sagital"),width=3),
                dbc.Col(dcc.Graph(id="graph-2d_axial"),width=3)]
app.layout = dbc.Container(
    [
        html.H1("Alzheimer Web Application"),
        html.Hr(),
        dbc.Row(
            row_content_2,
            align="start",
        #className="g-0",
        ),
        dbc.Row(row_content_1,align="center"  #className="g-0",
        ),
        dbc.Row([
            dbc.Col(dcc.Graph(id="dataframe"),width={"size": 5,"offset": 2}),
            dbc.Col(dcc.Graph(id="mean_relevance"),width={"size":5}),

        ],align="start"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="Brain_3d_coronal"),width={"size": 4}),
            dbc.Col(dcc.Graph(id="show_mask_3d"),width={"size": 4}),#justify="center",#,width=6
            dbc.Col(dcc.Graph(id="Brain_3d_axial"),width={"size": 4}),
        ]),
        dbc.Row([

        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="mixed_vis"),width={"size": 4}),
            dbc.Col(dcc.Graph(id="mixed_vis_sagital"),width={"size": 4 }),
            dbc.Col(dcc.Graph(id="mixed_vis_axial"),width={"size": 4 }),
            #dbc.Col(dcc.Graph(id="show_mask_3d"),width=3),
        ]),
    ],
    fluid=True,)

@app.callback(
[Output('graph-2d_court', 'figure'),Output("graph-2d_sagital","figure"),Output("graph-2d_axial","figure"),
 Output('graph-court', 'figure'),Output("graph_sagital","figure"),Output("graph_axial","figure")],
              [Input('dropdown', 'value'),Input("slider_threshold","value")])
def update_mask(selected_value,t):
    if selected_value is None and t is None:
        return two_d_vis(vis_map_1,x_coronal_test,title="coronal %s"%(print_prediction(p,x_coronal))),two_d_vis(vis_map_2,x_sagital_test,title="sagital %s"%(print_prediction(p_sagital,x_sagital))),two_d_vis(vis_map_3,x_axial_test,title="axial %s"%(print_prediction(p_axial,x_axial))),three_d_vis(vis_map_1, x_coronal_test),three_d_vis(vis_map_2,x_sagital_test),three_d_vis(vis_map_3,x_axial_test)
    elif selected_value is not None and t is None:
        k=T.index((selected_value))
        print("get the element from dropdown",k)
        return two_d_vis(vis_map_1,x_coronal_test,k=k,title="coronal %s"%(print_prediction(p,x_coronal))),two_d_vis(vis_map_2,x_sagital_test,k=k,title="sagital %s"%(print_prediction(p_sagital,x_sagital))),two_d_vis(vis_map_3,x_axial_test,k=k,title="axial %s"%(print_prediction(p_axial,x_axial))),three_d_vis(vis_map_1, x_coronal_test,k=k),three_d_vis(vis_map_2,x_sagital_test,k=k),three_d_vis(vis_map_3,x_axial_test,k=k)
    elif selected_value is None and t is not None:
        return two_d_vis(vis_map_1,x_coronal_test,t=t,title="coronal %s"%(print_prediction(p,x_coronal))),two_d_vis(vis_map_2,x_sagital_test,t=t,title="sagital %s"%(print_prediction(p_sagital,x_sagital))),two_d_vis(vis_map_3,x_axial_test,t=t,title="axial %s"%(print_prediction(p_axial,x_axial))),three_d_vis(vis_map_1, x_coronal_test,t=t),three_d_vis(vis_map_2,x_sagital_test,t=t),three_d_vis(vis_map_3,x_axial_test,t=t)
    else:
        k = T.index((selected_value))
        return two_d_vis(vis_map_1,x_coronal_test,t=t,k=k,title="coronal %s"%(print_prediction(p,x_coronal))),two_d_vis(vis_map_2,x_sagital_test,t=t,k=k,title="sagital %s"%(print_prediction(p_sagital,x_sagital))),two_d_vis(vis_map_3,x_axial_test,t=t,k=k,title="axial %s"%(print_prediction(p_axial,x_axial))),three_d_vis(vis_map_1, x_coronal_test,t=t,k=k),three_d_vis(vis_map_2,x_sagital_test,t=t,k=k),three_d_vis(vis_map_3,x_axial_test,t=t,k=k)

@app.callback(
[Output('dataframe', 'figure'),Output("mean_relevance","figure")],
[Input('dropdown_dataframe', 'value'),Input("slider_threshold","value")])
def update_dropdown_dataframe(selected,scale):
    if selected is None :
        return plot_dataframe_mask(vis_map_1,vis_map_2,vis_map_3,t=8,scale=scale), get_mean_sum_rel(pos_rel_1, pos_rel_2, pos_rel_3, text=subject_name)
    else:
        k = T.index(selected)
        return plot_dataframe_mask(vis_map_1,vis_map_2,vis_map_3,t=int(k),scale=scale), get_mean_sum_rel(pos_rel_1, pos_rel_2, pos_rel_3, text=subject_name)



@app.callback(
[Output("Brain_3d_coronal","figure"),Output("show_mask_3d","figure"),Output("Brain_3d_axial","figure")],
[Input('dropdown_dataframe', 'value'),Input("dropdown_vis","value"),Input("slider_threshold","value")])

def update_brain_3d(selected,vis,scale):
    if selected is None and (vis is None or T_vis.index(vis)==0):
        return three_d_vis_mask(vis_map_1, d_Test[0].transpose(1,0,2), k=8, scale=scale,text="Coronal") ,three_d_vis_mask(vis_map_2, d_Test[0], k=8, scale=scale,text="Sagittal"),three_d_vis_mask(vis_map_3, d_Test[0].transpose(2,0,1), k=8, scale=scale,text="Axial")
    elif (vis is None or T_vis.index(vis)==0) and selected is not None:
        k = T.index(selected)
        return three_d_vis_mask(vis_map_1, data_overlay[k].transpose(1,0,2), k=k, scale=scale,text="Coronal"),three_d_vis_mask(vis_map_2, data_overlay[k], k=k, scale=scale,text="Sagittal"),three_d_vis_mask(vis_map_3, data_overlay[k].transpose(2,0,1), k=k, scale=scale,text="Axial")
    elif selected is None and (T_vis.index(vis) ==1):
        return saliency_slicing(vis_map_1, d_Test[0].transpose(1,0,2), k=8, scale=scale,text="Coronal") ,saliency_slicing(vis_map_2, d_Test[0], k=8, scale=scale,text="Sagittal"),saliency_slicing(vis_map_3, d_Test[0].transpose(2,0,1), k=8, scale=scale,text="Axial")
    elif selected is not None and (T_vis.index(vis)==1):
        k=T.index(selected)
        return saliency_slicing(vis_map_1, data_overlay[k].transpose(1,0,2), k=k, scale=scale,text="Coronal") ,saliency_slicing(vis_map_2, data_overlay[k], k=k, scale=scale,text="Sagittal"),saliency_slicing(vis_map_3, data_overlay[k].transpose(2,0,1), k=k, scale=scale,text="Axial")
    elif selected is None and (T_vis.index(vis)==2):
        return saliency_scatter(vis_map_1, d_Test[0].transpose(1,0,2), k=8, scale=scale,text="Coronal") ,saliency_scatter(vis_map_2, d_Test[0], k=8, scale=scale,text="Sagittal"),saliency_scatter(vis_map_3, d_Test[0].transpose(2,0,1), k=8, scale=scale,text="Axial")
    elif selected is not None and (T_vis.index(vis)==2):
        k = T.index(selected)
        return saliency_scatter(vis_map_1, data_overlay[k].transpose(1,0,2), k=k, scale=scale,text="Coronal"),saliency_scatter(vis_map_2, data_overlay[k], k=k, scale=scale,text="Sagittal"),saliency_scatter(vis_map_3, data_overlay[k].transpose(2,0,1), k=k, scale=scale,text="Axial")


@app.callback(
[Output('mixed_vis', 'figure'),Output("mixed_vis_sagital","figure"),Output("mixed_vis_axial","figure")],
[Input('dropdown_region', 'value'),Input("slider_threshold","value"),Input("radio","value")])

def update_dropdown_region(selected,scale,radio="slicing on"):
    if selected is None:
        return mixed_vis(vis_map_1,d_Test[0].transpose(1,0,2),d_Test[0].transpose(1,0,2),scale=scale,radio="slicing off"),mixed_vis(vis_map_2,d_Test[0],d_Test[0],scale=scale,radio="slicing off"),mixed_vis(vis_map_3,d_Test[0].transpose(2,0,1),d_Test[0].transpose(2,0,1),scale=scale,radio="slicing off")
    else:
        k = T.index(selected)
        return mixed_vis(vis_map_1,data_overlay[k].transpose(1,0,2),d_Test[0].transpose(1,0,2),k=k,scale=scale,radio="slicing off"),mixed_vis(vis_map_2,data_overlay[k],d_Test[0],k=k,scale=scale,radio="slicing off"),mixed_vis(vis_map_3,data_overlay[k].transpose(2,0,1),d_Test[0].transpose(2,0,1),k=k,scale=scale,radio="slicing off")


if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)