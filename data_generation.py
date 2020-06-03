import time, os, copy, sys, json, argparse, warnings, cv2
import torch
from torch import nn
from laspy.file import File
from laspy.header import Header
from numba import jit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask import delayed, compute
from scipy import ndimage, stats, interpolate
from scipy.signal import medfilt
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny, peak_local_max
from skimage.morphology import dilation, remove_small_holes, remove_small_objects, convex_hull_object, disk, square, watershed
from skimage.draw import circle_perimeter, circle
from skimage.filters import roberts, sobel, scharr, prewitt, threshold_otsu, rank
from skimage.color import gray2rgb
from sklearn.cluster import DBSCAN
from skimage.measure import CircleModel, ransac
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import medfilt, find_peaks
from scipy.stats import weibull_min, kurtosis, skew
import statsmodels.distributions

warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_folder(out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

def files_list(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.laz')]

@jit(nopython=True)
def fill_matrix(data, array): 
	for i in range(data.shape[0]):
		array[data[i][0]][data[i][1]][data[i][2]] = 1

	return array

def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)

def distance(lat, lon, buffer_size, x, y, z):
    
    dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))

    index = np.where(dist < buffer_size)
    
    return x[index], y[index], z[index]

def mask_generator(min_r, max_r):
    masks = []

    for r in range(min_r, max_r + 1):

        image = np.zeros((max_r + 1, max_r + 1))
        circy, circx = circle(int(0.5*max_r), int(0.5*max_r), r, shape=image.shape)
        image[circy, circx] = -1/r
        circy, circx = circle_perimeter(int(0.5*max_r), int(0.5*max_r), r - 1, shape=image.shape)
        circy, circx = circle_perimeter(int(0.5*max_r), int(0.5*max_r), r, shape=image.shape)
        image[circy, circx] = 1
        masks.append(image)

    return np.stack(masks, axis=0)

def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)

def kantor_encoder(y, x):
    return 0.5*(y+x)*(y+x+1)+y

def buffer(x, y, z, min_dist):
    index = np.where(z > 0)
    x, y, z = x[index], y[index], z[index]
    x_c, y_c = np.mean(x), np.mean(y)
    dist = np.sqrt(np.power(x - x_c,2) + np.power(y - y_c,2))
    index = np.where(dist < min_dist)
    
    return x[index], y[index], z[index]

def scoring(array, masks, min_r, max_r, min_h, max_h):

    array_tensor = torch.from_numpy(np.expand_dims(array[min_h:max_h,:,:], axis=0)).double()
    conv_layer = nn.Conv2d(array_tensor.shape[1], masks.shape[0], max_r + 1, padding=1)
    masks_batch = np.repeat(masks[:,np.newaxis,:,:], array_tensor.shape[1], axis=1)
    conv_layer.weight.data = torch.from_numpy(masks_batch).double()
    conv_layer.bias.data = torch.from_numpy(np.zeros(masks.shape[0])).double()
    score = conv_layer(array_tensor)
    score = score.detach().numpy()

    return np.squeeze(np.max(score, axis=1).astype(int), axis=0)

def trees_points(X, Y, Z, ID):

    indices = np.argsort(ID)
    X, Y, Z = X[indices], Y[indices], Z[indices]
    unique, counts = np.unique(ID, return_counts=True)
    
    c_gr = counts[0] 
    X, Y, Z = X[c_gr:], Y[c_gr:], Z[c_gr:]

    indices = []

    for u_id in range(1, len(unique)):

        c_id = counts[u_id]
        Xt, Yt, Zt = X[:c_id], Y[:c_id] , Z[:c_id]
        X, Y, Z = X[c_id:], Y[c_id:], Z[c_id:]
        indices.append((Xt, Yt, Zt, np.max(Zt)))

    return indices

def tree_id(labels, X, Y):

    n_rows, n_columns = labels.shape
    y_i, x_i = np.repeat(np.arange(n_rows), n_columns), np.tile(np.arange(n_columns), n_rows)
    key_kantor, values = kantor_encoder(y_i, x_i).astype(int), np.ravel(labels).astype(int)
    dictionary = dict(zip(key_kantor.tolist(), values.tolist()))
    data_kantor = kantor_encoder(Y, X).astype(int)
    i = vec_translate(data_kantor, dictionary)

    return i

def dap_tree(x, y, z, h_max, z_min=1.25, z_max = 1.35):

    scale = 0.01
    min_x, max_y = np.min(x), np.max(y)
    indice = np.where((z<z_max)&(z>=z_min))
    x, y = (1/scale)*(x[indice]-min_x), (1/scale)*(max_y-y[indice])
    
    if len(x) > 15:

        xt, yt = x, y    
        clustering = DBSCAN(eps=3, min_samples=25).fit(np.stack((y, x), axis=1))
        labels = clustering.labels_
    
        if len(np.bincount(labels[labels >=0]))>0:
            index_cluster = np.where(labels==np.argmax(np.bincount(labels[labels >=0])))
            x, y = x[index_cluster], y[index_cluster]
    
        ransac_model, inliers = ransac(np.stack((y, x), axis=1), CircleModel, 15, 3, max_trials=50)
        x, y = x[inliers], y[inliers]
        lat, lon, radii = ransac_model.params

        dist = np.sqrt(np.power(xt-lon, 2)+np.power(yt-lat,2))
        indice = np.where(dist < radii+1)

        x, y = xt[indice], yt[indice]
        model = CircleModel()
        model.estimate(np.stack((y, x), axis=1)) 
        lat, lon, radii = model.params

        n_i = points_inside_radii(x, y, lat, lon, radii)
        n_t = len(x)
            
        if ((radii > 4) & (radii <50) &(n_t>18) & ((n_i/n_t)>0.7)):
            return (x, y, lat, lon, radii, max_y, min_x, h_max)            

    return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

def std_radii(x, y, z):
    
    h_min, h_max = 1.5, 9    
    height = np.arange(h_min,h_max, 0.1)
    std = []
    
    for h in range(0,height.shape[0]):
        
        h_inf, h_sup = height[h]-0.05, height[h]+0.05
        indices = np.intersect1d(np.where(z>h_inf)[0], np.where(z<h_sup)[0])        
        Xl, Yl = x[indices], y[indices]
        
        if len(Xl)>10:
            radiis = np.sqrt(np.power(Xl-np.mean(Xl),2)+np.power(Yl-np.mean(Yl),2))
            max_radii = np.percentile(radiis, 90)
            radiis = radiis[radiis < max_radii]
            std.append(np.std(radiis))
        else:
            std.append(0)
    
    return np.array(std)

def prun_std(r):

    h_min, h_max = 1.5, 9
    height = np.arange(h_min,h_max, 0.1)

    cum_inv = -np.cumsum(r/sum(r))/np.arange(1,len(r)+1)
    mult = np.linspace(1.0, 1.25, num=len(r))
    cum_inv = (cum_inv - np.min(cum_inv))/(np.max(cum_inv)-np.min(cum_inv))
    cum_inv = cum_inv * mult
    peaks, _ = find_peaks(cum_inv, height=0.85*np.max(cum_inv))
    
    try:
        peak = peaks[-1]
        
    except:
        peak = 0
    
    return np.around(height[peak],1)

def prun_den(z):

    z = z[(z > 1.6) & (z < 9)]

    if len(z)>5:
        shape, loc, scale = weibull_min.fit(z, floc=0)
        ecdf = statsmodels.distributions.ECDF(z)
        x = np.linspace(z.min(), z.max(), 100)
        diff = weibull_min(shape, loc, scale).cdf(x) - ecdf(x)

        indices = np.where(x > 2)
        x = x[indices]
        diff = diff[indices]
        level = np.around(x[np.argmax(diff)],1)
    else:
        level = 0.0 
    
    return level


def prun_height(level_std, level_den, height_std):

    std_threshold = 0.015
    delta_inf, delta_sup = 2.5, 0.5    
    h_min, h_max = 1.5, 9
    height = np.arange(h_min,h_max, 0.1) 
    
    if(level_std == 1.5):
        level_std = level_den

    if(abs(level_den-level_std)>0.9):
        level_den = level_std
    
    index_inf = int((level_den - delta_inf - h_min)/0.1)
    
    if index_inf < 0:
        index_inf = 0
        
    index_sup = int((level_den - delta_sup - h_min)/0.1)

    lft_side = height_std[index_inf:index_sup]           

    if((level_den < 2.8) or (np.std(lft_side) > std_threshold)):
        level_den = 0

    return level_den    

def prunning(x, y, z):
    
    height_std = std_radii(x, y, z)
    level_den = prun_den(z)
    level_std = prun_std(height_std)

    return prun_height(level_std, level_den, height_std)


def save_dap(x, y, lat, lon, radii, file_name):

    min_lon, max_lon = int(lon) - 2*int(radii), int(lon) + 2*int(radii)
    min_lat, max_lat = int(lat) - 2*int(radii), int(lat) + 2*int(radii)
   
    circle = plt.Circle((lon, lat), radii, color='blue', fill=False)
    plt.scatter(x, y, s=4, c='red')
    plt.scatter(lon, lat, s=5, c='blue')
    plt.gcf().gca().add_artist(circle)
    plt.axis([min_lon, max_lon, min_lat, max_lat])
    plt.savefig(file_name)
    plt.clf()

def write_df(file_name, lats, lons, h_prun):

    df = pd.DataFrame()
    id_tree = len(lats)
    prun = (h_prun > 0).astype(int)

    df['LAT'] = lats.tolist()
    df['LON'] = lons.tolist()
    df['H_POD'] = h_prun.tolist()
    df['POD'] = prun.tolist()
    
    return id_tree, df

def points_inside_radii(x, y, lat, lon, radii):
    dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))
    return len(dist[(dist >= radii - 2)])

def check_path(path):
    if path[-1] != '/':
        path = path + '/'

    return path

def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)

def random_unique(A):
    x = np.random.rand(A.shape[1])
    y = A.dot(x)
    unique, index = np.unique(y, return_index=True)
    return A[index]

def save_file(file_name, header, X, Y, Z):
    with File(file_name, mode="w", header=header) as outfile:
        outfile.x, outfile.y, outfile.z = X, Y, Z

def tree_points_als(x, y, z, lat, lon, buffer_size):

    dist = np.sqrt(np.power(x-lon, 2)+np.power(y-lat,2))
    indices = np.where(dist < buffer_size)
    x, y, z = x[indices], y[indices], z[indices]

    return (x, y, z)

def clip_clouds(file, input_folder, df, output_folder):

    inFile = File(input_folder + file, mode = "r")
    
    print(input_folder + file)

    header = inFile.header
    X, Y, Z = inFile.x, inFile.y, inFile.z

    lats, lons = df['LAT'].values, df['LON'].values

    file_name = file[:-4]
    folder = file_name + '/'

    # Create a folder
    create_folder(output_folder + folder)

    indices = []
    
    for id_tree in range(0, len(lats)):
        i = delayed(tree_points_als)(X, Y, Z, lats[id_tree], lons[id_tree], 0.8)
        indices.append(i)

    indices = compute(*indices)
    
    tree_id = 0
    n_points = []
    index, files, n_points = [], [], []

    for indice in indices:

        np = len(indice[0])
        n_points.append(np)
        
        if(np > 100):
            save_file(output_folder + folder + str(tree_id) +'.laz', header, indice[0], indice[1], indice[2])
            index.append(tree_id)
            files.append(output_folder + folder + str(tree_id) +'.laz')
            tree_id = tree_id + 1
    
    df['N_POINTS'] = n_points
    df = df[df['N_POINTS']>100]
    
    df['INDEX'] = index
    df['FILE'] = files

    return df

def run(file, input_folder, output_folder):
    
    file_name = file[:-4]
    folder = file_name + '/'

    # Create a folder
    create_folder(output_folder + folder)

    # Scale parameters for voxelization (unit in meters) 
    scale = 0.02
    scale_z = 0.1
    buffer_size = 1
    min_dist = 25
    
    # Search boundaries parameters for radii and height (unit in meters)
    min_r, max_r = 0.06, 0.12
    zmin, zmax = 1.5, 9

    # Reparametrization of parameters
    min_r, max_r = int(min_r/scale), int(max_r/scale)
    z_min, z_max = int(zmin/scale_z), int(zmax/scale_z)

    # Load LiDar file and read X,Y,Z
    inFile = File(input_folder + file, mode = "r")

    header = inFile.header
    X, Y, Z = inFile.x, inFile.y, inFile.z

    # LiDAR normalization and buffer clipping
    # Filter LiDAR points greater than 9 m
    X, Y, Z = buffer(X, Y, Z, min_dist)

    # Calculate Min Max for every dimension
    max_x, min_x = np.max(X), np.min(X)
    max_y, min_y = np.max(Y), np.min(Y)
    max_z, min_z = np.max(Z), np.min(Z)

    # Transform coordinates into a grid array index
    Xp = np.array((1/scale)*(X - min_x), dtype=np.int)
    Yp = np.array((1/scale)*(max_y -Y), dtype=np.int)
    Zp = np.array((1/scale_z)*(Z - min_z), dtype=np.int)

    # Group all LiDAR points in same grid cell
    dz_values = np.stack((Zp, Yp, Xp), axis=1).astype(int)
    dz_values = random_unique(dz_values)

    max_values = np.max(dz_values, axis=0).astype(int)
    min_values = np.min(dz_values, axis=0).astype(int)

    size_x = int(max_values[2]-min_values[2]+1)
    size_y = int(max_values[1]-min_values[1]+1)
    size_z = int(max_values[0]-min_values[0]+1)

    # Create an empty Voxel Array and fill it with points
    array = np.full((size_z, size_y, size_x), 0)
    array = fill_matrix(dz_values, array)
    array = array[:z_max,:,:]

    # Calculate the average cleanest height level
    points_cum = np.sum(array, axis=(1,2))
    threshold_cum = np.percentile(points_cum, 10)
    levels = np.sort(np.where(points_cum < threshold_cum)[0])
    min_l, max_l = int(np.mean(levels)-10), int(np.mean(levels)+10)
    
    # Check if min_l is negative, force it to zero
    if min_l < 0:
    	min_l = 0
    	    
    # Generate masks and convolution process to score probability of a tree
    masks = mask_generator(min_r, max_r)

    image = np.array(255*(array[int(np.mean(levels)),:,:]/np.max(array[int(np.mean(levels)),:,:])),dtype=np.uint8)
    image = gray2rgb(image)
    
    score = scoring(array, masks, min_r, max_r, min_l, max_l) 
    score = np.pad(score, [(2, 2), (2, 2)], mode='constant')
    
    sobel_score = sobel(score)
    sobel_score = np.array(255*(sobel_score/np.max(sobel_score)), dtype=np.uint8)

    sobel_max = rank.maximum(sobel_score, disk(5))

    circles = cv2.HoughCircles(sobel_max, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=15, minRadius=6, maxRadius=20)
    circles = np.uint16(np.around(circles))

    threshold_score = np.percentile(sobel_max, 97)
    n_rows, n_columns = sobel_score.shape

    labels = np.zeros((n_rows, n_columns))

    id_t = 1

    rows, columns = np.array([]), np.array([]) 

    for i in circles[0,:]:
        circy, circx = circle(i[1], i[0], i[2], shape=image.shape)

        if np.max(sobel_max[circy, circx] >= threshold_score):
            circy, circx = circle_perimeter(i[1], i[0], i[2], shape=image.shape)
            image[circy, circx] = (255, 0, 0)
            circy, circx = circle(i[1], i[0], 1.3*i[2], shape=labels.shape)
            rows, columns =np.append(rows, i[1]), np.append(columns, i[0])
            labels[circy, circx] = id_t
            id_t = id_t + 1

    lats, lons = -scale*rows + max_y - 0.5*scale, scale*columns + min_x + 0.5*scale

    i = tree_id(labels, Xp, Yp)
    indices = trees_points(X, Y, Z, i)
        
    # Prunning level calculations

    prun = []

    for indice in indices:
        pr = delayed(prunning)(indice[0], indice[1], indice[2])
        prun.append(pr)

    prun = compute(*prun)
    prun = np.array(prun)

    n_trees, df_table = write_df(file_name, lats, lons, prun)

    # Save images 
    score = gray2rgb(sobel_score)
    mpimg.imsave(output_folder + folder + file_name + '_score.png', score)
    mpimg.imsave(output_folder + folder + file_name + '.png', image)

    return df_table


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Airborne LiDAR Sensor Forestry Processing Algorithm')
    parser.add_argument('tls_path', type=str, help="Input dir for TLS laz files")
    parser.add_argument('als_path', type=str, help="Input dir for ALS laz files")
    parser.add_argument('output_path', type=str, help="Output dir to save the results")
    args = parser.parse_args()

    # File paths
    tls_folder = args.tls_path
    als_folder = args.als_path
    output_folder = args.output_path
    
    als_folder, tls_folder, output_folder = check_path(als_folder), check_path(tls_folder), check_path(output_folder)
    
    create_folder(output_folder)
    
    files = [f for f in os.listdir(tls_folder) if f.endswith('.laz')]
    frames = []

    for file in files:
        try:
            start_time = time.time()
            df = run(file, tls_folder, output_folder)
            df = clip_clouds(file, als_folder, df, output_folder)
            df.to_csv(output_folder + file[:-4] + '.csv', sep=',',index=False)
            print(file, ' had ', str(len(df.index)), 'trees saved.' )
            print(file, "has been succesfully processed in --- %s seconds ---" % (time.time() - start_time) )
            frames.append(df)

        except:
            print(file, "was corrupted.")

    result = pd.concat(frames)    
    result.to_csv(output_folder + 'CONSOLIDADO.csv', sep=',', index=False)
