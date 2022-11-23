from operator import not_
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def get_velocities(vals):
    '''
    Given a set of 3d points, calculates instanteous velocity at each point.
        vals  : input in the form [x, y, z, L] (L is label, and is not used)
        
        times : the time stamps for each instantaneous velocity
        velocities : velocity values in mph
        vel_x : velocity values in x direction in 3D units per second
        vel_y : velocity values in y direction in 3D units per second
        vel_z : velocity values in z direction in 3D units per second
        indices : indices in the vals array for which instantaneous velocites were found
    '''
    def dist_2_mph(distance, delta_time):
        '''
        Takes in distance between two points and the time difference. 
        Returns the velocity in miles per hour.
        '''
        return distance * 70.0 * 3600 / (100.0 * delta_time * 63360)

    times = []
    velocities = []
    vel_x = []
    vel_y = []
    vel_z = []
    indices = []

    x1 = 0  # Starting point.
    x2 = 1  # Second point.
    fps = 30.0  # Frame rate.
    while True:
        dist = np.linalg.norm(vals[x1,:3]-vals[x2,:3])  # Distance between two points. 
        time = (x2-x1)/fps                              # Time change between frames.
        times += [x2/fps]                               # Store time change. 
        velocities += [dist_2_mph(dist, time)]          # Store velocity.
        vel_x += [(vals[x2,0]-vals[x1,0])/time]         # Store velocity in x direction. 
        vel_y += [(vals[x2,1]-vals[x1,1])/time]         # Store velocity in y direction. 
        vel_z += [(vals[x2,2]-vals[x1,2])/time]         # Store velocity in z direction. 
        indices += [x2]                                 # Store indices.
        if x2 == len(vals[:,0])-1:                      # If is at the end, break. 
            break
        x1 += 1                                         # Increment index. 
        while vals[x1][0] is None:                      # If there is None data, skip.
            x1 += 1
        x2 += 1                                         # Increment second index. 
        while vals[x2][0] is None:                      # If there is None data, skip. 
            x2 += 1                                 
        
    return times, velocities, vel_x, vel_y, vel_z, indices

def analysis(points_3D):
    '''
    points_3D : input array should be N by 4, where the 4th column is a label for coloring.
    '''

    points_3D_copy = points_3D.copy()   # Make copy of 3D points for velocity calculations.

    # Remove all columns with None for plotting purposes.
    not_None = []
    for i in range(len(points_3D)):
        if points_3D[i][0] is not None:
            not_None.append(i)
    not_None = np.array(not_None)
    points_3D = points_3D[not_None,:]

    # Turn on interactive plots. 
    plt.ion()

    # Pole and Backboard
    post = np.array([
        [0, 0, 0],
        [0, 25-25*np.cos(0*np.pi/180), 163.5-25+25*np.sin(0*np.pi/180)],
        [0, 25-25*np.cos(15*np.pi/180), 163.5-25+25*np.sin(15*np.pi/180)],
        [0, 25-25*np.cos(30*np.pi/180), 163.5-25+25*np.sin(30*np.pi/180)],
        [0, 25-25*np.cos(45*np.pi/180), 163.5-25+25*np.sin(45*np.pi/180)],
        [0, 25-25*np.cos(60*np.pi/180), 163.5-25+25*np.sin(60*np.pi/180)],
        [0, 25-25*np.cos(75*np.pi/180), 163.5-25+25*np.sin(75*np.pi/180)],
        [0, 25-25*np.cos(90*np.pi/180), 163.5-25+25*np.sin(90*np.pi/180)],
        [0, 63, 163.5],
        [0., 68., 160.],
        [-15., 68., 160.],
        [-40., 68., 167.],
        [-40., 68., 177.],
        [-30., 68., 194.],
        [-10., 68., 205.],
        [10., 68., 205.],
        [30., 68., 194.],
        [40., 68., 177.],
        [40., 68., 167.],
        [15., 68., 160.],
        [0., 68., 160.],
        [0., 75., 171.]
    ])

    # Rim
    rim = []
    for i in range(0, 375, 15):
        rim.append([0. - 12.8571*np.sin(i*np.pi/180), 87.8571 - 12.8571*np.cos(i*np.pi/180), 171])
    rim = np.array(rim)

    # Lines
    baseline = np.array([
        [-300, 0, 0],
        [300, 0, 0],
    ])
    box = np.array([
        [0, 0, 0],
        [100, 0, 0],
        [100, 322, 0],
        [-100, 322, 0],
        [-100, 0, 0],
        [0, 0, 0]
    ])
    arc = np.array([
        [0, 322, 0],
        [100, 322, 0],
        [98.48, 322+17.36, 0],
        [93.96, 322+34.2, 0],
        [86.6, 322+50, 0],
        [76.6, 322+64.27, 0],
        [64.27, 322+76.6, 0],
        [50, 322+86.60, 0],
        [34.2, 322+93.96, 0],
        [17.36, 322+98.48, 0],
        [0, 322+100, 0],
        [-17.36, 322+98.48, 0],
        [-34.2, 322+93.96, 0],
        [-50, 322+86.60, 0],
        [-64.27, 322+76.6, 0],
        [-76.6, 322+64.27, 0],
        [-86.6, 322+50, 0],
        [-93.96, 322+34.2, 0],
        [-98.48, 322+17.36, 0],
        [-100, 322+0, 0],
        [0, 322+0, 0],
    ])

    # Velocity Calculations
    times, velocities, vel_x, vel_y, vel_z, indices = get_velocities(points_3D_copy)

    start = 11  # Ball center to use for our initial 3D position and initial velocities of trajectory.
    g = - 9.80678*39.3701*100.0/70.0    # Convert gravitational constant to our 3D plane units / second. 
    x0, y0, z0 = points_3D[start,0],points_3D[start,1],points_3D[start,2]   # Initial 3D position.
    vx, vy, vz = vel_x[start],vel_y[start],vel_z[start]     # Initial 3D velocity. 

    fps = 30.0
    t = np.linspace(0,1.25*(len(np.where(points_3D_copy[:,3]==3)[0])-start)/fps, 100)   # Range of velocity prediction line.
    X_3D = x0 + vx * t  # X values physics equation. 
    Y_3D = y0 + vy * t  # Y values physics equation. 
    Z_3D = z0 + (vz * t) + ((g * t * t) / 2.0)  # Z values physics equation.

    t = np.linspace(0,(len(points_3D_copy)-start)/fps, len(points_3D_copy)-start+1) # Replicate frame intervals after start to compare/plot velocities.
    pred_vals_x = x0 + vx * t   # X values physics equation. 
    pred_vals_y = y0 + vy * t   # Y values physics equation. 
    pred_vals_z = z0 + (vz * t) + ((g * t * t) / 2.0)   # Z values physics equation.

    # Setup 3D plot. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axes.set_xlim3d(left=-225, right=225) 
    ax.axes.set_ylim3d(bottom=-50, top=400) 
    ax.axes.set_zlim3d(bottom=0, top=450) 
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')

    # Plot 3D centers.
    ind_3d = np.where(points_3D[:,3]==3)
    ax.scatter(points_3D[ind_3d,0],points_3D[ind_3d,1],points_3D[ind_3d,2], c='y', edgecolors=None,label = 'ball center detected by 3 cameras'); 
    ind_3d = np.where(points_3D[:,3]==2)
    ax.scatter(points_3D[ind_3d,0],points_3D[ind_3d,1],points_3D[ind_3d,2], c='purple', edgecolors=None,label = 'ball center detected by 2 cameras'); 


    # Plot estimated Physics trajectory. 
    ax.plot(X_3D,Y_3D,Z_3D, c='r',label = 'projectile motion trajectory estimation'); 

    # Environment
    ax.plot3D(box[:,0],box[:,1],box[:,2], c='k');
    ax.plot3D(arc[:,0],arc[:,1],arc[:,2], c='k');
    ax.plot3D(baseline[:,0],baseline[:,1],baseline[:,2], c='k');
    ax.plot3D(post[:,0],post[:,1],post[:,2], c='k');
    ax.plot3D(rim[:,0],rim[:,1],rim[:,2], c='r');
    ax.legend()

    # Used for Plotting Hashes
    def polygon_plot(x, y, z):
        vertices = [list(zip(x,y,z))]
        poly = Poly3DCollection(vertices, color='k')
        ax.add_collection3d(poly)

    x = [100, 100, 115, 115]
    y = [120, 137, 137, 120]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)
    x = [-100, -100, -115, -115]
    y = [120, 137, 137, 120]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)

    x = [100, 100, 110, 110]
    y = [188.5, 192, 192, 188.5]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)
    x = [-100, -100, -110, -110]
    y = [188.5, 192, 192, 188.5]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)

    x = [100, 100, 110, 110]
    y = [240.7, 244.2, 244.2, 240.7]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)
    x = [-100, -100, -110, -110]
    y = [240.7, 244.2, 244.2, 240.7]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)

    x = [100, 100, 110, 110]
    y = [291.4, 294.9, 294.9, 291.4]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)
    x = [-100, -100, -110, -110]
    y = [291.4, 294.9, 294.9, 291.4]
    z = [0, 0, 0, 0]
    polygon_plot(x, y, z)

    plt.show()  # Shows interactive 3D plot. 

    # Gets velocites from predicted points from Physics trajectory. 
    times_pred, velocities_pred, vel_x_pred, vel_y_pred, vel_z_pred, indices_pred = get_velocities(np.array([pred_vals_x, pred_vals_y, pred_vals_z]).T)

    # Turn lists to arrays for indexing. 
    times = np.array(times)
    times_pred_adj = np.array(times_pred) + (start/fps)
    velocities = np.array(velocities)
    vel_x = np.array(vel_x)
    vel_y = np.array(vel_y)
    vel_z = np.array(vel_z)
    vel_x_pred = np.array(vel_x_pred)
    vel_y_pred = np.array(vel_y_pred)
    vel_z_pred = np.array(vel_z_pred)

    # Convert directional velocites to miles per hour. 
    def unitsperseconds_to_mph(distances):
        return distances * 70.0 * 3600 / (100.0 * 63360)
    vel_x = unitsperseconds_to_mph(vel_x)
    vel_y = unitsperseconds_to_mph(vel_y)
    vel_z = unitsperseconds_to_mph(vel_z)
    vel_x_pred = unitsperseconds_to_mph(vel_x_pred)
    vel_y_pred = unitsperseconds_to_mph(vel_y_pred)
    vel_z_pred = unitsperseconds_to_mph(vel_z_pred)

    # Plot total velocity with inaccurate points (<3 cameras captured center).
    fig=plt.figure()
    index = np.where(points_3D_copy[indices][:,3]==3)
    plt.scatter(times[index], velocities[index], c='y',label = 'ball center detected by 3 cameras')
    index = np.where(points_3D_copy[indices][:,3]==2)
    plt.scatter(times[index], velocities[index], c='purple',label = 'ball center detected by 2 cameras')
    plt.plot(times_pred_adj, velocities_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("Velocity vs. Time");
    plt.show()

    # Plot x velocity with inaccurate points (<3 cameras captured center).
    fig=plt.figure()
    index = np.where(points_3D_copy[indices][:,3]==3)
    plt.scatter(times[index], vel_x[index], c='y',label = 'ball center detected by 3 cameras')
    index = np.where(points_3D_copy[indices][:,3]==2)
    plt.scatter(times[index], vel_x[index], c='purple',label = 'ball center detected by 2 cameras')
    plt.plot(times_pred_adj, vel_x_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("X Velocity vs. Time");
    plt.show()

    # Plot y velocity with inaccurate points (<3 cameras captured center).
    fig=plt.figure()
    index = np.where(points_3D_copy[indices][:,3]==3)
    plt.scatter(times[index], vel_y[index], c='y',label = 'ball center detected by 3 cameras')
    index = np.where(points_3D_copy[indices][:,3]==2)
    plt.scatter(times[index], vel_y[index], c='purple',label = 'ball center detected by 2 cameras')
    plt.plot(times_pred_adj, vel_y_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("Y Velocity vs. Time");
    plt.show()

    # Plot z velocity with inaccurate points (<3 cameras captured center).
    fig=plt.figure()
    index = np.where(points_3D_copy[indices][:,3]==3)
    plt.scatter(times[index], vel_z[index], c='y',label = 'ball center detected by 3 cameras')
    index = np.where(points_3D_copy[indices][:,3]==2)
    plt.scatter(times[index], vel_z[index], c='purple',label = 'ball center detected by 2 cameras')
    plt.plot(times_pred_adj, vel_z_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("Z Velocity vs. Time");
    plt.show()

    # Plot total velocity without inaccurate points.
    fig=plt.figure()
    all_detect = np.where(points_3D_copy[:,3]==3)
    all_detect = np.intersect1d(all_detect, indices)
    ind = []
    for i in all_detect:
        ind.append(np.where(indices == i)[0][0])
    plt.scatter(times[ind], velocities[ind], c='y',label = 'ball center detected by 3 cameras')
    plt.plot(times_pred_adj, velocities_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("Velocity vs. Time");
    plt.show()

    # Plot x velocity without inaccurate points.
    fig=plt.figure()
    plt.scatter(times[ind], vel_x[ind], c='y',label = 'ball center detected by 3 cameras')
    plt.plot(times_pred_adj, vel_x_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("X Velocity vs. Time");
    plt.show()

    # Plot y velocity without inaccurate points.
    fig=plt.figure()
    plt.scatter(times[ind], vel_y[ind], c='y',label = 'ball center detected by 3 cameras')
    plt.plot(times_pred_adj, vel_y_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("Y Velocity vs. Time");
    plt.show()

    # Plot z velocity without inaccurate points.
    fig=plt.figure()
    plt.scatter(times[ind], vel_z[ind], c='y',label = 'ball center detected by 3 cameras')
    plt.plot(times_pred_adj, vel_z_pred, c='r')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mph)")
    plt.title("Z Velocity vs. Time");
    plt.show()

    # Plot ball height with inaccurate points (<3 cameras captured center).
    fig=plt.figure()
    x_time = np.linspace(0,len(points_3D_copy)/fps, len(points_3D_copy)+1)
    index_h = np.where(points_3D_copy[:,3]==3)
    plt.scatter(x_time[index_h], points_3D_copy[index_h,2]*70./1200., c='y',label = 'ball center detected by 3 cameras')
    index_h = np.where(points_3D_copy[:,3]==2)
    plt.scatter(x_time[index_h], points_3D_copy[index_h,2]*70./1200., c='purple',label = 'ball center detected by 2 cameras')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Height (ft)")
    plt.title("Height vs. Time");
    plt.show()