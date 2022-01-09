# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np, copy
from numba import jit
from scipy.spatial import ConvexHull
import scipy
import os

import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker(object):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self, bbox3D, info):
		"""
		Initialises a tracker using initial bounding box.
		"""
		# define constant velocity model
		self.kf = KalmanFilter(dim_x=10, dim_z=7)       
		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
		                      [0,1,0,0,0,0,0,0,1,0],
		                      [0,0,1,0,0,0,0,0,0,1],
		                      [0,0,0,1,0,0,0,0,0,0],  
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])

		# # with angular velocity
		# self.kf = KalmanFilter(dim_x=11, dim_z=7)       
		# self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
		#                       [0,1,0,0,0,0,0,0,1,0,0],
		#                       [0,0,1,0,0,0,0,0,0,1,0],
		#                       [0,0,0,1,0,0,0,0,0,0,1],  
		#                       [0,0,0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0,0,0],
		#                       [0,0,0,0,0,0,0,1,0,0,0],
		#                       [0,0,0,0,0,0,0,0,1,0,0],
		#                       [0,0,0,0,0,0,0,0,0,1,0],
		#                       [0,0,0,0,0,0,0,0,0,0,1]])     

		# self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
		#                       [0,1,0,0,0,0,0,0,0,0,0],
		#                       [0,0,1,0,0,0,0,0,0,0,0],
		#                       [0,0,0,1,0,0,0,0,0,0,0],
		#                       [0,0,0,0,1,0,0,0,0,0,0],
		#                       [0,0,0,0,0,1,0,0,0,0,0],
		#                       [0,0,0,0,0,0,1,0,0,0,0]])

		
		# self.kf.R[0:,0:] *= 10.   # measurement uncertainty
		self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
		self.kf.P *= 10.

		# self.kf.Q[-1,-1] *= 0.01    # process uncertainty
		self.kf.Q[7:, 7:] *= 0.01
		self.kf.x[:7] = bbox3D.reshape((7, 1))

		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.hits = 1           # number of total hits including the first detection
		self.hit_streak = 1     # number of continuing hit considering the first detection
		self.first_continuing_hit = 1
		self.still_first = True
		self.age = 0
		self.info = info        # other info associated

	def update(self, bbox3D, info): 
		""" 
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1          # number of continuing hit
		if self.still_first:
			self.first_continuing_hit += 1      # number of continuing hit in the fist time

		######################### orientation correction
		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		new_theta = bbox3D[3]
		if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
		if new_theta < -np.pi: new_theta += np.pi * 2
		bbox3D[3] = new_theta

		predicted_theta = self.kf.x[3]
		if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
			self.kf.x[3] += np.pi       
			if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
			if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
			if new_theta > 0: self.kf.x[3] += np.pi * 2
			else: self.kf.x[3] -= np.pi * 2

		#########################     # flip

		self.kf.update(bbox3D)

		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
		self.info = info

	def predict(self):       
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		self.kf.predict()      
		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		self.age += 1
		if (self.time_since_update > 0):
			self.hit_streak = 0
			self.still_first = False
		self.time_since_update += 1
		self.history.append(self.kf.x)
		return self.history[-1]

	def get_state(self):
		"""
		Returns the current bounding box estimate.
		"""
		return self.kf.x[:7].reshape((7, ))

@jit          
def poly_area(x,y):
	""" Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit         
def box3d_vol(corners):
	''' corners: (8,3) no assumption on axis direction '''
	a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
	b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
	c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
	return a*b*c

@jit          
def convex_hull_intersection(p1, p2):
	""" Compute area of two convex hull's intersection area.
		p1,p2 are a list of (x,y) tuples of hull vertices.
		return a list of (x,y) for the intersection and its volume
	"""
	inter_p = polygon_clip(p1,p2)
	if inter_p is not None:
		hull_inter = ConvexHull(inter_p)
		return inter_p, hull_inter.volume
	else:
		return None, 0.0  

def polygon_clip(subjectPolygon, clipPolygon):
	""" Clip a polygon with another polygon.
	Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

	Args:
		subjectPolygon: a list of (x,y) 2d points, any polygon.
		clipPolygon: a list of (x,y) 2d points, has to be *convex*
	Note:
		**points have to be counter-clockwise ordered**

	Return:
		a list of (x,y) vertex point for the intersection polygon.
	"""
	def inside(p):
		return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
 
	def computeIntersection():
		dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
		dp = [s[0] - e[0], s[1] - e[1]]
		n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
		n2 = s[0] * e[1] - s[1] * e[0] 
		n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
		return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]
 
	outputList = subjectPolygon
	cp1 = clipPolygon[-1]
 
	for clipVertex in clipPolygon:
		cp2 = clipVertex
		inputList = outputList
		outputList = []
		s = inputList[-1]
 
		for subjectVertex in inputList:
			e = subjectVertex
			if inside(e):
				if not inside(s): outputList.append(computeIntersection())
				outputList.append(e)
			elif inside(s): outputList.append(computeIntersection())
			s = e
		cp1 = cp2
		if len(outputList) == 0: return None
	return (outputList)

def iou3d(corners1, corners2):
	''' Compute 3D bounding box IoU, only working for object parallel to ground

	Input:
	    corners1: numpy array (8,3), assume up direction is negative Y
	    corners2: numpy array (8,3), assume up direction is negative Y
	Output:
	    iou: 3D bounding box IoU
	    iou_2d: bird's eye view 2D bounding box IoU

	todo (rqi): add more description on corner points' orders.
	'''
	# corner points are in counter clockwise order
	rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
	rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
	area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
	area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

	# inter_area = shapely_polygon_intersection(rect1, rect2)
	# _, inter_area = convex_hull_intersection(rect1, rect2)

	try:
		_, inter_area = convex_hull_intersection(rect1, rect2)
	except ValueError:
		inter_area = 0
	except scipy.spatial.qhull.QhullError:
		inter_area = 0

	iou_2d = inter_area/(area1+area2-inter_area)
	ymax = min(corners1[0,1], corners2[0,1])
	ymin = max(corners1[4,1], corners2[4,1])
	inter_vol = inter_area * max(0.0, ymax-ymin)
	vol1 = box3d_vol(corners1)
	vol2 = box3d_vol(corners2)
	iou = inter_vol / (vol1 + vol2 - inter_vol)
	return iou, iou_2d

@jit          
def roty(t):
	''' Rotation about the y-axis. '''
	c = np.cos(t)
	s = np.sin(t)
	return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
     
def convert_3dbox_to_8corner(bbox3d_input):
	''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
	    convert it to the 8 corners of the 3D box
	    
	    Returns:
	        corners_3d: (8,3) array in in rect camera coord
	'''
	# compute rotational matrix around yaw axis
	bbox3d = copy.copy(bbox3d_input)

	R = roty(bbox3d[3])    

	# 3d bounding box dimensions
	l = bbox3d[4]
	w = bbox3d[5]
	h = bbox3d[6]

	# 3d bounding box corners
	x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
	y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
	z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];

	# rotate and translate 3d bounding box
	corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
	#print corners_3d.shape
	corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
	corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
	corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]

	return np.transpose(corners_3d)

def load_files(data_path):
    file_list = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        file_list.append(file_path)
    
    return file_list, len(file_list)

def get_scenario_num(file_path):
    scenario_num = os.path.basename(file_path).split('.')[0]
    
    return scenario_num