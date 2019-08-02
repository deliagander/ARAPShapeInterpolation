import numpy as np
import cv2
import drawMesh
import os
import math
import time 


def computeRotationsAndTransforms(V_start, V_end, faces, no_vertices, no_faces):
	A = np.zeros((4*no_faces, 2*no_vertices))
	A_transforms = np.zeros((no_faces,2,2))
	R = np.zeros((no_faces,2,2))
	S = np.zeros((no_faces,2,2))
	angles = np.zeros(no_faces)


	for i, face in enumerate(faces):
		face = face.astype(int) - 1

		P = np.array([[V_start[face[0],0], V_start[face[0],1],1,0,0,0],
					[0,0,0,V_start[face[0],0], V_start[face[0],1],1],
					[V_start[face[1],0], V_start[face[1],1],1,0,0,0],
					[0,0,0,V_start[face[1],0], V_start[face[1],1],1],
					[V_start[face[2],0], V_start[face[2],1],1,0,0,0],
					[0,0,0,V_start[face[2],0], V_start[face[2],1],1]])


		Q = np.array([V_end[face[0],0],
					V_end[face[0],1],
					V_end[face[1],0],
					V_end[face[1],1],
					V_end[face[2],0],
					V_end[face[2],1]])

		a =  np.linalg.lstsq(P,Q, rcond=None)[0]


		A_transform = [[a[0], a[1]], 
						[a[3],a[4]]]


		R_a, d, R_bt = np.linalg.svd(A_transform)
		R_b = R_bt.T
		D = np.diag(d)

		rotation = R_a@R_b
		stretch = R_b.T@D@R_b

		if (rotation[0,1] < 0):
			angles[i] = np.arccos(rotation[0,0])
		else:	
			angles[i] = -np.arccos(rotation[0,0])

		P_inverse = np.linalg.inv(P)
		A_transforms[i,:,:] = A_transform
		R[i,:,:] = rotation
		S[i,:,:] = stretch

		for j, vertex in enumerate(face):
			vertex = (vertex).astype(int)
			A[i*4, vertex*2] = P_inverse[0,2*j]
			A[i*4, vertex*2+1] = P_inverse[0,2*j+1]
			A[i*4+1, vertex*2] = P_inverse[1,2*j]
			A[i*4+1, vertex*2+1] = P_inverse[1,2*j+1]
			A[i*4+2, vertex*2] = P_inverse[3,2*j]
			A[i*4+2, vertex*2+1] = P_inverse[3,2*j+1]
			A[i*4+3, vertex*2] = P_inverse[4,2*j]
			A[i*4+3, vertex*2+1] = P_inverse[4,2*j+1]


	return A, A_transforms, R, S, angles

def computVt(A, A_transforms, R, S, angles, t, no_faces, no_vertices, V_start, V_end, faces):

	I = np.array([[1,0],
				[0,1]])
	b = np.zeros(4*no_faces)

	V_t = np.zeros((no_vertices,2))

	A_first = A[:,:2]
	A = A[:,2:]


	inner = A_first[:,0]*((1-t)*V_start[0,0] + t*V_end[0,0]) + A_first[:,1]*((1-t)*V_start[0,1] + t*V_end[0,1])

	for i, face in enumerate(faces):

		R_t = [[np.cos(angles[i]*t), -np.sin(angles[i]*t)],
			[np.sin(angles[i]*t), np.cos(angles[i]*t)]]

		A_t = R_t@((1-t)*I + t*S[i,:,:])

		b[i*4] = A_t[0,0]
		b[i*4+1] = A_t[0,1]
		b[i*4+2] = A_t[1,0]
		b[i*4+3] = A_t[1,1]

	b = b-inner

	V = np.linalg.lstsq(A.T@A,A.T@b, rcond=None)[0]

	V_t[0,0] = (1-t)*(V_start[0,0]) + t*V_end[0,0]
	V_t[0,1] = (1-t)*(V_start[0,1]) + t*V_end[0,1]

	V_t[1:,0] = V[0::2]
	V_t[1:,1] = V[1::2]

	return V_t

# MOUSE EVENT. CLICK ANYWHERE IN THE WINDOW TO SEE FRAME BY FRAME
def frame_by_frame(event, x, y, flags, param):
	global frames, no_vertices, no_faces, V_start, faces, V_end, img, img2, count			
	drawMesh.draw_mesh_red(V_end,edges,img2)
	if count == (frames+1):
		count = 1
	if event == cv2.EVENT_LBUTTONDOWN:
		t = count/frames
		print('Time:', t)
		start_time = time.time()
		V_t = computVt(A, A_transforms, R, S, angles, t, no_faces, no_vertices, V_start, V_end, faces)
		elapsed_time = time.time() - start_time
		print('Time taken for ARAP Interplation:', elapsed_time)
		count = count+1
		img = img2.copy()
		drawMesh.draw_mesh(V_t,edges,img)
		cv2.imshow(windowName, img)


if __name__ == "__main__":
	num = 0
	windowName = 'ARAP'
	cv2.setUseOptimized(True)

	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

	# CHANGE SOURCE AND TARGET MESH HERE
	start  = open(os.path.join(__location__,'man.obj'), 'r') 
	end  = open(os.path.join(__location__,'man2.obj'), 'r') 

	no_vertices, no_faces, V_start, faces = drawMesh.read_file(start)
	_, _, V_end, faces_end = drawMesh.read_file(end)

	# CHANGE NUMBER OF FRAMES HERE
	frames = 50

	img = np.zeros((800, 1280, 3), np.uint8)
	img2 = img.copy()
	img_clear = img.copy()

	edges = drawMesh.get_edges(no_faces,faces)

	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, frame_by_frame)

	A, A_transforms, R, S, angles = computeRotationsAndTransforms(V_start, V_end, faces, no_vertices, no_faces)

	count = 1

	drawMesh.draw_mesh_red(V_end,edges,img)
	drawMesh.draw_mesh(V_start,edges,img)
	cv2.namedWindow(windowName)

	V_t = np.zeros((no_vertices,2))

	while (True):
		cv2.imshow(windowName, img)
		# PRESS SPACE BAR TO RECORD A VIDEO
		if cv2.waitKey(1) == 32:
			count = 1
			drawMesh.draw_mesh(V_start,edges,img)
			fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
			out = cv2.VideoWriter('arap_interpolation.avi',fourcc, 20, (1280,800),isColor=True)
			for k in range(frames):
				t = count/frames
				print('Time:',t)
				V_t = computVt(A, A_transforms, R, S, angles, t, no_faces, no_vertices, V_start, V_end, faces)
				count = count+1
				img = img_clear.copy()
				drawMesh.draw_mesh(V_t,edges,img)
				cv2.imshow(windowName, img)
				cv2.waitKey(1)
				out.write(img)
			
			

	cv2.destroyAllWindows()










