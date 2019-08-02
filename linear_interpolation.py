import numpy as np
import cv2
import drawMesh
import os
import time

# CLICK ANYWHERE IN THE WINDOW TO GO THROUGH FRAME BY FRAME
def frame_by_frame(event, x, y, flags, param):
	global frames, no_vertices, no_faces, V_start, faces, V_end, img, img2, count			
	drawMesh.draw_mesh_red(V_end,edges,img2)
	if count == (frames+1):
		count = 1
	if event == cv2.EVENT_LBUTTONDOWN:
		t = count/frames
		print(t)
		start_time = time.time()
		for v, vertex in enumerate(V_start):
			V_t[v,0] = (1-t)*(vertex[0]) + t*V_end[v,0]
			V_t[v,1] = (1-t)*(vertex[1]) + t*V_end[v,1]
		elapsed_time = time.time() - start_time
		print('Time taken for Linear Interplation:', elapsed_time)
		count = count+1
		img = img2.copy()
		drawMesh.draw_mesh(V_t,edges,img)
		cv2.imshow(windowName, img)


if __name__ == "__main__":
	num = 0
	windowName = 'LINEAR'
	cv2.setUseOptimized(True)

	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

	# CHANGE SOURCE AND TARGET MESH HERE
	start  = open(os.path.join(__location__,'man.obj'), 'r') 
	end  = open(os.path.join(__location__,'man0.obj'), 'r') 

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
			out = cv2.VideoWriter('linear_interpolation.avi',fourcc, 20, (1280,800),isColor=True)
			
			for k in range(frames):
				t = count/frames
				print(t)
				for v, vertex in enumerate(V_start):
					V_t[v,0] = (1-t)*(vertex[0]) + t*V_end[v,0]
					V_t[v,1] = (1-t)*(vertex[1]) + t*V_end[v,1]
				count = count+1
				img = img_clear.copy()
				drawMesh.draw_mesh(V_t,edges,img)
				cv2.imshow(windowName, img)
				cv2.waitKey(1)
				out.write(img)




	out.release()
	cv2.destroyAllWindows()
