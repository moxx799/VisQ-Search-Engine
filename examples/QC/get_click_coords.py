# # from tkinter import *
# # from tkinter.filedialog import askopenfilename
# # from skimage.io import imread
# # from PIL import Image, ImageTk
# # import tkinter.simpledialog
# # from scipy import ndimage


# # root = Tk()

# # #setting up a tkinter canvas
# # # w = Canvas(root, width=1000, height=700)
# # w=Canvas(root, width = 14699, height = 21527)
# # w.pack()

# # #adding the image
# # # File = askopenfilename(parent=root, initialdir="./",title='Select an image') #nice but wait RWM
# # File = '/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/atlas_borders.tif'
# # original_array = imread(File) # Image.open(File)
# # xmt, ymp = original_array.shape
# # original = Image.fromarray(original_array ) #RWM because tif
# # original = original.resize((21527,14699)) #resize image
# # img = ImageTk.PhotoImage(original)
# # w.create_image(0, 0, image=img, anchor="nw")

# # Ry = xmt/21527
# # Rx = ymp/14699

# # #ask for pressure and temperature extent
# # # xmt = tkinter.simpledialog.askfloat("Temperature", "degrees in x-axis")
# # # ymp = tkinter.simpledialog.askfloat("Pressure", "bars in y-axis")

# # #ask for real PT values at origin
# # # xc = tkinter.simpledialog.askfloat("Temperature", "Temperature at origin")
# # # yc = tkinter.simpledialog.askfloat("Pressure", "Pressure at origin")

# # #instruction on 3 point selection to define grid
# # tkinter.messagebox.showinfo("Instructions", "Click: \n"
# #                                             "1) Origin \n"
# #                                             "2) Temperature end \n"
# #                                             "3) Pressure end")

# # # From here on I have no idea how to get it to work...

# # # Determine the origin by clicking
# # # def getorigin(eventorigin):
# # #     global x0,y0
# # #     x0 = eventorigin.x
# # #     y0 = eventorigin.y
# # #     print(x0,y0)
# # #     w.bind("<Button 1>",getextentx)
# # # #mouseclick event


# # # Determine the extent of the figure in the x direction (Temperature)
# # def getextentx(eventextentx):
# #     global xe
# #     xe = eventextentx.x
# #     print(xe)
# #     w.bind("<Button 1>",getextenty)

# # # Determine the extent of the figure in the y direction (Pressure)
# # def getextenty(eventextenty):
# #     global ye
# #     ye = eventextenty.y
# #     print(ye)
# #     tkinter.messagebox.showinfo("Grid", "Grid is set. You can start picking coordinates.")
# #     w.bind("<Button 1>",printcoords)

# # #Coordinate transformation into Pressure-Temperature space
# # # def printcoords(event):
# # #     xe, ye = 0, 0
# # #     xmpx = xe-x0
# # #     xm = xmt/xmpx
# # #     ympx = ye  -y0
# # #     ym = ymp/ ympx

# # #     #coordinate transformation
# # #     newx = (event.x-x0)*(xm)+xc
# # #     newy = (event.y-y0)*(ym)+yc
# # #     # newx=(event.x)*(xm)
# # #     # newy = (event.y)*(ym)

# # #     #outputting x and y coords to console
# # #     print (newx,newy)
# # #     # print(event.x, event.y)

# # def printcoords(event):
# #     # print(event.x, event.y)
# #     # finding coordinates
# #     # newx, newy= ndimage.map_coordinates(original_array, [event.x, event.y], mode='nearest')
# #     print (Rx * event.x, Ry * event.y)
# #     # print (newx,newy)

# # #mouseclick event
# # w.bind("<Button 1>",printcoords)
# # # w.bind("<Button 1>",getextentx)

# # root.mainloop()

# # importing the module
# import cv2
# from skimage.io import imread

# # function to display the coordinates of
# # of the points clicked on the image
# def click_event(event, x, y, flags, params):

# 	# checking for left mouse clicks
# 	if event == cv2.EVENT_LBUTTONDOWN:

# 		# displaying the coordinates
# 		# on the Shell
# 		print(x, ' ', y)

# 		# displaying the coordinates
# 		# on the image window
# 		font = cv2.FONT_HERSHEY_SIMPLEX
# 		cv2.putText(img, str(x) + ',' +
# 					str(y), (x,y), font,
# 					1, (255, 0, 0), 2)
# 		# cv2.imshow('image', img)
# 		# Custom window
# 		cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
# 		cv2.imshow('custom window', img)
# 		cv2.resizeWindow('custom window', 800, 400)

# 	# checking for right mouse clicks
# 	if event==cv2.EVENT_RBUTTONDOWN:

# 		# displaying the coordinates
# 		# on the Shell
# 		print(x, ' ', y)

# 		# displaying the coordinates
# 		# on the image window
# 		font = cv2.FONT_HERSHEY_SIMPLEX
# 		b = img[y, x, 0]
# 		g = img[y, x, 1]
# 		r = img[y, x, 2]
# 		cv2.putText(img, str(b) + ',' +
# 					str(g) + ',' + str(r),
# 					(x,y), font, 1,
# 					(255, 255, 0), 2)
# 		cv2.imshow('image', img)

# # driver function
# if __name__=="__main__":

# 	# reading the image
# 	# img = cv2.imread('lena.jpg', 1)
#     File = '/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/atlas_borders.tif'
#     img = imread(File)
#     # displaying the image
#     cv2.imshow('image', img)

# 	# setting mouse handler for the image
# 	# and calling the click_event() function
#     cv2.setMouseCallback('image', click_event)

# 	# wait for a key to be pressed to exit
#     cv2.waitKey(0)

# 	# close the window
#     # Custom window
# #cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
# #cv2.imshow('custom window', image)
# #cv2.resizeWindow('custom window', 200, 200)
# #cv2.destroyAllWindows()


import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize 

dst = '/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/clicks/clicks_Swanson0.npy'

click_list=[]
#File = '/project/roysam/rwmills/data/brain/50Plex/S1/final/S1_R2C4.tif'
original_array=imread('/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/Swanson_S1_Atlas_Fit.tif')
original_array= resize(original_array, (int(29398/ 12), int(43054/ 12)),anti_aliasing=True)
#File = '/project/roysam/rwmills/data/brain/50Plex/S1/final/S1_R2C4.tif' #neun
#'/project/roysam/rwmills/repos/cluster-contrast-reid/examples/data/atlas_borders.tif'
#original_array = imread(File) # Image.open(File)
vmin =np.amin(original_array)
vmax =np.amax(original_array) /2
mutable_object = {}
fig = plt.figure()
def onclick(event):
    print('you pressed', event.key, event.xdata, event.ydata)
    X_coordinate = event.xdata
    Y_coordinate = event.ydata
    mutable_object['click'] = X_coordinate
    click_list.append((int(event.ydata), int(event.xdata) ))

cid = fig.canvas.mpl_connect('button_press_event', onclick)
# lines, = plt.plot([1,2,3])
plt.imshow(original_array, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()

X_coordinate = mutable_object['click']
print(X_coordinate)
click_list = np.array(click_list)

np.save(dst,click_list )
