#=============================================
# File Name: PC3_ex3_pyqtgraph_xyz_playback.py
#
# Requirement:
# Hardware: BM201-ISK or BM501-AOP
# Firmware: PC3-I471
# playback kit(hardware): toolkit-PC3-AOP
#
# lib: pc3
# plot tools: pyqtgraph
# Plot point cloud(V6) in 3D figure 
# type: Raw data
# Baud Rate: playback: 119200
#			 real time: 921600
#
#=============================================

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

import pyqtgraph as pg
import numpy as np
from mmWave import pc3_v2
import serial
from threading import Thread


from datetime import date,datetime,time
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import tkinter as tk
import time
import sys
class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
	def __init__(self, X, Y, Z, text):
		gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
		self.text = text
		self.X = X
		self.Y = Y
		self.Z = Z

	def setGLViewWidget(self, GLViewWidget):
		self.GLViewWidget = GLViewWidget

	def setText(self, text):
		self.text = text
		self.update()

	def setX(self, X):
		self.X = X
		self.update()

	def setY(self, Y):
		self.Y = Y
		self.update()

	def setZ(self, Z):
		self.Z = Z
		self.update()

	def paint(self):
		a = 0
		'''
		# For some version gl is ok some get following error
		#Error while drawing item <__main__.CustomTextItem object at 0x7fe379b51a60>
		
		self.GLViewWidget.qglColor(QtCore.Qt.cyan)
		self.GLViewWidget.renderText(round(self.X), round(self.Y), round(self.Z), self.text)
		'''

st = datetime.now()
sim_startFN = 0
sim_stopFN = 0

# colorSet = [[255,255, 0,255], [0, 255, 0, 255], [0, 100, 255, 255], [248, 89, 253, 255], [89, 253, 242, 255],[89, 253, 253, 255],
# 		  [253, 89, 226, 255],[253, 229, 204, 255],[51, 255, 255, 255],[229, 204, 255, 255], [89, 253, 100, 255], 
# 		  [127, 255, 212, 255], [253, 165, 89, 255],[255, 140, 0, 255],[255, 215, 0, 255],[0, 0, 255, 255]]

colorSet = [[1.0,1.0, 0,1.0], [0, 1.0, 0, 1.0], [0, 0.4, 1.0, 1.0], [0.97, 0.35, 1.0, 1.0], [0.35, 0.99, 0.99, 1.0],
			[0.99, 0.35, 0.88, 1.0],[0.99, 0.9, 0.8, 1.0],[0.2, 1.0, 1.0, 1.0],[0.9, 0.8, 1.0, 1.0], [0.35, 0.99, 0.4, 1.0], 
			[0.5, 1.0, 0.83, 1.0], [0.99, 0.64, 0.35, 1.0],[0.35, 0.9, 0.75, 1.0],[1.0, 0.5, 0, 1.0],[1.0, 0.84, 0, 1.0],[0, 0, 1.0, 1.0]]

class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None):
        GLGraphicsItem.__init__(self)

        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = Z
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtCore.Qt.white)
        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)

class Custom3DAxis(gl.GLAxisItem):
	#Class defined to extend 'gl.GLAxisItem'
	def __init__(self, parent, color=(0.0,0.0,0.0,.6)):
		gl.GLAxisItem.__init__(self)
		self.parent = parent
		self.c = color
		
	def add_tick_values(self, xticks=[], yticks=[], zticks=[]):
		#Adds ticks values. 
		x,y,z = self.size()
		xtpos = np.linspace(0, x, len(xticks))
		ytpos = np.linspace(0, y, len(yticks))
		ztpos = np.linspace(0, z, len(zticks))
		#X label
		for i, xt in enumerate(xticks):
			val = CustomTextItem((xtpos[i]), Y= 0, Z= 0, text='{}'.format(xt))
			val.setGLViewWidget(self.parent)
			self.parent.addItem(val)
		#Y label
		for i, yt in enumerate(yticks):
			val = CustomTextItem(X=0, Y=round(ytpos[i]), Z= 0, text='{}'.format(yt))
			val.setGLViewWidget(self.parent)
			self.parent.addItem(val)
		#Z label
		for i, zt in enumerate(zticks):
			val = CustomTextItem(X=0, Y=0, Z=round(ztpos[i]), text='{}'.format(zt))
			val.setGLViewWidget(self.parent)
			self.parent.addItem(val)

class Custom3Dlabel(gl.GLGraphicsItem.GLGraphicsItem):
	#Class defined to extend 'gl.GLAxisItem'
	def __init__(self, parent, color=(0.0,0.0,0.0,.6)):
		gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
		self.parent = parent
		self.c = color
		
	#def add_values(self, xticks=[], yticks=[], zticks=[]):
	def set_values(self,text, x,y,z):
		val = CustomTextItem(X = x, Y= y, Z= z, text='{}'.format(text))
		val.setGLViewWidget(self.parent)
		self.parent.addItem(val)

##################### Parameter ################################### 
QUEUE_LEN = 15

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()


# gl_txt = GLTextItem(10, 10, 10, "FALL STATE:")
# gl_txt.setGLViewWidget(w)
# w.addItem(gl_txt)

# ALERT: Assume RADAR board tilt 0 degree
# ALERT: This value may be changed depends on your RADAR installation

# JB_RADAR_INSTALL_HEIGHT = 2.00
JB_RADAR_INSTALL_HEIGHT = 2.6
H0 = JB_RADAR_INSTALL_HEIGHT
H1=1.5
H2=0.5
fall_state = 0
FALL_THRESHOLD = -0.4
FALL_DURATION = 3 # warning 3 seconds
FALL_SEC = 0

####### create box to represent device ######
verX = 0.0625
verY = 0.05
verZ = 0.125
zOffSet =  JB_RADAR_INSTALL_HEIGHT
verts = np.empty((2,3,3))
verts[0,0,:] = [-verX, 0,  verZ + zOffSet]
verts[0,1,:] = [-verX, 0, -verZ + zOffSet]
verts[0,2,:] = [verX,  0, -verZ + zOffSet]
verts[1,0,:] = [-verX, 0,  verZ + zOffSet]
verts[1,1,:] = [verX,  0,  verZ + zOffSet]
verts[1,2,:] = [verX,  0, -verZ + zOffSet]
 
evmBox = gl.GLMeshItem(vertexes=verts,smooth=False,drawEdges=True,edgeColor=pg.glColor('r'),drawFaces=False)
w.addItem(evmBox)



#############################################

#size=50:50:50
g = gl.GLGridItem()
g.setSize(x=50,y=50,z=50)
#g.setSpacing(x=1, y=1, z=1, spacing=None)
w.addItem(g)

axis = Custom3DAxis(w, color=(0.2,0.2,0.2,1.0))
axis.setSize(x=5, y=5, z=5)
xt = [0,1,2,3,4,5]  
axis.add_tick_values(xticks=xt, yticks=xt, zticks=xt)
w.addItem(axis)

################### Read file ############################
POS_playback = True

v6_col_names = ['time','fN','type','elv','azimuth','doppler','range' ,'snr','sx', 'sy', 'sz']
if POS_playback == True:
	v7_col_names = ['time','fN','type','posX','posY','posZ','velX','velY','velZ','accX','accY','accZ','ec0','ec1','ec2','ec3','ec4','ec5','ec6','ec7','ec8','ec9','ec10','ec11','ec12','ec13','ec14','ec15','g','confi','tid']
else:
	v7_col_names = ['time','fN','type','posX','posY','velX','velY','accX','accY','posZ','velZ','accZ','tid']
v8_col_names = ['time','fN','type','targetID']

def getRecordData(frameNum):
	s_fn = frameNum + sim_startFN
	#print("frame number:{:}".format(s_fn))
	v6d = v6simo[v6simo['fN'] == s_fn]
	#v6d =  v6dd[v6dd['doppler'] < 0.0]
	#print(v6d)
	v7d = v7simo[v7simo['fN'] == s_fn]
	v8d = v8simo[v8simo['fN'] == s_fn]
	chk = 0
	if v6d.count != 0:
		chk = 1
	return (chk,v6d,v7d,v8d)

def readFile(fileName):
	global sim_startFN,sim_stopFN,v6simo,v7simo,v8simo

	df = pd.read_csv(fileName)
	df.dropna()
	df = df.drop(df[df['type'] == 'v7'].index)
	if POS_playback == True:
		df = df.drop(columns=['accZ','ec0','ec1','ec2','ec3','ec4','ec5','ec6','ec7','ec8','ec9','ec10','ec11','ec12','ec13','ec14','ec15','g','confi','tid'])
	else:
		df = df.drop(columns=['accZ','tid'])
	df.columns = v6_col_names
	print(df.info())
	print(df.info(memory_usage="deep"))
	v6simOri = df[(df.type == 'v6')]
	#print("-------------------v6sim------------:{:}".format(v6simOri.shape))

	v6simo = v6simOri.loc[:,['fN','type','elv','azimuth','doppler','range' ,'snr','sx', 'sy', 'sz']] # in this case
	v6simo['elv'] = v6simo['elv'].astype(float, errors = 'raise')

	#------------- v7 sim ---------------
	df7 = pd.read_csv(fileName)
	df7 = df7.drop(df7[df7['type'] == 'v6'].index)
	df7.columns = v7_col_names
	v7simc = df7[df7['type'] == 'v7']
	if POS_playback == True:
		v7simo  = v7simc.loc[:,['fN','posX','posY','posZ','velX','velY','velZ','accX','accY','accZ','ec0','ec1','ec2','ec3','ec4','ec5','ec6','ec7','ec8','ec9','ec10','ec11','ec12','ec13','ec14','ec15','g','confi','tid']]
	else:
		v7simo  = v7simc.loc[:,['fN','type','posX','posY','velX','velY','accX','accY','posZ','velZ','accZ','tid']]
		v7simo['posX'] = v7simo['posX'].astype(float, errors = 'raise') 

	#------------- v8 sim ---------------
	v8simc = df[df['type'] == 'v8']
	v8simo  = v8simc.loc[:,['fN','type','elv']]
	v8simo.columns = ['fN','type','targetID']
	#print(v8simo)

	sim_startFN = df['fN'].values[0]
	sim_stopFN  = df['fN'].values[-1]

	return (v6simo,v7simo,v8simo)

################### Real Time or read from file switch ************
rtSwitch = False # real time mode
# rtSwitch = False  # read data from file
#
#use USB-UART
#port = serial.Serial("/dev/ttyUSB0",baudrate = 921600, timeout = 0.5)
#
#for Jetson nano UART port
#port = serial.Serial("/dev/ttyTHS1",baudrate = 921600, timeout = 0.5) 
#
#for pi 4 UART port
#port = serial.Serial("/dev/ttyS0",baudrate = 921600, timeout = 0.5)
#
#Drone Object Detect Radar initial 
#port = serial.Serial("/dev/tty.usbmodemGY0052534",baudrate = 921600, timeout = 0.5)
#port = serial.Serial("/dev/tty.SLAB_USBtoUART3",baudrate = 921600, timeout = 0.5)  
#for NUC ubuntu 
#port = serial.Serial("/dev/ttyACM1",baudrate = 921600, timeout = 0.5)

#port = serial.Serial("/dev/tty.usbmodem14203",baudrate = 115200 , timeout = 0.5)
port = 0 

radar = pc3_v2.Pc3_v2(port)

v6len = 0
v7len = 0
v8len = 0

pos = np.zeros((100,3))
color = [1.0, 0.0, 0.0, 1.0]
sp1 = gl.GLScatterPlotItem(pos=pos,color=color,size = 4.0)
w.addItem(sp1)

sp2 = gl.GLScatterPlotItem(pos=pos,color=color,size = 10.0)
w.addItem(sp2)

#for playback use

#(v6smu,v7smu,v8smu) = radar.readFile("pc3_2021-12-19-21-25-28.csv")
# (v6smu,v7smu,v8smu) = readFile("/home/user/workspace/1_radmoX/mmWave/PC3_v2/fall_data/pc3_2022-04-20-16-36-13.csv")

if len(sys.argv) > 1:
    fileName = sys.argv[1]
else:
	fileName = "pc3_2022-04-21-15-55-14.csv"
(v6smu,v7smu,v8smu) = readFile("../fall_data/fall_data/" + fileName)


print("------------------ v6smu --------start:{:}  stop:{:}--------------".format(sim_startFN,sim_stopFN))
print(v6smu)
print("------------------ v7smu ----------------------")
print(v7smu)
print("------------------ v8smu ----------------------")
print(v8smu)

pos1 = np.empty((50,3))
pos2 = np.empty((50,3))
uFlag1 = True
uFlag2 = True
gcolorA = []
def update():
	## update volume colors
	global pos1,pos2,lblA,color,uFlag1,uFlag2,gcolorA
	if uFlag1 == True:
		uFlag1 = False
		sp1.setData(pos=pos1,color=color)

	if uFlag2 == True:
		uFlag2 = False
		gcolor = np.array(gcolorA)
		sp2.setData(pos=pos2,color=gcolor)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(150)
fn = 0 
fn_step = 0
prev_fn = 0
lblA =[]
fn_state = False
def showData(dck,v6i,v7i,v8i):
	if dck:
		v6len = len(v6i)
		v7len = len(v7i)
		v8len = len(v8i)
		print("Sensor Data: [v6,v7,v8]:[{:d},{:d},{:d}]".format(v6len,v7len,v8len))
		if v6len > 0:
			print("\n--------v6-----------fn:{:} len({:})".format(fn,v6len))
			print(v6i)
		if v7len > 0:
			print("\n--------v7-----------fn:{:} len({:})".format(fn,v7len))
			print(v7i)
		if v8len > 2:
			print("\n--------v8-----------fn:{:} len({:})".format(fn,v8len-2))
			print(v8i)
			
locBuf = []
objBuf = pd.DataFrame([], columns=['fN','posX','posY','posZ','tid'])
v6_state = False
v7_state = False
v6_playback_state = False
v7_playback_state = False
all_playback_state = False

def radarExec():
	global v6len,v7len,v8len,pos1,pos2,prev_fn,color,flag1,flag2,uFlag1,uFlag2,sim_stopFN,fn,fn_step,fn_state,objBuf,locBuf,JB_RADAR_INSTALL_HEIGHT,QUEUE_LEN,colorSet,gcolorA,lblTextA
	v6 = []
	v7 = []
	v8 = []
	sample_point = 6
	flag1 = True
	flag2 = True
	# (dck,v6,v7, v8)  = radar.tlvRead(False,df = 'DataFrame')
	
	#(playback) 
	
	print("JB> fn:{:}    start:{:}   stop:{:}".format(fn,sim_startFN,sim_stopFN))
	hdr = radar.getHeader()
	fn = int(fn_step)
	(dck,v6,v7,v8) = getRecordData(fn)

	# w.setWindowTitle("fN = {:d}".format(sim_startFN+fn_step))

	fn_step = fn_step + 1
	time.sleep(0.2)

	if (sim_startFN+fn_step) > sim_stopFN:
		fn_step = 0
		w.setWindowTitle("fN = {:d}".format(sim_startFN+fn_step))
		
	if (fn == 0):
		objBuf = objBuf.iloc[0:0]

	#showData(dck,v6,v7,v8)
	

	if  fn != prev_fn:
		print("--------------{:}-----------".format(fn))
		prev_fn = fn
		v8len = len(v8)
		v6len = len(v6)
		v7len = len(v7)
		
		# print("Sensor Data: [v6,v7,v8]:[{:d},{:d},{:d}]".format(v6len,v7len,v8len-2))
		if v6_state == True:
			if v6len != 0 and flag1 == True:
				flag1 = False
				posTemp = v6
				v6Temp = v6
				v6op = v6    #v6Temp[(v6Temp.sx > -0.5) & (v6Temp.sx < 0.5) & (v6Temp.sy < 1.0) & (v6Temp.doppler != 0) ]
				d = v6op.loc[:,['sx','sy','sz']] 
				dd = v6op.loc[:,['sx','sy','sz','doppler']] 
				
				#(1.2)DBSCAN 
				#d_std = StandardScaler().fit_transform(xy6A)
				if len(d) > sample_point:
					d_std = StandardScaler().fit_transform(d)
					
					#db = DBSCAN(eps=1.4, min_samples=3).fit(d_std)
					db = DBSCAN(eps= 0.5, min_samples=sample_point).fit(d_std) # 1.2
					# db = DBSCAN(eps=0.15, min_samples=6).fit(d_std)  # 
						
					labels = db.labels_  #cluster ID
					
					dd_np = dd.to_numpy()
					#(1.3)insert labels to sensor temp Array(stA) stA = [x,y,z,Doppler,labels]
					stA = np.insert(dd_np,4,values=labels,axis= 1) #[x,y,z,Doppler,labels]
					# print("==[{:}]========== stA =====d:{:}=======stA:{:}".format(fn,d.shape,stA.shape))
					print(stA)
					mask = (labels == -1)
					sensorA = []
					sensorA = stA[~mask]
					# print("==[{:}]====== sensorA ========={:}".format(fn,sensorA.shape))
					# print(sensorA)
					lblA = sensorA[:,4]
					
					dm = d[~mask] #
					pos_np = dm.to_numpy()
					pos_np[:,2] += JB_RADAR_INSTALL_HEIGHT #0.5
					#print(d)
					
					color = np.empty((len(lblA),4), dtype=np.float32)
					for i in range(len(lblA)):
						x = int(lblA[i])
						color[i] = colorSet[x] 
					
					pos1 = pos_np
					
					lbs = labels[~mask] 
					print("mask set:{:}".format(set(lbs)))
					cnt = 0
					for k in set(lbs):
						gpMask = (lbs == k)
						m = sensorA[gpMask]
						mA = (np.mean(m,axis=0))
						# print(mA)
						# print("Get 3D Box: k:{:} box= \n{:}".format(k,get3dBox(sensorA[gpMask])))
						(x,xl,y,yl,_,_,nop) = get3dBox(sensorA[gpMask])
					uFlag1 = True
					flag1 = True
		else:
			pos1 = np.empty((50,3))
			uFlag1 = True
			flag1 = True

		if v7_state == True:
			if v7len != 0 and flag2 == True:
				flag2 = False
				
				#(1.1) insert v7 to data Queue(objBuf) 
				objBuf = objBuf.append(v7.loc[:,['fN','posX','posZ','posY','tid']], ignore_index=True)
				locBuf.insert(0,fn)
				if len(locBuf) > QUEUE_LEN:
					objBuf = objBuf.loc[objBuf.fN != locBuf.pop()]
				#print("========objBuf:len:{:}".format(len(objBuf)))
				#print(locBuf)
				print("objBuf: ", objBuf)				
				
				# fall detection algorithm
				fall_detection(objBuf) 
				print("------------FALL STATE > {:}-------------".format(fall_state))
				#(1.2)set color based on tid
				tidA = objBuf['tid'].values.tolist()  #tidA.astype(int)
				gcolorA = []
				
				for i in range(len(tidA)):
					gcolorA.append(colorSet[int(tidA[i])%15])
					
					#labeling id in target 
					#idString = "" # "id{:}".format(int(tidA[i])%15)
					#lblTextA.append(idString)					
				
				#(1.3)TargetID 
				xBuf = objBuf.loc[:,['posX','posY','posZ']]  # WALL MOUNT
				pos_np2 = xBuf.to_numpy()
				
				#Radar install position
				pos_np2[:,2] =  JB_RADAR_INSTALL_HEIGHT + pos_np2[:,2]  # WALL MOUNT
				
				pos2 = pos_np2
				# objBuf = objBuf.iloc[0:0]
				
				uFlag2 = True
				flag2 = True
		else:
			pos2 = np.empty((50,3))
			uFlag2 = True
			flag2 = True
		 
	#port.flushInput()


def fall_detection(yBuf):
	global fall_state,H0,H1,H2, FALL_SEC
	# print("v7len: ", len(yBuf))
	frame_period = 0.05
	print(yBuf)
	if len(yBuf)> 3:
		curFN = yBuf.iloc[-1]["fN"]
		preFN = yBuf.iloc[-2]["fN"]
		lastFN = yBuf.iloc[-4]["fN"]
		curH = yBuf.iloc[-1]['posZ']
		lastH = yBuf.iloc[-4]['posZ']
		# velocity is minus which means the direction from top to zero
		vel = (curH-lastH)/((curFN-lastFN)*frame_period) 
		h=H0-curH
		print("fN:{:}, current H- last H: {:}, vel:{:}".format(yBuf.iloc[-1]['fN'], curH-lastH, vel))
		if fall_state==0:
			if h>H1:
				fall_state = 1
			elif h <=H1 and h >=H2:
				fall_state = 2
			elif h <=H2:
				fall_state = 3
		elif fall_state == 1:
			if vel < FALL_THRESHOLD:
				fall_state = 4 # FALL EVENT
			if h>H1:
				fall_state = 1
			elif h <=H2:
				fall_state = 3
		elif fall_state == 2:
			if vel < FALL_THRESHOLD:
				fall_state = 4
			if h >=H1:
				fall_state=1
			elif h <=H2:
				fall_state =3
		elif fall_state ==3 :
			if vel < FALL_THRESHOLD:
				fall_state = 4
			elif h <=H1 and h >=H2:
				fall_state = 3
			elif h > H1:
				fall_state = 1
		elif fall_state == 4:
			# frame period = fn*50ms
			FALL_SEC += (curFN - preFN)*frame_period
			if FALL_SEC  > FALL_DURATION:
				fall_state = 0
				FALL_SEC = 0
			if h >=H1:
				fall_state=1
				FALL_SEC = 0
	

def get3dBox(targetCloud): 
	xMax = np.max(targetCloud[:,0])
	xr   = np.min(targetCloud[:,0])
	xl = np.abs(xMax-xr)

	yMax = np.max(targetCloud[:,1])
	yr = np.min(targetCloud[:,1])
	yl = np.abs(yMax-yr)
	
	zMax = np.max(targetCloud[:,2])
	zr = np.min(targetCloud[:,2])
	zl = np.abs(zMax-zr)
	
	nop = len(targetCloud)
	return (xr,xl,yr,yl,zr,zl,nop)	
		 
def uartThread(name):
	#port.flushInput()
	while True:
		radarExec()
					
# thread1 = Thread(target = uartThread, args =("UART",))
# thread1.setDaemon(True)
# thread1.start()

def text():
	window = tk.Tk()
	window.title('window')
	window.geometry('500x300')
	
	def v6_playback():
		global v6_state,v6_playback_state,all_playback_state

		if all_playback_state == False:
			if v6_state == False:
				v6_state = True
				v6_playback_state = True
			else:
				v6_state = False
				v6_playback_state = False				
				
	def v7_playback():
		global v7_state,v7_playback_state,all_playback_state

		if all_playback_state == False:
			if v7_state == False:
				v7_state = True
				v7_playback_state = True
			else:
				v7_state = False
				v7_playback_state = False

	def all_playback():
		global v6_state,v7_state,all_playback_state

		if v6_playback_state == False and v7_playback_state == False:
			if v6_state == False and v7_state == False:
				v6_state = True
				v7_state = True
				all_playback_state = True
			else:
				v6_state = False
				v7_state = False
				all_playback_state = False
	def play():
		global v7_state, v7_playback_state
		if v7_state == False:
			v7_state = True
			v7_playback_state = True
		radarExec()
	b2 = tk.Button(window, text='v6 playback', width=10,
				height=2, command=v6_playback)

	b3 = tk.Button(window, text='v7 playback', width=10,
				height=2, command=v7_playback)

	b4 = tk.Button(window, text='all playback', width=10,
				height=2, command=all_playback)	
	b5 = tk.Button(window, text='single frame', width=10,
				height=2, command=play)	
	b2.pack()
	b3.pack()
	b4.pack()
	b5.pack()	
	# 第8步，主視窗迴圈顯示
	window.mainloop()

thread2 = Thread(target = text)
thread2.setDaemon(True)
thread2.start()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
	import sys
	if (sys.flags.interactive != 1) or not hasattr(QtCore,'PYQT_VERSION'):
		QtGui.QApplication.instance().exec_()
