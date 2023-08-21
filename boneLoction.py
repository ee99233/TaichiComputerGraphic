from random import Random
from turtle import shape
import taichi as ti
import taichi.math as tm
from math import pi
import numpy as np
ti.init(arch=ti.gpu, debug=True);
vertics=ti.Vector.field(3,dtype=float,shape=4);
indices=ti.field(int,shape=8);
TempRotation=ti.Vector.field(3,dtype=float,shape=4);
colors = ti.Vector.field(3, dtype=float, shape=4);
TempLocation = ti.Vector([-2.0,-3.0,0.0])
TargetLocation = ti.Vector([0.0,0.0,0.0])
@ti.data_oriented
class XlsBoneArray: 
    def __init__(self):
        self.trans = ti.Matrix.field(3,3,dtype=ti.f32,shape=(4))
        self.rotator=ti.Vector.field(n=3,dtype=ti.f32,shape=(4))
        self.XMat = ti.Matrix.field(3,3,dtype=ti.f32,shape=(4))
        self.YMat = ti.Matrix.field(3,3,dtype=ti.f32,shape=(4))
        self.ZMat = ti.Matrix.field(3,3,dtype=ti.f32,shape=(4))
        self.Q = ti.Matrix.field(3,3,dtype=ti.f32,shape=(4))
        self.LocalLocation=ti.Vector.field(n=3,dtype=ti.f32,shape=(4))
        self.WorldLocation=ti.Vector.field(n=3,dtype=ti.f32,shape=(4))
        self.ActorLoction=ti.Vector([-10.0,0.0,4.0])
    @ti.kernel
    def init(self):
        self.LocalLocation[0]=ti.Vector([0.0,0.0,0.0])
        self.LocalLocation[1]=ti.Vector([5.0,0.0,0.0])
        self.LocalLocation[2]=ti.Vector([5.0,0.0,0.0])
        self.LocalLocation[3]=ti.Vector([5.0,0.0,0.0])
        n=0
        while(n<4):
         if(n==0):
          self.WorldLocation[n]= self.ActorLoction+self.LocalLocation[n]
         else:
          self.WorldLocation[n]=self.WorldLocation[n-1]+self.LocalLocation[n]
          #print('SELF W=',self.WorldLocation[n],'SELF FORWD',self.WorldLocation[n-1])
         n+=1
    @ti.kernel
    def UpdateLocation(self):
        n=0
        self.WorldLocation[0]= self.ActorLoction+self.LocalLocation[0]
        while(n<3):
            xrad=self.rotator[n][0]/180.0*pi
            self.XMat[n][0,0]=1.0
            self.XMat[n][1,1]=ti.cos(xrad)
            self.XMat[n][1,2]=-ti.sin(xrad)
            self.XMat[n][2,1]=ti.sin(xrad)
            self.XMat[n][2,2]=ti.cos(xrad)
            yrad=self.rotator[n][1]/180.0*pi
            self.YMat[n][1,1]=1.0
            self.YMat[n][0,0]=ti.cos(yrad)
            self.YMat[n][0,2]=ti.sin(yrad)
            self.YMat[n][2,0]=-ti.sin(yrad)
            self.YMat[n][2,2]=ti.cos(yrad)
            zrad=self.rotator[n][2]/180.0*pi
            self.ZMat[n][2,2]=1.0
            self.ZMat[n][0,0]=ti.cos(zrad)
            self.ZMat[n][0,1]=-ti.sin(zrad)
            self.ZMat[n][1,0]=ti.sin(zrad)
            self.ZMat[n][1,1]=ti.cos(zrad)
            if(n<3):
               q=self.ZMat[n]@self.YMat[n]@self.XMat[n]
               if n>0:
                  self.Q[n]=self.Q[n-1]@q
                  #print('Qn=',self.Q[n],'i=',n)
                  #print('q=',q)
               else:
                  self.Q[n]=q
                  #print('Qn=',self.Q[n])
               new_x=ti.Vector([0.0,0.0,0.0])
               for i in range(3):
                  for j in range(3):
                     new_x[i]+=self.Q[n][i,j]*self.LocalLocation[n+1][j] 
               self.WorldLocation[n+1]= new_x+self.WorldLocation[n]
               #print("i=",n+1,"world location",self.WorldLocation[n+1])        
            n+=1
    @ti.kernel
    def GetLocation(self,n:int)->ti.math.vec3:
       return  self.WorldLocation[n]
    @ti.func
    def Fk(self,n:int):
        index=n
        while(n<3):
            xrad=self.rotator[n][0]/180.0*pi
            self.XMat[n][0,0]=1.0
            self.XMat[n][1,1]=ti.cos(xrad)
            self.XMat[n][1,2]=-ti.sin(xrad)
            self.XMat[n][2,1]=ti.sin(xrad)
            self.XMat[n][2,2]=ti.cos(xrad)
            yrad=self.rotator[n][1]/180.0*pi
            self.YMat[n][1,1]=1.0
            self.YMat[n][0,0]=ti.cos(yrad)
            self.YMat[n][0,2]=ti.sin(yrad)
            self.YMat[n][2,0]=-ti.sin(yrad)
            self.YMat[n][2,2]=ti.cos(yrad)
            zrad=self.rotator[n][2]/180.0*pi
            self.ZMat[n][2,2]=1.0
            self.ZMat[n][0,0]=ti.cos(zrad)
            self.ZMat[n][0,1]=-ti.sin(zrad)
            self.ZMat[n][1,0]=ti.sin(zrad)
            self.ZMat[n][1,1]=ti.cos(zrad)
            if(n<3):
               q=self.ZMat[n]@self.YMat[n]@self.XMat[n]
               if n>0:
                  self.Q[n]=self.Q[n-1]@q
               else:
                  self.Q[n]=q
               new_x=ti.Vector([0.0,0.0,0.0])
               for i in range(3):
                  for j in range(3):
                    new_x[i]+=self.Q[n][i,j]*self.LocalLocation[n+1][j] 
               self.WorldLocation[n+1]= new_x+self.WorldLocation[n]    
            n+=1
    @ti.kernel
    def UpdateRotation(self,n:int,Rotation:ti.math.vec3):
       self.rotator[n][0]=Rotation[0]
       self.rotator[n][1]=Rotation[1]
       self.rotator[n][2]=Rotation[2]
    @ti.kernel
    def UpdateIk(self,TargetLocation:ti.math.vec3):
       n=2
       iswhile=True
       xmin=0.5
       Xlength=(self.WorldLocation[3]-TargetLocation).norm()
       if((self.WorldLocation[0]-TargetLocation).norm()>15.0):
          xmin=(self.WorldLocation[0]-TargetLocation).norm()-15.0
          xmin=xmin
          print(xmin)
       print('xxxxx=',Xlength)
       if(Xlength<xmin):
         iswhile=False
       while(Xlength>xmin):
          deltav1=(TargetLocation-self.WorldLocation[n]).normalized()
          detatav2=(self.WorldLocation[3]-self.WorldLocation[n]).normalized()
          axis=ti.math.cross(deltav1,detatav2).normalized()
          angle=ti.math.acos(ti.math.dot(deltav1,detatav2))*180.0/pi
          rot_Mat=self.axisRotator(axis,angle)
          self.Q[n]=self.Q[n]@rot_Mat
          self.rotator[n]=self.toEuler(self.Q[n])
          self.Fk(n)
          Xlength=(self.WorldLocation[3]-TargetLocation).norm()
          print('xxx=',Xlength,"xmin",xmin)
          n-=1
          if(n<0):
            n=2


    @ti.func 
    def axisRotator(self,axis:ti.math.vec3,angel:float)->ti.types.matrix(3,3,ti.f32):
       seta=angel/180.0*pi
       axisnormal=axis.normalized()
       axs_rot=ti.Matrix([[0]*3 for _ in range(3)],ti.float32)
       axs_rot[0,0]=axisnormal[0]*axisnormal[0]*(1.0-ti.math.cos(seta))+ti.math.cos(seta)
       axs_rot[0,1]=axisnormal[0]*axisnormal[1]*(1.0-ti.math.cos(seta))-axisnormal[2]*ti.math.sin(seta)
       axs_rot[0,2]=axisnormal[0]*axisnormal[2]*(1.0-ti.math.cos(seta))+axisnormal[1]*ti.math.sin(seta)
       axs_rot[1,0]=axisnormal[0]*axisnormal[1]*(1.0-ti.math.cos(seta))+axisnormal[2]*ti.math.sin(seta)
       axs_rot[1,1]=axisnormal[1]*axisnormal[1]*(1.0-ti.math.cos(seta))+ti.math.cos(seta)
       axs_rot[1,2]=axisnormal[1]*axisnormal[2]*(1.0-ti.math.cos(seta))-axisnormal[0]*ti.math.sin(seta)
       axs_rot[2,0]=axisnormal[0]*axisnormal[2]*(1.0-ti.math.cos(seta))-axisnormal[1]*ti.math.sin(seta)
       axs_rot[2,1]=axisnormal[1]*axisnormal[2]*(1.0-ti.math.cos(seta))+axisnormal[0]*ti.math.sin(seta)
       axs_rot[2,2]=axisnormal[2]*axisnormal[2]*(1.0-ti.math.cos(seta))+ti.math.cos(seta)
       #print(axs_rot)
       return axs_rot
    @ti.func 
    def toEuler(self,RotMat:ti.types.matrix(3,3,ti.f32))->ti.types.vector(3,dtype=ti.float32):
      #print("RotMat",RotMat)
      sy=ti.math.sqrt(RotMat[0,0]*RotMat[0,0]+RotMat[1,0]*RotMat[1,0])
      Rotator=ti.Vector([0.0,0.0,0.0])
      if(sy<1e-6):
         Rotator[0]=0.0
         Rotator[1]=ti.math.atan2(-RotMat[2,0],sy)
         Rotator[2]=ti.math.atan2(-RotMat[1,2],RotMat[2,2])
      else:
         Rotator[0]=ti.math.atan2(RotMat[2,1],RotMat[2,2])
         Rotator[1]=ti.math.atan2(-RotMat[2,0],sy)
         Rotator[2]=ti.math.atan2(RotMat[1,0],RotMat[0,0])
      Rotator[0]= Rotator[0]*180.0/pi#roll
      Rotator[1]= Rotator[1]*180.0/pi#pitch
      Rotator[2]= Rotator[2]*180.0/pi#yaw
      return Rotator 
    @ti.func
    def Jcob(self,TargetLocation:ti.math.vec3):
       n=0
       axs_rot=ti.Matrix([[0]*9 for _ in range(3)],ti.float32)
       prevaxs_rot=ti.Matrix([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
       result_rot=ti.Matrix([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
       prevaxs_rot[0]=self.rotator[0][2]
       prevaxs_rot[1]=self.rotator[0][1]
       prevaxs_rot[2]=self.rotator[0][0]
       prevaxs_rot[3]=self.rotator[1][2]
       prevaxs_rot[4]=self.rotator[1][1]
       prevaxs_rot[5]=self.rotator[1][0]
       prevaxs_rot[6]=self.rotator[2][2]
       prevaxs_rot[7]=self.rotator[2][1]
       prevaxs_rot[8]=self.rotator[2][0]
       deltar=TargetLocation-self.WorldLocation[3]
       while(n<3):
            deltarr=TargetLocation-self.WorldLocation[n]
            xrad=self.rotator[n][0]/180.0*pi
            self.XMat[n][0,0]=1.0
            self.XMat[n][1,1]=ti.cos(xrad)
            self.XMat[n][1,2]=-ti.sin(xrad)
            self.XMat[n][2,1]=ti.sin(xrad)
            self.XMat[n][2,2]=ti.cos(xrad)
            yrad=self.rotator[n][1]/180.0*pi
            self.YMat[n][1,1]=1.0
            self.YMat[n][0,0]=ti.cos(yrad)
            self.YMat[n][0,2]=ti.sin(yrad)
            self.YMat[n][2,0]=-ti.sin(yrad)
            self.YMat[n][2,2]=ti.cos(yrad)
            zrad=self.rotator[n][2]/180.0*pi
            self.ZMat[n][2,2]=1.0
            self.ZMat[n][0,0]=ti.cos(zrad)
            self.ZMat[n][0,1]=-ti.sin(zrad)
            self.ZMat[n][1,0]=ti.sin(zrad)
            self.ZMat[n][1,1]=ti.cos(zrad)
            setaz=ti.Vector([0.0,0.0,0.0])
            setay=ti.Vector([0.0,0.0,0.0])
            setax=ti.Vector([0.0,0.0,0.0])
            if(n==0):
               axisz=ti.Vector([0.0,0.0,1.0])
               setaz=ti.math.cross(deltarr,axisz)
               axisy=ti.Vector([0.0,1.0,0.0])
               setay=ti.math.cross(deltarr,axisy)
               axisx=ti.Vector([1.0,0.0,0.0])
               setax=ti.math.cross(deltarr,axisx)
            else:
               axisz=self.Q[n]@ti.Vector([0.0,0.0,1.0])
               #print(self.Q[n])
               setaz=ti.math.cross(deltarr,axisz)
               axisy=self.Q[n]@self.ZMat[n]@ti.Vector([0.0,1.0,0.0])
               setay=ti.math.cross(deltarr,axisy)
               axisx=self.Q[n]@self.ZMat[n]@self.YMat[n]@ti.Vector([1.0,0.0,0.0])
               setax=ti.math.cross(deltarr,axisx)
            #print('setaz=',setaz,'setay=',setay,'setax=',setax)
            axs_rot[0,3*n]=setaz[0]
            axs_rot[1,3*n]=setaz[1]
            axs_rot[2,3*n]=setaz[2]
            axs_rot[0,3*n+1]=setay[0]
            axs_rot[1,3*n+1]=setay[1]
            axs_rot[2,3*n+1]=setay[2]
            axs_rot[0,3*n+2]=setax[0]
            axs_rot[1,3*n+2]=setax[1]
            axs_rot[2,3*n+2]=setax[2]
            n+=1
      #  for i in range(3):
      #     for j in range(9):
      #        print('i=',i,'j=',j,axs_rot[i,j])
       result_rot=prevaxs_rot-0.2*deltar@axs_rot
       #print(deltar)
       print(result_rot)
       self.rotator[0][2]=result_rot[0]
       self.rotator[0][1]=result_rot[1]
       self.rotator[0][0]=result_rot[2]
       self.rotator[1][2]=result_rot[3]
       self.rotator[1][1]=result_rot[4]
       self.rotator[1][0]=result_rot[5]
       self.rotator[2][2]=result_rot[6]
       self.rotator[2][1]=result_rot[7]
       self.rotator[2][0]=result_rot[8]
       self.Fk(0)
    @ti.kernel
    def JcobIk(self,TargetLocation:ti.math.vec3):
       Xlength=(TargetLocation-self.WorldLocation[3]).norm()
       n=200
       while(n>0):
          self.Jcob(TargetLocation)
          Xlength=(TargetLocation-self.WorldLocation[3]).norm()
          print(Xlength)
          n-=1
      
                 
indices[0]=0;
indices[1]=1;
indices[2]=1;
indices[3]=2;
indices[4]=2;
indices[5]=3;
colors[0]=(1.0,0.0,0.0)
colors[1]=(0.0,1.0,0.0)
colors[2]=(0.0,0.0,1.0)
colors[3]=(1.0,1.0,0.0)
      

window=ti.ui.Window("demo",res=(1280,720));
gui=window.get_gui();
canvas=window.get_canvas();
canvas.set_background_color((1.0,1.0,1.0));
scence=ti.ui.Scene();
camera=ti.ui.Camera();
XlSArray=XlsBoneArray();

XlSArray.init()
XlSArray.UpdateLocation()
Iklocation=XlSArray.GetLocation(3)
TargetLocation=TempLocation+Iklocation
XlSArray.JcobIk(TargetLocation)
for i in range(4):
    vertics[i]=XlSArray.GetLocation(i)
    #print(vertics[i])
# while window.running:
#       camera.position(0,0,40)
#       camera.lookat(0,0,0)
#       scence.set_camera(camera)
#       camera.up(0, 1, 0)
#       scence.ambient_light((1, 1, 1))
#       # for i in range(4):
#       #   XlSArray.UpdateRotation(i,TempRotation[i])
#       window.get_event()
#       if window.is_pressed(ti.ui.RIGHT,'r'):
#           TargetLocation=TempLocation+Iklocation
#           #print(TargetLocation)
#       if window.is_pressed(ti.ui.RIGHT,'s'):   
#           XlSArray.UpdateIk(TargetLocation)
#           print(XlSArray.GetLocation(3))
#           XlSArray.UpdateLocation()
#           print(XlSArray.GetLocation(3))
#           #print("ik=",XlSArray.GetLocation(3))
#       #XlSArray.UpdateLocation()
#       #print("fk=",XlSArray.GetLocation(3))
#       for i in range(4):
#         vertics[i]=XlSArray.GetLocation(i)
      
#       scence.lines(vertics,4.0,indices,per_vertex_color=colors)
#       canvas.scene(scence)  
#       with gui.sub_window("SubWindow", x=0, y=0, width=0.3, height=0.3) as g: 
#       #  TempRotation[0][2]=g.slider_float("TempRotation[0][2]", TempRotation[0][2], minimum=-180.0, maximum=180.0)
#       #  TempRotation[1][2]=g.slider_float("TempRotation[1][2]", TempRotation[1][2], minimum=-180.0, maximum=180.0)
#       #  TempRotation[2][2]=g.slider_float("TempRotation[2][2]", TempRotation[2][2], minimum=-180.0, maximum=180.0)
#        TempLocation[0]=g.slider_float("TempLocation.x", TempLocation[0], minimum=-10.0, maximum=10.0)
#        TempLocation[1]=g.slider_float("TempLocation.y", TempLocation[1], minimum=-10.0, maximum=10.0)
#       window.show()