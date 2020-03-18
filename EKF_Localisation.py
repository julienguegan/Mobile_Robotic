import math
from math import sin,cos,atan2
from math import pi,ceil
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand,randn
from numpy import diag
import random
from numpy.linalg import norm,inv,eig

#%-------- Drawing Covariance -----%
def PlotEllipse(x,P,nSigma):
    eH = []
    P = P[0:2,0:2] #% only plot x-y part
    x = x[0:2]
    plt.plot(XStore[0, :], XStore[1, :], ".k")
    if (not np.all(diag(P))):
        D,V = eig(P)
#        y = nSigma*[cos(0:0.1:2*pi);sin(0:0.1:2*pi)];
#        el = V*sqrtm(D)*y;
#        el = [el el(:,1)]+repmat(x,1,size(el,2)+1);
#        eH = line(el(1,:),el(2,:));
    return eH
    
def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot   = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0]).flatten()
    py = np.array(fx[1, :] + xEst[1]).flatten()
    plt.plot(px, py, "--r")

def DoVehicleGraphics(x,P,nSigma,Forwards):
    #plt.cla()
    ShiftTheta = atan2(Forwards[1],Forwards[0])
    # h = PlotEllipse(x,P,nSigma)
    plot_covariance_ellipse(x,P)
    #    set(h,'color','r');
    DrawRobot(x,'b',ShiftTheta);
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.001)


#%-------- Drawing Vehicle -----%
def DrawRobot(Xr,col,ShiftTheta):
    p  = 0.02 # % percentage of axes size
    a  = plt.axis()
    l1 = (a[1] - a[0]) * p
    l2 = (a[3]-a[2])*p;
    P  = np.array([[-1,1,0,-1],[-1,-1,3,-1]]) # basic triangle
    theta = Xr[2] - pi/2 + ShiftTheta; # rotate to point along x axis (theta = 0)
    c = cos(theta);
    s = sin(theta);
    P = np.array([[c,-s],[s,c]]) @ P; # rotate by theta
    P[0,:] = P[0,:] * l1 + Xr[0]; # scale and shift to x
    P[1,:] = P[1,:] * l2 + Xr[1];
    plt.plot(P[0,:], P[1,:], col, LineWidth=1, alpha=0.2);# draw
    plt.plot(Xr[0], Xr[1],'+')


#
def DoGraphs(InnovStore,PStore,SStore,XStore,XErrStore):
    
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
    ax1.plot(InnovStore[0,:],LineWidth=1,label='distance')
    ax1.plot(SStore[0,:],'r')
    ax1.plot(-SStore[0,:],'r')
    ax1.set_xlabel('t')
    ax1.legend(loc="upper right")
    ax2.plot(InnovStore[1,:] * 180/pi,LineWidth=1,label='angle (deg)')
    ax2.plot(SStore[1,:],'r')
    ax2.plot(-SStore[1,:],'r')
    ax2.set_xlabel('t')
    ax2.legend(loc="upper right")
    fig1.show() 
    
    fig1.suptitle('innovation = $y_{mesure} - y_{predit}$')
    
    input("Press Enter to continue...")

#
#    figure(1) print -depsc 'EKFLocation.eps'
#
#    figure(2)
#    subplot(2,1,1)plot(InnovStore(1,:))hold onplot(SStore(1,:),'r')plot(-SStore(1,:),'r')
#    title('Innovation')ylabel('range')
#    subplot(2,1,2)plot(InnovStore(2,:)*180/pi)hold onplot(SStore(2,:)*180/pi,'r')plot(-SStore(2,:)*180/pi,'r')
#    ylabel('Bearing (deg)')xlabel('time')
#    print -depsc 'EKFLocationInnov.eps'
#
#    figure(2)
#    subplot(3,1,1)plot(XErrStore(1,:))hold onplot(3*PStore(1,:),'r')plot(-3*PStore(1,:),'r')
#    title('Covariance and Error')ylabel('x')
#    subplot(3,1,2)plot(XErrStore(2,:))hold onplot(3*PStore(2,:),'r')plot(-3*PStore(2,:),'r')
#    ylabel('y')
#    subplot(3,1,3)plot(XErrStore(3,:)*180/pi)hold onplot(3*PStore(3,:)*180/pi,'r')plot(-3*PStore(3,:)*180/pi,'r')
#    ylabel('\theta')xlabel('time')
#    print -depsc 'EKFLocationErr.eps'

#
def GetObservation(k):
    global Mapglobal, xTrueglobal, PYTrueglobal, nSteps
    if (k>2500 and k<3500):
        z = None
        iFeature = -1
    else:
        iFeature = random.randint(0,Map.shape[1]-1)
        z = DoObservationModel(xTrue, iFeature,Map)+np.sqrt(PYTrue)@randn(2)
        z[1] = AngleWrap(z[1])
    return [z,iFeature]

def DoObservationModel(xVeh, iFeature,Map):
    Delta = Map[0:2,iFeature-1] - xVeh[0:2]
    z     = np.array([norm(Delta), atan2(Delta[1],Delta[0]) - xVeh[2]])
    z[1]  = AngleWrap(z[1])
    return z

def AngleWrap(a):
    if (a>np.pi):
        a=a-2*pi
    elif (a<-np.pi):
        a = a+2*pi;
    return a

#
def SimulateWorld(k):
    global xTrue
    u = GetRobotControl(k)
    xTrue = tcomp(xTrue,u)
    xTrue[2] = AngleWrap(xTrue[2])

#
def GetOdometry(k):
    global LastOdom # internal to robot low-level controller
    global QTrue
    global xTrue
    if(LastOdom is None):
        LastOdom = xTrue
    u = GetRobotControl(k)
    xnow = tcomp(LastOdom,u)
    uNoise = np.sqrt(QTrue) @ randn(3)
    xnow = tcomp(xnow,uNoise)
    LastOdom = xnow
    return xnow

# construct a series of odometry measurements
def GetRobotControl(k):
    global nSteps
    u = np.array([0, 0.025,  0.1*np.pi/180*math.sin(3*np.pi*k/nSteps)])
    #u = [0 0.15  0.3*pi/180]
    assert u.ndim == 1
    return u

# h(x) Jacobian
def GetObsJac(xPred, iFeature, Map):
    delta = Map[0:2, iFeature] - xPred[0:2]
    r     = norm(delta)
    Jac_H = np.zeros((2,3))
    Jac_H[0,0] = -delta[0] / r
    Jac_H[0,1] = -delta[1] / r
    Jac_H[1,0] =  delta[1] / (r**2)
    Jac_H[1,1] = -delta[0] / (r**2)
    Jac_H[1,2] = -1
    return Jac_H

# f(x,u) Jacobian wrt x
def A(x,u):
    s1 = sin(x[2])
    c1 = cos(x[2])
    Jac_A  = np.array([[1, 0, -u[0] * s1 - u[1] * c1],
                       [0, 1,  u[0] * c1 - u[1] * s1],
                       [0, 0,  1]])  
    return Jac_A
    
# f(x,u) Jacobian wrt u
def B(x,u):
    s1 = sin(x[2])
    c1 = cos(x[2])
    Jac_B  = np.array([[c1, -s1, 0],
                       [s1,  c1, 0],
                       [0,    0, 1]])
    return Jac_B

def tinv(tab):
    assert tab.ndim == 1
    tba = 0.0*tab;
    for t in range(0,tab.shape[0],3):
        tba[t:t+3] = tinv1(tab[t:t+3])
    assert tba.ndim == 1
    return tba

def tinv1(tab):
    assert tab.ndim == 1
    # calculates the inverse of one transformations
    s = math.sin(tab[2])
    c = math.cos(tab[2])
    tba = np.array([-tab[0]*c - tab[1]*s,
            tab[0]*s - tab[1]*c,
           -tab[2]])
    assert tba.ndim == 1
    return tba

# composes two transformations
def tcomp(tab,tbc):
    assert tab.ndim == 1
    assert tbc.ndim == 1
    result = tab[2]+tbc[2]
    result = AngleWrap(result)
 
    s = sin(tab[2])
    c = cos(tab[2])
    # print(np.array([[c, -s],[s, c]]) , tbc, tbc[0:2])
    tac = tab[0:2]+ np.array([[c, -s],[s, c]]) @ tbc[0:2]
    tac = np.append(tac,result)
    #print("tcomp:",tab,tbc,tac)
    return tac

LastOdom = None

nSteps = 10000
# Location of beacons
Map = 140 * rand(2,1) - 70 # Map = 140*rand(2,30)-70

# True covariance of errors used for simulating robot movements
QTrue  = np.diag([0.01,0.01,1*math.pi/180]) ** 2
PYTrue = diag([2.0,3*math.pi/180]) ** 2

# Modeled errors used in the Kalman filter process
QEst  = np.eye(3,3) @ QTrue  + QTrue * (8/10)
PYEst = np.eye(2,2) @ PYTrue - PYTrue * (8/10)

xTrue     = np.array([1,-40,-math.pi/2])
xOdomLast = GetOdometry(0)

#initial conditions:
xEst = xTrue
xEst = np.array([0,0,0])
PEst = np.diag([1,1,1*(math.pi/180)**2])

#  storage  #
InnovStore = np.nan * np.zeros((2,nSteps))
SStore     = np.NaN * np.zeros((2,nSteps))
PStore     = np.NaN * np.zeros((3,nSteps))
XStore     = np.NaN * np.zeros((3,nSteps))
XErrStore  = np.NaN * np.zeros((3,nSteps))

#initial graphics
plt.cla()
plt.plot(Map[0,:],Map[1,:],'*g', markersize=15)
plt.text(Map[0,:] + 2,Map[1,:] + 2,'amer', color='green', fontsize=15)
#hObsLine = line([0,0],[0,0])
#set(hObsLine,'linestyle',':')

for k in range(1,nSteps):
    
    #do world iteration
    SimulateWorld(k)
    
    #figure out control by subtracting current and previous odom values
    xOdomNow = GetOdometry(k)
    #print(xOdomNow,xOdomLast)
    u = tcomp(tinv(xOdomLast),xOdomNow)
    xOdomLast = xOdomNow
    
    #do prediction
    xPred    = tcomp(xEst,u) # function f
    xPred[2] = AngleWrap(xPred[2])
    PPred    = A(xEst,u) @ PEst @ A(xEst,u).T + B(xEst,u) @ QEst @ B(xEst,u).T
        
    #observe a randomn feature
    [y_mesure, iFeature] = GetObservation(k)
        
    if y_mesure is not None:
        #predict observation
        y_predit = DoObservationModel(xPred,iFeature,Map)
        
        # get observation Jacobian
        H = GetObsJac(xPred,iFeature,Map)
      
        #do Kalman update:
        Innov    = y_mesure - y_predit
        Innov[1] = AngleWrap(Innov[1])
        
        S = H @ PPred @ H.T + PYEst
        K = PPred @ H.T @ inv(S)
        
        xEst = xPred + K @ Innov
        
        xEst[2] = AngleWrap(xEst[2])

        PEst = PPred - K @ H @ PPred
        PEst = 0.5 * (PEst + PEst.T) # ensure symetry
        
    else:
        #There was no observation available
        xEst  = xPred
        PEst  = PPred
        Innov = np.array([np.NaN,np.NaN])
        S     = np.NaN * np.eye(2)
            
    # plot every 200 updates
    if (k-2) % 75 == 0:
        DoVehicleGraphics(xEst,PEst[0:2,0:2],8,[0,1])
        if y_mesure is not None:
#            set(hObsLine,'XData',[xEst[0],Map[0,iFeature]])
#            set(hObsLine,'YData',[xEst[1],Map[1,iFeature]])
            pass
#        drawnow
    
    #store results:
    InnovStore[:,k] = Innov
    PStore[:,k]     = np.sqrt(diag(PEst))
    SStore[:,k]     = np.sqrt(diag(S))
    XStore[:,k]     = xEst
    XErrStore[:,k]  = xTrue-xEst
    
    
DoGraphs(InnovStore,PStore,SStore,XStore,XErrStore)


