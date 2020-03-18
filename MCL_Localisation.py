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
    rot = np.array([[math.cos(angle), math.sin(angle)],
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

def GetObservation(k):
    global Mapglobal, xTrueglobal, PYTrueglobal, nSteps
    if (k>600 and k<900):
        z = None
        iFeature=-1
    else:
        iFeature = random.randint(0,Map.shape[1]-1)
        z = DoObservationModel(xTrue, iFeature,Map)+np.sqrt(PYTrue)@randn(2)
        z[1] = AngleWrap(z[1])
    return [z,iFeature]

def DoObservationModel(xVeh, iFeature,Map):
    Delta = Map[0:2,iFeature-1]-xVeh[0:2]
    z = np.array([norm(Delta),
        atan2(Delta[1],Delta[0])-xVeh[2]])
    z[1] = AngleWrap(z[1])
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
    global LastOdom #internal to robot low-level controller
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

# Functions to be completed

# h(x) Jacobian
def GetObsJac(xPred, iFeature,Map):
    jH = np.zeros((2,3))
    Delta = (Map[0:2,iFeature]-xPred[0:2])
    r = norm(Delta)
    jH[0,0] = -Delta[0] / r
    jH[0,1] = -Delta[1] / r
    jH[1,0] = Delta[1] / (r**2)
    jH[1,1] = -Delta[0] / (r**2)
    jH[1,2] = -1
    return jH

# f(x,u) Jacobian # x
def A(x,u):
    s1 = math.sin(x[2])
    c1 = math.cos(x[2])

    Jac  = np.array([[1, 0, -u[0]*s1-u[1]*c1],
            [0, 1, u[0]*c1-u[1]*s1],
            [0, 0, 1]])
            
    return Jac
    
# f(x,u) Jacobian # u
def B(x,u):
    s1 = sin(x[2])
    c1 = cos(x[2])

    Jac  = np.array([[c1, -s1, 0],
            [s1, c1, 0],
            [0, 0, 1]])

    return Jac

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
    
def tcomp(tab,tbc):
# composes two transformations
    assert tab.ndim == 1
    assert tbc.ndim == 1
    result = tab[2]+tbc[2]

    #result = AngleWrap(result)
 
    s = sin(tab[2])
    c = cos(tab[2])
    # print(np.array([[c, -s],[s, c]]) , tbc, tbc[0:2])
    tac = tab[0:2]+ np.array([[c, -s],[s, c]]) @ tbc[0:2]
    tac = np.append(tac,result)
    #print("tcomp:",tab,tbc,tac)
    return tac

LastOdom = None

# change this to see how sensitive we are to the number of particle
# (hypotheses run) especially in relation to initial distribution!
nParticles = 400;

nSteps = 1000
# Location of beacons
#Map = 140*rand(2,30)-70
Map = 140*rand(2,10)-70
#Map = 140*rand(2,1)-70

# True covariance of errors used for simulating robot movements
QTrue  = np.diag([0.01,0.01,1*math.pi/180]) ** 2
PYTrue = diag([2,3*math.pi/180]) ** 2

# Modeled errors used in the Kalman filter process
QEst  = 1 * np.eye(3,3) @ QTrue
PYEst = 1 * np.eye(2,2) @ PYTrue

xTrue = np.array([1,-40,-math.pi/2])
xOdomLast = GetOdometry(0)

# initial conditions:
# xEst = xTrue
# xEst = np.array([0,0,0])
PEst = 10*np.diag([1,1,(1*math.pi/180)**2])

# initial conditions: - a point cloud around truth
xPred = xTrue[:,np.newaxis] + diag([8,8,0.4]) @ randn(3,nParticles);

#  storage  
InnovStore = np.nan*np.zeros((2,nSteps))
SStore     = np.NaN*np.zeros((2,nSteps))
PStore     = np.NaN*np.zeros((3,nSteps))
XStore     = np.NaN*np.zeros((3,nSteps))
XErrStore  = np.NaN*np.zeros((3,nSteps))

# initial graphics
plt.cla()
plt.plot(Map[0,:],Map[1,:],'.g')

for k in range(1,nSteps):
    
    # do world iteration
    SimulateWorld(k)
    
    # all particles are equally important
    L = np.ones((nParticles))/nParticles
    
    # figure out control by subtracting current and previous odom values
    xOdomNow = GetOdometry(k)
    u = tcomp(tinv(xOdomLast),xOdomNow)
    xOdomLast = xOdomNow
    
    # do prediction
    # for each particle we add in control vector AND noise
    # the control noise adds diversity within the generation
    for p in range(nParticles):
        xPred[:,p] = tcomp(xPred[:,p].squeeze(),u+np.squeeze(np.sqrt(QEst) @ randn(3,1)))
                
    #observe a randomn feature
    [z,iFeature] = GetObservation(k)
        
    if z is not None:
        #predict observation
        for p in range(nParticles):
            zPred = DoObservationModel(xPred[:,p],iFeature,Map)
        
            #how different
            Innov = z-zPred
            #get likelihood (new importance). Assume gaussian here but any pdf works!
            #if predicted obs is very different from actual obs this score will be low
            #->this particle is not very good at representing state. A lower score means
            #it is less likely to be selected for the next generation...
            L[p] = np.exp(-0.5*Innov[np.newaxis,:] @ np.linalg.inv(PYEst) @ Innov[:,np.newaxis]) + 0.001
        #print("Weights: ",L)

    # Compute position as weighted mean of particles
    xEst = np.average(xPred,axis=1,weights=L)
    # squaredError = (xP-xEst).*(xP-xEst);
    # xVariance= [xVariance sqrt(mean(squaredError(1,:)+squaredError(2,:)))];


    # reselect based on weights:
    # particles with big weights will occupy a greater percentage of the
    # y axis in a cummulative plot
    CDF = np.cumsum(L)/np.sum(L)
    # so randomly (uniform) choosing y values is more likely to correspond to
    # more likely (better) particles...
    iSelect  = rand(nParticles)
    # find the particle that corresponds to each y value (just a look up)
    iNextGeneration = np.interp(iSelect,CDF,range(nParticles),left=0).astype(int).ravel()
    # print(iNextGeneration,iSelect,CDF,range(nParticles))
    # copy selected particles for next generation...
    xPred = xPred[:,iNextGeneration]
    L = L[iNextGeneration]
            
    # plot every 200 updates
    if (k % 75 == 0):
        print("max weight: ",max(L))
        DoVehicleGraphics(xEst,PEst[0:2,0:2],8,[0,1])
        plt.plot(xPred[0],xPred[1],'.b')
        if z is not None:
            plt.plot(Map[0,iFeature],Map[1,iFeature],'.g')
            pass
    
    # store results:
    InnovStore[:,k] = Innov
    XStore[:,k] = xEst
    XErrStore[:,k] = xTrue-xEst

DoGraphs(InnovStore,PStore,SStore,XStore,XErrStore)

# ===== RESULTS ========== #
'''
n_amer :       |     1       |     10        |     10         |      10      |     10        |     10
n_particle     |    600      |    400        |     400        |     400      |     400       |     400
init_particle  | [8,8,0.4]   | [8,8,0.4]     |  [80,80,30]    |   [8,8,0.4]  |   [8,8,0.4]   |   [8,8,0.4]
QTrue          |[0.01,0.01,1]| [0.01,0.01,1] |  [0.01,0.01,1] |  [0.5,0.5,5] |  [0.5,0.5,5]  | [0.01,0.01,1]
PYTrue         |  [2.0, 3]   |  [2.0, 3]     |   [2.0, 3]     |    [2.0, 3]  |   [6.0, 5]    |   [2.0, 3]
QEst           |     1       |     1         |     1          |     1        |      1        |     5
PYEst          |     1       |     1         |     1          |     1        |      1        |     5
______________________________________________________________________________________________________________

var error dist |    ~10      |    ~20        |    ~150        |      ~30     |    ~30        |     ~20
var error angl |   ~300      |    ~300       |    ~3000       |     ~300     |    ~300       |      ~300

n_particle = 600 : ~10 avec parfois des sauts de 300
n_particle = 200 :  idem

+ n_particles => 
+ bruit init_particle => + error






'''