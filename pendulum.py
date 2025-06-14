import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import place

guiFlag = False

dt = 1/240  # pybullet simulation step
th0 = 0     # starting position (radian)
thd = np.pi # desired position (radian)
g = 9.81    # m/s^2
L = 0.8     # m
m = 1       # kg


A = np.array([[0, 1],
              [g / L, 0]])
B = np.array([[0],
              [1 / (m * L ** 2)]])
poles = np.array([-9, -20])
K = -place(A, B, poles)
print(np.linalg.eigvals(A - B @ K))
physicsClient = p.connect(p.GUI if guiFlag else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")
pendulumId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)

p.changeDynamics(pendulumId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(pendulumId, 2, linearDamping=0, angularDamping=0)

# Set starting position and speed
p.resetJointState(pendulumId, 1, th0, 0)

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=pendulumId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5 # seconds
logTime = np.arange(0, maxTime, dt)
size = len(logTime)
logThetaSim = np.zeros(size)
logVelSim = np.zeros(size)
logTauSim = np.zeros(size)
idx = 0

for t in logTime:
    th = p.getJointState(pendulumId, 1)[0]
    vel = p.getJointState(pendulumId, 1)[1]
    logThetaSim[idx] = th

    tau = K[0,0] * (th - thd) + K[0,1] * vel
    logTauSim[idx] = tau

    p.setJointMotorControl2(bodyIndex=pendulumId, jointIndex=1, force=tau, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()

    vel = p.getJointState(pendulumId, 1)[1]
    logVelSim[idx] = vel

    idx += 1
    if guiFlag:
        time.sleep(dt)

p.disconnect()

plt.subplot(3,1,1)
plt.plot(logTime, logThetaSim, 'b', label="Sim Pos")
plt.plot([logTime[0], logTime[-1]], [thd, thd], 'r--', label="Ref Pos")
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(logTime, logVelSim, 'b', label="Sim Vel")
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.plot(logTime, logTauSim, 'b', label="Sim Tau")
plt.grid(True)
plt.legend()
plt.show()
