import numpy as np

def rx(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def rz(theta):
    return np.array([[np.cos(theta / 2) - 1j * np.sin(theta / 2), 0],
                     [0, np.cos(theta / 2) + 1j * np.sin(theta / 2)]])

def theta_phi(theta, phi):
    return rz(phi) @ rx(-theta) @ rz(-phi)

result1 = theta_phi(np.pi, np.pi/6) @ theta_phi(np.pi, 0) @ theta_phi(np.pi, np.pi/2)\
         @ theta_phi(np.pi, 0) @ theta_phi(np.pi, np.pi/6)

result2 = theta_phi(np.pi, 2 * np.pi/3) @ theta_phi(np.pi, np.pi/2) @ theta_phi(np.pi, np.pi)\
         @ theta_phi(np.pi, np.pi/2) @ theta_phi(np.pi, 2 * np.pi/3)

print(result1)

print(result2)

print(result1 @ result2 @ result1 @ result2)
