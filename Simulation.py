"""
This script runs FDTD simulation for set model and parameters
"""
import meep as mp
import numpy as np
from scipy.linalg import hadamard
import sys
import datetime
import matplotlib.pyplot as plt
import os

def create_sources(freq,df, NA, R, resolution, n, hadamar_column, x0, y0, z0, psy):
    fcen = freq;
    # df = 0.1
    # NA = 0.75 #NA setting
    # R = 5   #radius setting
    # n = 4   #hadamard matrix order
    # x0=0; y0=0; z0=0  #focusing point
    # resolution = 5
    sources = []
    theta0 = np.arcsin(NA);
    theta = theta0 / 2;
    xs = [];
    ys = [];
    zs = [];
    step_phi = 1 / resolution / R
    d = int(theta0 / step_phi) + 1

    # Hadamard Generation
    # psy = np.pi
    tau = 1 / fcen * psy / 2 / np.pi
    n1 = np.round(theta0 / step_phi) + 1
    j = np.round((np.log2(n1 * n1) + 1) / 2)
    n1 = np.power(2, n)
    HD = hadamard(n1)
    r = int(d / n1) + 1
    t = 0
    q = 0
    t = 0
    i = 0
    # Count of Number of sources and Hadamard Column Projection
    while theta >= -theta0 / 2:
        phi = theta0 / 2
        while phi >= -theta0 / 2:
            if (theta * theta + phi * phi <= theta0 * theta0 / 4):
                q = q + 1
            phi = phi - step_phi
        theta = theta - step_phi

    HD1 = [row[hadamar_column] for row in HD]
    n = len(HD1)
    r = int((q - 1) / n) + 1
    HD2 = np.zeros(q)
    i = 0
    while i < q:
        t = int(i / r)
        HD2[i] = HD1[t]
        i = i + 1

    i = 0
    theta0 = np.arcsin(NA);
    theta = theta0 / 2;
    xs = [];
    ys = [];
    zs = [];
    step_phi = 1 / resolution / np.abs(R)

    while theta >= -theta0 / 2:
        phi = theta0 / 2
        while phi >= -theta0 / 2:
            x = (R * np.cos(theta) * np.sin(phi)) - x0
            y = (R * np.sin(theta)) - y0
            z = (R * np.cos(theta) * np.cos(phi)) - z0
            if (theta * theta + phi * phi <= theta0 * theta0 / 4):
                xs.append(R * np.cos(theta) * np.sin(phi) - x0)
                ys.append(R * np.sin(theta) - y0)
                zs.append(R * np.cos(theta) * np.cos(phi) - z0)

                if HD2[i] == 1:
                    sources.append(mp.Source(
                        mp.GaussianSource(fcen, fwidth=df, start_time=tau, cutoff=1),
                        component=mp.Ex,
                        center=mp.Vector3(x, y, z),
                        size=mp.Vector3(0, 0, 0),
                    ))

                if HD2[i] == -1:
                    sources.append(mp.Source(
                        mp.GaussianSource(fcen, fwidth=df, start_time=0, cutoff=1),
                        component=mp.Ex,
                        center=mp.Vector3(x, y, z),
                        size=mp.Vector3(0, 0, 0),
                    ))

                i = i + 1
            phi = phi - step_phi
        theta = theta - step_phi
    return sources
now = datetime.datetime.now()
output_directory ="output/SimulationOut-" + str(now.date())+"_"+str(now.hour)+"-"+str(now.minute)
#if not os.path.exists(output_directory):
#    os.mkdir(output_directory)


resolution = 20
freq = 1.25  # 800 nm
#freq = 0.956937799043 # 1045 nm
# freq = 1.24223602484  #  805 nm

model_pml = 2
sizeX = float(20); sizeY = float(20); sizeZ = float(55); thicknessPML = 1  # structure size
addFlux = 1 / resolution * 2
cell = mp.Vector3(sizeX+2*model_pml, sizeY+2*model_pml, sizeZ+2*model_pml)


N = 5000 #number of particles per plane
coord = np.zeros((3, N))
Z = 0;  dZ = 0.10;  radius = 0.1;  geometry = []

fcen = freq; df = 0.1
# sources = [mp.Source(mp.ContinuousSource(frequency=freq),
#                      component=mp.Ex,
#                      center=mp.Vector3(),
#                      size=mp.Vector3(), amplitude=1)]
sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ex,
                     center=mp.Vector3(0, 0, -(sizeZ/2 + 6*addFlux)), #3D case
                   # center=mp.Vector3(-sizeX/2, 0, 0),
                     size=mp.Vector3(sizeX, sizeY, 0), amplitude=1)]
#sources = create_sources(freq, df, 0.7, 5, resolution, 4, 0, 0, 0, 24, 0)
#sys.stderr.write(np.str(mp.Vector3(z=-0.5*sizeZ-addFlux/2)))
#sys.stderr.write(np.str(mp.Vector3(sizeX+addFlux/2, sizeY+addFlux/2, 0)))
geometry = []
while Z < sizeZ/2-1:
    for i in range(N):
        coord[0, i] = np.random.rand()*sizeX-sizeX/2
        coord[1, i] = np.random.rand()*sizeY-sizeY/2
        coord[2, i] = Z
        bufferShape = mp.Sphere(center=mp.Vector3(coord[0, i], coord[1, i], coord[2, i]),
                              radius=radius,
                              material=mp.Medium(epsilon=1.44))
        geometry.append(bufferShape)
    Z = Z + dZ



# for i in range(N):
#         coord[0, i] = np.random.rand()*sizeX-sizeX/2
#         coord[1, i] = np.random.rand()*sizeY-sizeY/2
#         coord[2, i] = Z
#         bufferShape = mp.Sphere(center=mp.Vector3(coord[0, i], coord[1, i], coord[2, i]),
#                               radius=radius,
#                               material=mp.Medium(epsilon=12))
#         geometry.append(bufferShape)
pml_layers = [mp.PML(thicknessPML)]
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    # geometry=geometry,
                    epsilon_input_file='Input_files/input-eps-2019-04-03_11-46_20x20x50um_PML2um_GAP5um_.hdf5',
                    # epsilon_input_file='output/2019-01-08_15-42.hdf5',
                    sources=sources,
                    resolution=resolution)
# nearfield_box = sim.add_near2far(freq, 0, 1,
#                                  mp.Near2FarRegion(mp.Vector3(z=-0.5*sizeZ), size=mp.Vector3(sizeX, sizeY, 0), weight=-1),
#                                  mp.Near2FarRegion(mp.Vector3(z=0.5*sizeZ), size=mp.Vector3(sizeX, sizeY, 0), weight=+1),
#                                  mp.Near2FarRegion(mp.Vector3(x=-0.5*sizeX), size=mp.Vector3(0, sizeY, sizeZ), weight=-1),
#                                  mp.Near2FarRegion(mp.Vector3(x=+0.5*sizeX), size=mp.Vector3(0, sizeY, sizeZ), weight=+1),
#                                  mp.Near2FarRegion(mp.Vector3(y=-0.5*sizeY), size=mp.Vector3(sizeX, 0, sizeZ), weight=-1),
#                                  mp.Near2FarRegion(mp.Vector3(y=+0.5*sizeY), size=mp.Vector3(sizeX, 0, sizeZ), weight=+1))



nearfield_box = sim.add_near2far(fcen, df, 20,
                                 mp.Near2FarRegion(mp.Vector3(z=-0.5*sizeZ-addFlux), size=mp.Vector3(sizeX+2*addFlux, sizeY+2*addFlux, 0), weight=-1),
                                 mp.Near2FarRegion(mp.Vector3(z=0.5*sizeZ+addFlux), size=mp.Vector3(sizeX+2*addFlux, sizeY+2*addFlux, 0), weight=+1),
                                 mp.Near2FarRegion(mp.Vector3(x=-0.5*sizeX-addFlux), size=mp.Vector3(0, sizeY+2*addFlux, sizeZ+2*addFlux), weight=-1),
                                 mp.Near2FarRegion(mp.Vector3(x=+0.5*sizeX+addFlux), size=mp.Vector3(0, sizeY+2*addFlux, sizeZ+2*addFlux), weight=+1),
                                 mp.Near2FarRegion(mp.Vector3(y=-0.5*sizeY-addFlux), size=mp.Vector3(sizeX+2*addFlux, 0, sizeZ+2*addFlux), weight=-1),
                                 mp.Near2FarRegion(mp.Vector3(y=+0.5*sizeY+addFlux), size=mp.Vector3(sizeX+2*addFlux, 0, sizeZ+2*addFlux), weight=+1))
                                 # )
sim.use_output_directory(output_directory)

pt = mp.Vector3(x=0, y=0, z=0)  #point in cell where we monitor the field decay
sim.run(mp.at_beginning(mp.output_epsilon), mp.at_every(10, mp.output_efield_x, mp.output_efield_y, mp.output_efield_z),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt, 1e-3))
#sim.run(mp.at_beginning(mp.output_epsilon), mp.at_every(10, mp.output_efield_x, mp.output_efield_y, mp.output_efield_z), until=200)
sim.output_farfields(nearfield_box,"farfield", resolution, center=mp.Vector3(0, 0, 1000), size=mp.Vector3(sizeX, sizeY, 0))
#sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt, 1e-3))
# buf = sim.get_farfield(nearfield_box, mp.Vector3(0,0))
# xFar = np.linspace(-sizeX/2, sizeX/2, sizeX*resolution)
# yFar = np.linspace(-sizeY/2, sizeY/2, sizeX*resolution)
# fieldPlane = np.zeros((len(xFar), len(yFar)))
# X, Y = np.meshgrid(xFar, yFar)
# for i in range(len(xFar)):
#     for j in range(len(yFar)):
#         buf = sim.get_farfield(nearfield_box, mp.Vector3(xFar[i], yFar[j], 10000))
#         fieldPlane[i, j] = np.abs(buf[0])

# plt.imshow(fieldPlane)
# plt.show()
# mpirun -np 8 python Simulation.py | tee log.out
