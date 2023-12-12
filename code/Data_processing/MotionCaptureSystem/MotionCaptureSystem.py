import numpy as np
import pandas as pd
from scipy.spatial import distance

motion_data = pd.read_csv('HC45-2.csv',header = None)
motion_data.columns = ["x1","y1","z1","x2","y2","z2","x3","y3","z3"]

motion_data = motion_data.dropna()

x2 = motion_data["x1"]
y2 = motion_data["y1"]
z2 = motion_data["z1"]
x3 = motion_data["x2"]
y3 = motion_data["y2"]
z3 = motion_data["z2"]
x4 = motion_data["x3"]
y4 = motion_data["y3"]
z4 = motion_data["z3"]

motion_data.index[np.isinf(motion_data).any(1)]

rows = len(motion_data)

zero_data = np.zeros(shape=(len(motion_data),1))
angle = pd.DataFrame(zero_data, columns=['Angle'])

for i in range(rows):
    a = np.array((x2.iloc[i],y2.iloc[i],z2.iloc[i]))
    b = np.array((x3.iloc[i],y3.iloc[i],z3.iloc[i]))
    c = np.array((x4.iloc[i],y4.iloc[i],z4.iloc[i]))
    dist1 = distance.euclidean(a, b)
    dist2 = distance.euclidean(c, b)
    dist3 = distance.euclidean(a, c)
    angle3 = (dist1*dist1 + dist2*dist2 - dist3*dist3)/(2*dist1*dist2)
    angle3 = np.arccos(angle3)
    angle.loc[i,'Angle'] = np.degrees(angle3)

angle.to_excel('HC45-2.xlsx')
