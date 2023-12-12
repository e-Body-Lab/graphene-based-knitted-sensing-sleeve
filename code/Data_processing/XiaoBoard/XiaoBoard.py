import pandas as pd
import numpy as np

def resCal(adc):
  r1 = 471
  Vin = 3.3
  buffer = adc * Vin
  Vout = buffer / 3750
  buffer = (Vin / Vout) - 1
  R = buffer * r1
  # R = (1024/adc-1)*r1;
  return R

data = pd.read_csv('liftup-1.csv', header=None)

rows = data.shape[0]
time = [None] * rows 
resistance = [None] * rows
for i in range(rows):
  current = data[0][i]
  index = current.find('->')
  time[i] = current[0:index]
  resistance[i] = current[index+2 : len(current)]

d = {'Time': time, 'Resistance': resistance}
df = pd.DataFrame(data=d)
df['Resistance'] = pd.to_numeric(df['Resistance'])


rows_new = int(rows / 12)
time_new = [None] * rows_new
r1 = [None] * rows_new
r2 = [None] * rows_new
r3 = [None] * rows_new
r4 = [None] * rows_new
r5 = [None] * rows_new
ax = [None] * rows_new
ay = [None] * rows_new
az = [None] * rows_new
gx = [None] * rows_new
gy = [None] * rows_new
gz = [None] * rows_new

counter = 0

for j in range(0,df.shape[0]-12,12):
    time_new[counter] = df.iloc[j+1,0]
    r1[counter] = resCal(df.iloc[j+1,1])
    r2[counter] = resCal(df.iloc[j+2,1])
    r3[counter] = resCal(df.iloc[j+3,1])
    r4[counter] = resCal(df.iloc[j+4,1])
    r5[counter] = resCal(df.iloc[j+5,1])
    ax[counter] = df.iloc[j+6,1]
    ay[counter] = df.iloc[j+7,1]
    az[counter] = df.iloc[j+8,1]
    gx[counter] = df.iloc[j+9,1]
    gy[counter] = df.iloc[j+10,1]
    gz[counter] = df.iloc[j+11,1]
    counter = counter + 1


d2 = {'Time': time_new, 'R1': r1, 'R2':r2, 'R3':r3, 'R4':r4, 'R5':r5, 'ax':ax, 'ay':ay, 'az':az, 'gx':gx, 'gy':gy, 'gz':gz }
df2 = pd.DataFrame(data=d2)

df2.to_excel("liftup-1.xlsx", index = False)
