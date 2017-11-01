#Dicretized Gerchberg-Saxton algorithm
#Kim Myungjoon (vcxzx@kaist.ac.kr)

#KAIST, Department of Material Science and Engineering
#Advanced Photonic Materials & Devices Laboratory (apmd.kaist.ac.kr)

#Dependency : Python > 3.4 version, numpy > 1.10 version, matplotlib, PIL

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from PIL import Image

phase_level = int(input("phase level : "))
tolerance = float(input("convergence tolerance  : "))
filename = 'Lenna.png'
try:
    im = Image.open(filename)
except FileNotFoundError as e:
    print(e)


if filename.find('.') == -1:
    filename_only = filename
else:
    filename_only = filename[::-1]
    filename_only = filename_only[filename_only.find('.')+1:]
    filename_only = filename_only[::-1]


random.seed()

##############
# image load #

    
im = im.convert('L')
pixel = im.load()
Total_Intensity=0
Target_Intensity = np.zeros((im.size[0],im.size[1]))
Source_Intensity = np.zeros((im.size[0],im.size[1]))
Initial_Phase = np.zeros((im.size[0],im.size[1]))

for i in range(im.size[0]):
    for j in range(im.size[1]):
        Source_Intensity[i,j] = 1
        Initial_Phase[i,j] = random.uniform(-np.pi,np.pi)
        if not pixel[i,j]==0:
            Target_Intensity[i,j] = pixel[i,j]
            Total_Intensity += pixel[i,j]
    
#############
            
Target_Intensity = Target_Intensity / Total_Intensity * sum(sum(Source_Intensity)) #normalize

#############
x_len = im.size[0]
y_len = im.size[1]
#####################################################################


def Phase_Discretization(Phase):
    return np.around(Phase/np.pi*(phase_level/2))/(phase_level/2)*np.pi

def GS(Source, Target):
    Target_Amplitude = np.sqrt(Target)
    Source_Amplitude = np.sqrt(Source)    
    A = Initial_Phase
    B = np.zeros((im.size[0],im.size[1]))
    C = np.zeros((im.size[0],im.size[1]))
    D = np.zeros((im.size[0],im.size[1]))

    previous_error = 0
    while True:
        B = Source_Amplitude*np.exp(1j*np.angle(A))

        
        C = np.fft.fft2(B,norm="ortho")
        C_ = np.fft.fftshift(C)
        
        error_value = Error_calculation(C_)
        print("Intensity Error : ", error_value)
        if abs(error_value-previous_error) < 10**(-1*tolerance):
            break
        D = Target_Amplitude*np.exp(1j*np.angle(C_))
        D_ = np.fft.fftshift(D)
        A = np.fft.ifft2(D_,norm="ortho")
        previous_error = error_value
        
    return Phase_Discretization(np.angle(A)), error_value

def Error_calculation(C):
    Amplitude = abs(C)
    Target_Amplitude = np.sqrt(Target_Intensity)
    Difference = sum(sum((Target_Amplitude - Amplitude)**2))
    return Difference

      
Current_Phase = np.zeros((im.size[0],im.size[1]))
Minimum_Phase, Error = GS(Source_Intensity,Target_Intensity)

Minimum_Wave = np.sqrt(Source_Intensity)*np.exp(1j*Minimum_Phase)
F = np.fft.fft2(Minimum_Wave, norm = "ortho")
F_ = np.fft.fftshift(F)
Calculated_Intensity = abs(F_)**2

### change x,y to fit original image ###
Target_Intensity2 = np.zeros((im.size[1],im.size[0]))
Calculated_Intensity2 = np.zeros((im.size[1],im.size[0]))
Minimum_Phase2 = np.zeros((im.size[1],im.size[0]))


for i in range(im.size[0]):
    for j in range(im.size[1]):
        Target_Intensity2[j][i] = Target_Intensity[i][-1*(j+1)]
        Calculated_Intensity2[j][i] = Calculated_Intensity[i][-1*(j+1)]
        Minimum_Phase2[j][i] = Minimum_Phase[i][-1*(j+1)]

Phase_Value = np.around(Minimum_Phase2/np.pi*(phase_level/2))
for i in range(im.size[0]):
    for j in range(im.size[1]):
        if Phase_Value[j][i] == -1*phase_level/2:
            Phase_Value[j][i] = phase_level/2
#################            
            
### file writing ###            
Phase = Phase_Value.astype(int)

Error = Error_calculation(F_)
print("Final Error (Discretization) : ", Error)

f = open(filename_only + "_PhaseMask.txt",'w')
for i in range(im.size[1]-1,-1,-1):
    for j in range(im.size[0]):
        f.write(str(Phase[i][j]) + ' ')
    f.write('\n')
f.close()
###################


###plot###
x = np.arange(int(-x_len/2),int(x_len/2))
y = np.arange(int(-y_len/2),int(y_len/2))
plt.title('Calculated Intensity')
plt.pcolormesh(x,y,Calculated_Intensity2)
plt.set_cmap('gray')
plt.xlim(int(-x_len/2),int(x_len/2)-1)
plt.ylim(int(-y_len/2),int(y_len/2)-1)

plt.colorbar()
plt.clim(0,10)
plt.savefig(filename_only + '_Intensity.png', bbox_inches='tight')

plt.show()
###########
