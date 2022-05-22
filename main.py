# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:19:12 2022

@author: Tavi
"""
import math
import numpy as np
import bitstring
import random
from skimage import io
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread

a = 1.76
b = 0.1
miu = 10**5
M = 'CRIPTOGRAFIETEORIAHAOSULUIHENON3DIMENSIONALSIRDECARACTEREFOARTELUNG'
n = 32 # 32 biti
p = 4 # number of threads
#M = io.imread(r'C:\Users\Tavi\Desktop\Licenta\imagine\lena.png')
dims = np.shape(M)

t1 = datetime.now()


#plt.figure(), plt.imshow(img), plt.show()

encrypted = []


def Henon3D(): # Henon three dimensional discrete-time map
    if dims:
        m = np.reshape(M, np.prod(dims))
        np.append(m, [1, 1, 1] * random.randrange(65,90))
    else:
        m = np.ones(len(M)+3) * random.randrange(65,90)
        for i in range(len(M)):
            m[i] = ord(M[i])
    x1 = np.zeros(len(m))
    x2 = np.zeros(len(m))
    x3 = np.zeros(len(m))
    x1[0] = 0.8147
    x2[0] = 0.9057
    x3[0] = 0.1269
    for i in range(len(m)-1):
        x1[i+1] = a - x2[i]**2 - b * x3[i] + m[i]/miu
        x2[i+1] = x1[i]
        x3[i+1] = x2[i]
    return x3


def BitConversion(x3): # conversia din float in 32 biti
    encrypted = []
    for i in range(len(x3)):
        f1 = bitstring.BitArray(float=x3[i], length=32)
        l = np.zeros(n)
        for j in range(n):
            l[j] = f1[j]
        encrypted.append(l)
    return encrypted

def BitConversion2(id, x3): # conversia din float in 32 biti

    s = len(x3) // p
    kp = len(x3) % p
    if kp == 0:
        first = s*id
        last = first + s
    elif id < kp:
        first = id * (s+1)
        last = first + s + 1
    else:
        first = s * id + kp
        last = first + s
    for i in range(first,last):
        print(i)
        f1 = bitstring.BitArray(float=x3[i], length=32)
        l = np.zeros(n)
        for j in range(n):
            l[j] = f1[j]
        encrypted.append(l)


def GenerareMatrice(): # Creare matrice pentru permutare
    T = np.zeros((n, n))
    for i in range(n - 1, (n - 1) // 2 - n % 2, -1):
        T[n - 1 - i][i] = 1
    for i in range(0, (n - 1) // 2 + ((n + 1) % 2), 1):
        T[(n - 1) // 2 + 1 + i][i] = 1
    return T

def BitToFloat(encrypted): # conversia din insiruirea de biti in numere de tip float
    x = []
    for i in range(len(encrypted)):
        if (32-n) != 0:
            encrypted[i] = np.append(encrypted[i], np.zeros(32 - n))  # adaugare biti de 0 pentru a obtine un sir de 32 biti
        s = ''
        for j in encrypted[i]:
            s += str(int(j))  # conversia celor 32 biti din lista intr-un string
        f1 = bitstring.BitArray(bin=s)  # conversia sirului binar intr-un numar de tip float
        x.append(f1.float)
    return x

def RefacereMesaj(x): # conversia din biti in string
    n = np.zeros(len(M))
    for i in range(len(M)):
        n[i] = (miu * (x[i + 3] + x[i + 1] ** 2 + b * x[i] - a))  # refacerea mesajului initial in cod ASCII
    s = ""
    for i in n:
        s += chr(round(i))  # refacerea mesajului initial sub forma de string
    return s

def RefacereImagine(x): # conversia din biti in matricea initiala
    n = np.zeros(np.prod(dims))
    for i in range(np.prod(dims) - 3):
        n[i] = round((miu * (x[i + 3] + x[i + 1] ** 2 + b * x[i] - a)))  # refacerea mesajului initial in cod ASCII
    s = np.reshape(n,dims)
    return s

x3 = Henon3D()

#fig 2
plt.figure(),plt.plot(x3)
plt.xlabel("time[s]")
plt.ylabel("Amplitude")
plt.show()

#fig 4
ps = np.abs(np.fft.fft(x3))**2
time_step = 1 / 30
freqs = np.fft.fftfreq(x3.size, time_step)
idx = np.argsort(freqs)

plt.plot(freqs[idx], ps[idx])
plt.xlabel("Frequency[Hz]")
plt.ylabel("Spectral density Energy")
plt.show()

T = GenerareMatrice()
Trev = np.linalg.inv(T)   # calculeaza inversa matricei T


'''
threads = []
for i in range(p):
    t = Thread(target=BitConversion2, args=(i,x3))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
'''
encrypted = BitConversion(x3)

for i in range(len(encrypted)):
    encrypted[i] = np.matmul(encrypted[i],T) # schimbam ordinea bitilor

#fig 3
plt.figure(),plt.plot(encrypted)
plt.xlabel("time[s]")
plt.ylabel("Amplitude")
plt.show()

#fig 5
ps = np.abs(np.fft.fft(encrypted))**2
time_step = 1 / 30
freqs = np.fft.fftfreq(len(encrypted), time_step)
idx = np.argsort(freqs)

plt.plot(freqs[idx], ps[idx])
plt.xlabel("Frequency[Hz]")
plt.ylabel("Spectral density Energy")
plt.show()

# decriptare

for i in range(len(encrypted)):
    encrypted[i] = np.matmul(encrypted[i], Trev) # reintoarcere la ordinea initiala a bitilor
print("inmultire matrici2")
x = BitToFloat(encrypted)
print("conversie in float")
s = RefacereMesaj(x)

print(s)
'''
s = RefacereImagine(x)
s = np.uint8(s) # conversia elementelor de tip float in uint8

plt.figure(),plt.imshow(s), plt.show()
'''

