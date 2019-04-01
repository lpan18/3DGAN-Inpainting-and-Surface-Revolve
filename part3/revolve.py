import numpy as np
import math
# import matplotlib.pyplot as plt
import bpy

def deCasteljau(cPoly, t):
    nrCPs = len(cPoly)
    P = np.zeros((nrCPs, 2, 4))
    P[:,:,0] = cPoly
    for i in range(nrCPs):
        for j in range(nrCPs-i-1):
            # print(i,j)
            P[j,:,i+1] = (1-t)*P[j,:,i] + t*P[j+1,:,i]
    return P[0,:,nrCPs-1]

def makeFaces(verts,num_u):
    faces=[]
    i = 0
    while(i < len(verts)):     
        if((i+1) % num_u == 0): 
            faces.append((i,i+1-num_u,i+1,i+num_u))
        else:
            faces.append((i,i+1,i+1+num_u,i+num_u))
        i += 1
    return faces

def main():
    # Init control polygon
    cPoly = np.loadtxt('/home/lpa25/Documents/CMPT743-a3/part3/controlpoints.txt', delimiter=' ')
    stepSize = 0.01
    num_c = int(1/stepSize)+1
    c = np.zeros((num_c, 2))

    # Iterate over curve
    idx = 0
    for i in np.arange(0,1+stepSize,stepSize):
        c[idx,:] = deCasteljau(cPoly, i)
        idx += 1
    # plt.plot(cPoly[:,0], cPoly[:,1],'b-s')
    # plt.plot(c[:,0],c[:,1],'r-')
    # plt.show()

    # Generate u
    PI = math.pi
    step_u = PI/30
    num_u = int(2*PI/step_u)+1
    
    # Revolving the curve around Z axis
    verts1=[]  

    for i in range(num_c):
        for u in np.arange(0,2*PI+step_u,step_u):
            verts1.append((c[i,0]*np.cos(u),c[i,0]*np.sin(u),c[i,1]))
            
    faces1 = makeFaces(verts1, num_u)      

    # Revolving the curve along the start-end line
    # calculate rotation angle from start-end axis to z-axis
    dy = c[0,0]-c[num_c-1,0]
    dz = c[0,1]-c[num_c-1,1]
    sin_theta = dy/math.sqrt(dy*dy + dz*dz)
    cos_theta = dz/math.sqrt(dy*dy + dz*dz)

    # Compute transformation matrix
    temp1 = np.zeros((num_c,1))
    temp2 = np.ones((num_c,1))
    concated_c = np.concatenate((temp1,c,temp2),axis=1)

    rotate_mat = np.array([[1,    0,      0,    0],
                        [0, cos_theta, -sin_theta, 0],
                        [0, sin_theta, cos_theta,  0],
                        [0,     0,      0,        1]])
            
    translate_mat = np.array([[1, 0, 0, -concated_c[0,0]],
                            [0, 1, 0, -concated_c[0,1]],
                            [0, 0, 1, -concated_c[0,2]],
                            [0, 0, 0,  1]])
                
    tranform_mat = np.dot(rotate_mat, translate_mat)

    # Transform all points
    transformed_c =  np.dot(concated_c, np.transpose(tranform_mat))
    # plt.plot(transformed_c[:,1],transformed_c[:,2],'b-s')
    # plt.show()

    # Revolving the curve around Z axis
    temp_verts = np.zeros((num_c*num_u,4))
    idx = 0
    for i in range(num_c):
        for u in np.arange(0,2*PI+step_u,step_u):
            temp_verts[idx] = [transformed_c[i,1]*np.cos(u),transformed_c[i,1]*np.sin(u),transformed_c[i,2],1]
            idx += 1

    temp_verts = np.dot(temp_verts, np.transpose(np.linalg.inv(tranform_mat)))[:,0:3]
    
    verts2 = [tuple(row) for row in temp_verts]
    faces2 = makeFaces(verts2, num_u)   

    # Render mesh
    # Curve
    c = np.concatenate((temp1,c),axis=1)
    verts0 = [tuple(row) for row in c]
    edges0 = []
    i = 0
    while(i < len(verts0)-1):     
        edges0.append((i,i+1))
        i += 1
    me0 = bpy.data.meshes.new('Curve' + 'Mesh')
    ob0 = bpy.data.objects.new('Curve', me0)
    bpy.context.scene.objects.link(ob0)
    me0.from_pydata(verts0, edges0, [])

    # Revolve along Z axis
    me1 = bpy.data.meshes.new('Revolve-Z-axis' + 'Mesh')
    ob1 = bpy.data.objects.new('Revolve-Z-axis', me1)
    bpy.context.scene.objects.link(ob1)
    me1.from_pydata(verts1, [], faces1)

    # Revolve along start-end axis
    me2 = bpy.data.meshes.new('Revolve-Start-End-axis' + 'Mesh')
    ob2 = bpy.data.objects.new('Revolve-Start-End-axis', me2)
    bpy.context.scene.objects.link(ob2)
    me2.from_pydata(verts2, [], faces2)

if __name__ == '__main__':
    main()