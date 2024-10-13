import math
import numpy as np

def generateSphereVertices(radius, sectorCount, stackCount):
    vertices = []
    sectorStep = 2 * math.pi / sectorCount
    stackStep = math.pi / stackCount

    for i in range(stackCount + 1):
        stackAngle = math.pi / 2 - i * stackStep  
        xy = radius * math.cos(stackAngle)       
        z = radius * math.sin(stackAngle)        
        for j in range(sectorCount + 1):
            sectorAngle = j * sectorStep         
            x = xy * math.cos(sectorAngle)        
            y = xy * math.sin(sectorAngle)       
            vertices.extend([x, y, z])

    return vertices

def generateMeshIndices(sectorCount, stackCount):
    indices = []

    for i in range(stackCount):
        k1 = i * (sectorCount + 1)    
        k2 = k1 + sectorCount + 1     
        for j in range(sectorCount):
            if i != 0:
                indices.extend([k1 + j, k2 + j, k1 + j + 1])
                indices.extend([-1])
            if i != (stackCount - 1):
                indices.extend([k1 + j + 1, k2 + j, k2 + j + 1])
                indices.extend([-1])
    return indices

def generateSphereNormals(center,vertices): # can also be used for spheres
    normals = []
    for i in range(0,len(vertices)-1,3):
        v = vertices[i:i+3]
        #checar o calculo dessas normais depois
        n = (v-center)/np.linalg.norm(v-center)
        normals.extend(n)
    return normals

def averageTriNormals(normals):

    x_mean = np.array([normals[0][0],normals[1][0],normals[2][0]]).mean()
    y_mean = np.array([normals[0][1],normals[1][1],normals[2][1]]).mean()
    z_mean = np.array([normals[0][2],normals[1][2],normals[2][2]]).mean()

    avg_n = np.array([x_mean,y_mean,z_mean])
    return avg_n


def generateCircleVertices(center,radius, sectorCount):
    vertices = []
    sectorStep = 2 * math.pi / sectorCount
    for i in range(sectorCount):
        sectorAngle = i * sectorStep
        x = center[0] + radius * math.cos(sectorAngle)
        z = center[2] + radius * math.sin(sectorAngle)
        vertices.extend([x, center[1], z])
    return vertices


def get_level(dudx, dudy, dvdx, dvdy):
    epsilon = 1e-6
    l = max(np.sqrt(dudx**2 + dvdx**2), np.sqrt(dudy**2 + dvdy**2), epsilon)
    level = int(np.log2(l))
    level = max(0, level)
    return level


def compute_barycentric_coordinates(tri, x, y):
    x1, y1 = tri[0], tri[1]
    x2, y2 = tri[3], tri[4]
    x3, y3 = tri[6], tri[7]
    d = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
    alpha = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / d
    beta = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / d
    gamma = 1 - alpha - beta
    return alpha, beta, gamma


def get_mipmaps(texture): # returns a list of downsampled images to be used as mipmaps, REMEMBER TO REFACTOR THIS LATER IT STINKS
    mipmap_levels = [texture]  
    mip = texture.copy()
    height = mip.shape[0]
    width = mip.shape[1]
    condition = height > 1 and width > 1
    while condition:
        new_height = max(1, height // 2)
        new_width = max(1, width // 2)
        blurred_image = np.zeros((new_height, new_width, mip.shape[2]), dtype=mip.dtype)
        for y in range(new_height):
            for x in range(new_width):
                filter = mip[2 * y:2 * y + 2, 2 * x:2 * x + 2]
                blurred_image[y, x] = np.mean(filter, axis=(0, 1))
        mipmap_levels.append(blurred_image)
        mip = blurred_image
        height = mip.shape[0]
        width = mip.shape[1]
        condition = height > 1 and width > 1
    return mipmap_levels


