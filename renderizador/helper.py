import math

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
