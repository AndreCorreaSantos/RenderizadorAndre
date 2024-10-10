import math

def generateSphereVertices(radius, PI, sectorCount, stackCount):
    vertices = []
    # x, y, z, xy, nx, ny, nz, lengthInv = 1.0 / radius, 1.0 / radius, 1.0 / radius, 1.0 / radius
    # s, t = 0, 0
    sectorStep = 2 * PI / sectorCount
    stackStep = PI / stackCount
    sectorAngle, stackAngle = 0, 0

    for i in range(0, stackCount + 1):
        stackAngle = PI / 2 - i * stackStep
        xy = radius * math.cos(stackAngle)
        z = radius * math.sin(stackAngle)

        for j in range(0, sectorCount + 1):
            sectorAngle = j * sectorStep
            x = xy * math.cos(sectorAngle)
            y = xy * math.sin(sectorAngle)
            vertices.append([x, y, z])
    return vertices


def generateMeshIndices(sectorCount, stackCount):
    indices = []
    k1, k2 = 0, 0
    for i in range(0, stackCount):
        k1 = i * (sectorCount + 1)
        k2 = k1 + sectorCount + 1
        for j in range(0, sectorCount):
            if i != 0:
                indices.extend([k1, k2, k1 + 1])
            if i != (stackCount - 1):
                indices.extend([k1 + 1, k2, k2 + 1])
    return indices