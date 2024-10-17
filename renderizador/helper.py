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

def interpolateNormals(normals, alpha, beta, gamma):
    n0 = normals[0]
    n1 = normals[1]
    n2 = normals[2]
    n = alpha * n0 + beta * n1 + gamma * n2
    return n

def generateNormal(tri_vertices):
    v0 = np.array(tri_vertices[0])
    v1 = np.array(tri_vertices[1])
    v2 = np.array(tri_vertices[2])
    n = np.cross(v1-v0,v2-v0)

    return n/np.linalg.norm(n)

def generateNormals(vertices,indices,flat=False):


    vertices = [vertices[i:i+3] for i in range(0,len(vertices),3)]
    num_vertices = len(vertices)

    faces = []
    face = []
    vertex_faces = {}
    face_count = 0
    for ind in indices:
        if ind != -1 and ind not in vertex_faces:
            vertex_faces[ind] = []
            
        if ind == -1:
            faces.append(face)

            vertex_faces[face[0]].append(face_count)
            vertex_faces[face[1]].append(face_count)
            vertex_faces[face[2]].append(face_count)
            face = []
            face_count += 1
            
            continue
        face.append(ind)

    
    
    vertex_normals = [np.array([0.0, 0.0, 0.0]) for _ in range(num_vertices)]

    # Calculate normals
    for vertex_id, face_ids in vertex_faces.items():
        face_normals = []
        for face_id in face_ids:
            face_vertices = [vertices[i] for i in faces[face_id]]
            face_normals.append(generateNormal(face_vertices))

        mean_normal = np.sum(face_normals, axis=0)
        norm = np.linalg.norm(mean_normal)
        if norm != 0:
            mean_normal /= norm
        vertex_normals[vertex_id] = mean_normal

    vertex_normals_flat = []
    for normal in vertex_normals:
        vertex_normals_flat.extend(normal.tolist())

    return vertex_normals_flat


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


def applyLighting(color, normal, light, view_dir):

    diffuse_c = np.array(color['diffuseColor'])
    specular_c = np.array(color['specularColor'])
    shininess_c = color['shininess']
    ok = False
    if np.linalg.norm(diffuse_c) == 0:
        ok = True
    ambient_intensity = light['ambientIntensity']
    light_color = np.array(light['color'])
    light_direction = -np.array(light['direction'])
    light_intensity = light['intensity']

    normal = normal / np.linalg.norm(normal)
    light_direction = light_direction / np.linalg.norm(light_direction)
    view_dir = view_dir / np.linalg.norm(view_dir)

    ambient = ambient_intensity * diffuse_c

    max_dot = max(np.dot(normal, light_direction), 0.0)
    diffuse = light_intensity * diffuse_c * max_dot


    half_vector = (light_direction + view_dir)
    half_vector /= np.linalg.norm(half_vector)
    spec_angle = max(np.dot(normal, half_vector), 0.0)
    specular = light_intensity * specular_c * (spec_angle ** (shininess_c * 128))


    color_result = light_color * (ambient + diffuse+specular)


    return color_result


