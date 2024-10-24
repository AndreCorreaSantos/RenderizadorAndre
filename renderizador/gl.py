#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: André Corrêa Santos
Disciplina: Computação Gráfica
Data: 12/08/2024
"""

import time  # Para operações com tempo
import gpu  # Simula os recursos de uma GPU
import math  # Funções matemáticas
import numpy as np  # Biblioteca do Numpy
import helper


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    perspective_matrix = np.mat([])
    look_at = np.mat([])
    transform_stack = []
    normal_transform_stack = []
    vertex_colors = []
    vertex_tex_coord = []
    image = None
    colorPerVertex = False
    directionalLights = []
    viewDirection = []
    position = np.array([0,0,0])
    flat = False
    invertNormals = False
    

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    super_buffer = None
    z_buffer = None

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.super_buffer = np.zeros((GL.width*2, GL.height*2, 3), dtype=np.uint8)
        GL.z_buffer =  - np.inf * np.ones((GL.width*2, GL.height*2)) 


    @staticmethod
    def polypoint2D(point: list[float], colors: dict[str, list[float]]) -> None:
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        color = np.array(colors["emissiveColor"]) * 255

        for i in range(0, len(point), 2):
            x = int(point[i])
            y = int(point[i + 1])
            if (x <=GL.width and x >= 0) and (y <=GL.width and y >= 0):
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

    @staticmethod
    def polyline2D(lineSegments: list[float], colors: dict[str, list[float]]) -> None:
        """Função usada para renderizar Polyline2D."""

        color = np.array(colors["emissiveColor"]) * 255

        for i in range(0, len(lineSegments) - 2, 2):
            p0 = [lineSegments[i], lineSegments[i + 1]]
            p1 = [lineSegments[i + 2], lineSegments[i + 3]]

            if p0[0] > p1[0]:
                p0, p1 = p1, p0

            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]

            if dx != 0:  # evitar divisao por zero
                slope = dy / dx
            else:
                slope = 10**5  # caso divisao por zero chutar numero grande

            if np.abs(slope) <= 1:
                y = p0[1]

                for x in range(int((p0[0])), int((p1[0]))):
                    if (x <GL.width and x >= 0) and (y <GL.height and y >= 0):
                        gpu.GPU.draw_pixel([int(x), int((y))], gpu.GPU.RGB8, color)
                    y += slope
            else:
                if p0[1] > p1[1]:

                    p0, p1 = p1, p0

                slope = 1 / slope
                x = p0[0]
                for y in range(int((p0[1])), int((p1[1]))):
                    if (x <GL.width and x >= 0) and (y <GL.height and y >= 0):
                        gpu.GPU.draw_pixel([int((x)), int(y)], gpu.GPU.RGB8, color)
                    x += slope

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""

        color = np.array(colors["emissiveColor"]) * 255.0

        tolerance = 20.0

        for x in range(0, GL.width):
            for y in range(0, GL.height):
                inPerimeter = abs((x) ** 2 + (y) ** 2  - radius**2) <= tolerance
                inScreen = (x >= 0 and x < GL.width) and (y >= 0 and y < GL.height)
                if inPerimeter and inScreen:

                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)


    # caso normais, para um dado pixel projetado 2D, pegar os valores 3D do pixel na superficie
    # os dados 3D do pixel, posicao deve sofrer rotacao e translacao, normal deve receber rotacao --> quais transformações eu aplico nas normais??
    # interpolar a normal, nao sei como --> por enquanto pegar a média
    # 
    @staticmethod
    def triangleSet2D(vertices, colors, vertex_colors=None, texture_values=None,world_values=None):
        """Function used to render TriangleSet2D with depth testing, barycentric interpolation, and texture mapping."""

        if texture_values is not None and GL.image is not None:
            mipmaps = helper.get_mipmaps(GL.image)
    

        if len(vertices) < 9:
            if len(vertices) == 6:
                new_vertices = []   
                for i in range(0, len(vertices), 2):
                    new_vertices.append(vertices[i])
                    new_vertices.append(vertices[i + 1])
                    new_vertices.append(-1.0)
                vertices = new_vertices
            else:
                print("ERROR NO TRIANGLES SENT")
                return

        original_color = colors
        color = np.array(colors["emissiveColor"]) * 255
        if GL.colorPerVertex:
            vertex_colors = np.array(vertex_colors) * 255
        else:
            color = np.array(colors["emissiveColor"]) * 255

        

        transparency = colors.get("transparency", 0.0)

    

        tex_index = 0 # Index for texture_values
        for i in range(0, len(vertices), 9):
            tri = vertices[i : i + 9]

            if world_values is not None:
                world_points = world_values[0]
                world_normals = world_values[1]
            
                index = i // 3
                triw_points = world_points[index:index+3]
                triw_normals = np.array(world_normals[index:index+3])


            if len(tri) != 9:
                continue

            xs = [tri[j] for j in range(0, len(tri), 3)]  # x coordinates
            ys = [tri[j] for j in range(1, len(tri), 3)]  # y coordinates
            zs = [tri[j] for j in range(2, len(tri), 3)]  # z coordinates

            # Bounding Box
            box = [int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))]
            super_factor = 1
            # Scale only x and y for super sampling
            super_box = [super_factor* v for v in box]
            super_tri = []
            for j in range(0, len(tri), 3):
                super_tri.extend([super_factor* tri[j], super_factor* tri[j + 1], tri[j + 2]])  # Scale x and y, keep z

            if GL.colorPerVertex:
                # Extract per-triangle colors
                
                tri_colors = vertex_colors[i : i + 9]
                if len(tri_colors) != 9:
                    continue
                c1 = np.array(tri_colors[0:3])
                c2 = np.array(tri_colors[3:6])
                c3 = np.array(tri_colors[6:9])

                
            
            if texture_values is not None and GL.image is not None:
                tri_tex_coords = texture_values[tex_index : tex_index + 6]
                tex_index += 6  # Increment texture index by 6 for the next triangle

                if len(tri_tex_coords) != 6:
                    continue
                
                u1, v1 = tri_tex_coords[0], tri_tex_coords[1]
                u2, v2 = tri_tex_coords[2], tri_tex_coords[3]
                u3, v3 = tri_tex_coords[4], tri_tex_coords[5]

 
            # Corresponding z-values at each vertex
            z1, z2, z3 = tri[2], tri[5], tri[8]
            # Compute 1/z for each vertex
            w1 = 1.0 / z1 if z1 != 0 else 0.0
            w2 = 1.0 / z2 if z2 != 0 else 0.0
            w3 = 1.0 / z3 if z3 != 0 else 0.0
            pixel_normal = None
        
            # Iterating over the bounding box
            for x in range(super_box[0], super_box[1] + 1):
                for y in range(super_box[2], super_box[3] + 1):

                    if (0 <= x < GL.width * 2) and (0 <= y < GL.height * 2):
                        
                        bary_coords = helper.compute_barycentric_coordinates(super_tri, x + 0.5, y + 0.5)
                        if bary_coords is None:
                            continue
                        alpha, beta, gamma = bary_coords

                        # Check if the point is inside the triangle
                        if alpha < 0 or beta < 0 or gamma < 0:
                            continue  # Point is outside the triangle

                        # Interpolate z-value
                        z = alpha * z1 + beta * z2 + gamma * z3

                        if transparency == 0:
                            # Opaque pixel
                            if z > GL.z_buffer[x][y]:
                                GL.z_buffer[x][y] = z
                            else:
                                continue  # Discard pixel if behind another triangle

                        # Perspective-correct interpolation
                        one_over_z = alpha * w1 + beta * w2 + gamma * w3
                        if one_over_z == 0:
                            continue  # Avoid division by zero

                        if texture_values is not None and GL.image is not None:
                            scale_factor = GL.image.shape[0]

                            # Perspective-correct interpolation of u and v
                            u = (alpha * u1 * w1 + beta * u2 * w2 + gamma * u3 * w3) / one_over_z
                            v = (alpha * v1 * w1 + beta * v2 * w2 + gamma * v3 * w3) / one_over_z

                            # Interpolate u and v at neighboring pixels (x+1, y) and (x, y+1) for mipmap level calculation
                            a10, b10, g10 = helper.compute_barycentric_coordinates(super_tri, x + 1.0, y)
                            u10 = (a10 * u1 * w1 + b10 * u2 * w2 + g10 * u3 * w3) / (a10 * w1 + b10 * w2 + g10 * w3)
                            v10 = (a10 * v1 * w1 + b10 * v2 * w2 + g10 * v3 * w3) / (a10 * w1 + b10 * w2 + g10 * w3)

                            a01, b01, g01 = helper.compute_barycentric_coordinates(super_tri, x, y + 1.0)
                            u01 = (a01 * u1 * w1 + b01 * u2 * w2 + g01 * u3 * w3) / (a01 * w1 + b01 * w2 + g01 * w3)
                            v01 = (a01 * v1 * w1 + b01 * v2 * w2 + g01 * v3 * w3) / (a01 * w1 + b01 * w2 + g01 * w3)

                            # Calculate derivatives for mipmap level
                            dudx = scale_factor * (u10 - u)
                            dvdx = scale_factor * (v10 - v)
                            dudy = scale_factor * (u01 - u)
                            dvdy = scale_factor * (v01 - v)

                            level = helper.get_level(dudx, dudy, dvdx, dvdy)

                            # Select the appropriate mipmap level
                            mipmap = mipmaps[level]
                            map_scale = mipmap.shape[0]

                            # Handle texture wrapping or clamping
                            u = int(u * map_scale) % map_scale
                            v = int(v * map_scale) % map_scale  # Flip v-axis if necessary
                            uv = np.array([u, v])
                            
                            rotation_matrix = np.array([[0, -1],
                                                        [1,  0]])
                            # Apply the 90-degree clockwise rotation
                            u,v= np.dot(rotation_matrix, uv)

                            # Get pixel color from the mipmap
                            color = mipmap[v][u][0:3]


                        elif world_values is not None:
                            pixel_normal = helper.interpolateNormals(triw_normals, alpha, beta, gamma)
                            pixel_position = helper.interpolateNormals(triw_points, alpha, beta, gamma)
                        



                        elif GL.colorPerVertex:
                            
                            # Interpolate color components with perspective correction
                            r = (alpha * c1[0] * w1 + beta * c2[0] * w2 + gamma * c3[0] * w3) / one_over_z
                            g = (alpha * c1[1] * w1 + beta * c2[1] * w2 + gamma * c3[1] * w3) / one_over_z
                            b = (alpha * c1[2] * w1 + beta * c2[2] * w2 + gamma * c3[2] * w3) / one_over_z

                            pixel_color = np.array([r, g, b])
                            pixel_color = np.clip(pixel_color, 0, 255).astype(np.uint8)
                            color = pixel_color
                            # #setting to white for debug
                            # color = np.array([255,255,255])

                        if pixel_normal is not None and len(GL.directionalLights) > 0:
                           
                            final_color = colors["emissiveColor"]
                            v = (GL.position - pixel_position)/np.linalg.norm(GL.position - pixel_position)
                            # print(len(GL.directionalLights))
                            
                            for light in GL.directionalLights:

                                final_color += helper.applyLighting(original_color, pixel_normal, light,v)

                            # color = final_color*255
                            # color = (pixel_normal + 1)*127.5
                            # if final_color[0] > 1 or final_color[1] > 1 or final_color[2] > 1:
                            #     print("ERROR")
                            
                            color = np.clip(final_color*255,0,255).astype(np.uint8)

                            # print(GL.directionalLights)
                        # Handle transparency blending
                        if transparency > 0:
                            previous_color = GL.super_buffer[x][y]
                            opacity = 1 - transparency
                            blended_color = [
                                int(color[0] * opacity + previous_color[0] * transparency),
                                int(color[1] * opacity + previous_color[1] * transparency),
                                int(color[2] * opacity + previous_color[2] * transparency),
                            ]
                            GL.super_buffer[x][y] = blended_color
                        else:
                            # No transparency, overwrite the color
                            GL.super_buffer[x][y] = color

            # Downsample and draw pixels
            for x in range(box[0], box[1] + 1):
                for y in range(box[2], box[3] + 1):
                    if (0 <= x < GL.width) and (0 <= y < GL.height):
                        
                        c1 = GL.super_buffer[super_factor * x][super_factor * y]
                        c2 = GL.super_buffer[super_factor * x][super_factor * y + 1]
                        c3 = GL.super_buffer[super_factor * x + 1][super_factor * y]
                        c4 = GL.super_buffer[super_factor * x + 1][super_factor * y + 1]
                        super_colors = np.array([c1, c2, c3, c4]).mean(axis=0).astype(np.uint8)
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, super_colors)
    @staticmethod
    def triangleSet(point, colors,vertex_colors=None,texture_values=None,normals=None):
        """Função usada para renderizar TriangleSet."""

        
        # Helper function to multiply matrices
        def multiply_mats(mat_list):
            accumulator = np.identity(mat_list[0].shape[0])
            for mat in mat_list:
                accumulator = accumulator @ mat
            return accumulator

        # Helper function to transform 3D points to 2D
        def transform_points(points, min_x, min_y, min_z, max_z):
            w = GL.width
            h = GL.height
            delta_z = max_z - min_z

            screenMatrix = np.mat([
                [w / 2, 0.0, 0.0, min_x + w / 2],
                [0.0, -h / 2, 0.0, min_y + h / 2],
                [0.0, 0.0, delta_z, min_z],
                [0.0, 0.0, 0.0, 1.0]
            ])
            transformed_points = []
            for i in range(0, len(points), 3):
                p = points[i:i + 3]
                p.append(1.0)  # homogeneous coordinate
                p = np.array(p)

                # Apply all transformation matrices
                transform_mat_res = multiply_mats(GL.transform_stack)
                look_at_p = GL.look_at @ transform_mat_res @ p
                z = np.array(look_at_p).flatten()[2]

                p = GL.perspective_matrix @ GL.look_at @ transform_mat_res @ p

                # Z-Divide
                p = np.array(p).flatten()
                p = p / p[-1]
                p = screenMatrix @ p

                p = np.array(p).flatten()
                transformed_points.append(p[0])
                transformed_points.append(p[1])
                transformed_points.append(z)

            return transformed_points

        # Transform the 3D points to 2D
        xs = [point[i] for i in range(0, len(point), 3)]
        ys = [point[i] for i in range(1, len(point), 3)]
        zs = [point[i] for i in range(2, len(point), 3)]
        world_values = None

        if normals is not None:
            #packaging normals coordinates

            ns = [] # contains xyz for normal
            for i in range(0,len(normals),3):
                n = normals[i:i+3]
                ns.append(n)
            
            world_points = []
            world_normals = []
            for i in range(0,len(xs)):
                # point in world space
                obj_to_world = multiply_mats(GL.transform_stack)
                world_p = obj_to_world @ np.array([xs[i],ys[i],zs[i],1.0])

                # normal in world space
                n_to_world = multiply_mats(GL.normal_transform_stack)
                n = ns[i]
                n.append(1.0)
                world_n = np.array(n)
                world_n = n_to_world @ world_n


                # throwing away homogenous coordinates
                world_p = np.array(list(np.array(world_p).flatten())[0:3])
                world_n = np.array(list(np.array(world_n).flatten())[0:3])

                world_points.append(world_p)
                world_normals.append(world_n)


            # if GL.invertNormals:
            #     normals = [-n for n in normals]

            world_values = [world_points,world_normals]

        elif len(GL.directionalLights) > 0: #  # ACHO QUE ESTOU PROJETANDO ERRADO PARA ESPACO DO MUNDO
            # calculate flat normals here!
            normals = []
            points = []
            obj_to_world = multiply_mats(GL.transform_stack)
            for i in range(0, len(xs), 3):
       
                tri_vertices = []

                for j in range(0,3):
                    vertex = [xs[i+j],ys[i+j],zs[i+j],1.0]
                    world_vertex = obj_to_world @ vertex
                    tri_vertices.append(np.array(list(np.array(world_vertex).flatten())[0:3]))
                    

                n = np.array(helper.generateNormal(tri_vertices))
                normals.extend([n,n,n])

                points.extend(np.array(tri_vertices))

            # if GL.invertNormals:
            #     normals = [-n for n in normals]
            world_values = [points,normals]

       
        projected_vertices = transform_points(point, min(xs), min(ys), min(zs), max(zs))

        # Call triangleSet2D with the transformed 2D vertices

        

        GL.triangleSet2D(projected_vertices, colors,vertex_colors,texture_values=texture_values,world_values=world_values)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.

        #LÓGICA TRANSLAÇÕES E ROTAÇÕES LOOK AT

        # Consertar 
        GL.position = np.array(position)
        GL.viewDirection =  orientation
        cam_pos = np.matrix([
            [1.0,0.0,0.0,position[0]],
            [0.0,1.0,0.0,position[1]],
            [0.0,0.0,1.0,position[2]],
            [0.0,0.0,0.0,        1.0],
        ])
        
        x = orientation[0]
        y = orientation[1]
        z = orientation[2]
        t = orientation[3]
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        rotation_m = np.mat([
            [cos_t + x**2 * (1 - cos_t), x * y * (1 - cos_t) - z * sin_t, x * z * (1 - cos_t) + y * sin_t, 0],
            [y * x * (1 - cos_t) + z * sin_t, cos_t + y**2 * (1 - cos_t), y * z * (1 - cos_t) - x * sin_t, 0],
            [z * x * (1 - cos_t) - y * sin_t, z * y * (1 - cos_t) + x * sin_t, cos_t + z**2 * (1 - cos_t), 0],
            [0, 0, 0, 1]
        ])

        look_at_trans =  np.linalg.inv(cam_pos)
        look_at_rot = np.linalg.inv(rotation_m)

        # TRANSLADANDO E DEPOIS ROTACIONANDO
        look_at_mat = look_at_rot@look_at_trans
        GL.look_at = look_at_mat


        aspect_ratio = GL.width/GL.height
        near = GL.near
        far = GL.far
        top = near * np.tan(fieldOfView / 2)
        right = top * aspect_ratio

        perspective_m = np.matrix([
            [near / right, 0.0, 0.0, 0.0],
            [0.0, near / top, 0.0, 0.0],
            [0.0, 0.0, -(far + near) / (far - near), -2.0 * (far * near) / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ])

        # retornando matriz que aplica LOOK_AT e projeção perspectiva
        # print("perspective")
        # print(perspective_m)

        GL.perspective_matrix = perspective_m




    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        
        scale_m = np.mat([
            [scale[0],0.0,0.0,0.0],
            [0.0,scale[1],0.0,0.0],
            [0.0,0.0,scale[2],0.0],
            [0.0,0.0,0.0,1.0]
            ]
        )
        translation_m = np.mat([
            [1.0,0.0,0.0,translation[0]],
            [0.0,1.0,0.0,translation[1]],
            [0.0,0.0,1.0,translation[2]],
            [0.0,0.0,0.0,1.0],
            ]
        )
        x = rotation[0]
        y = rotation[1]
        z = rotation[2]
        t = rotation[3]
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        rotation_m = np.mat([
            [cos_t + x**2 * (1 - cos_t), x * y * (1 - cos_t) - z * sin_t, x * z * (1 - cos_t) + y * sin_t, 0],
            [y * x * (1 - cos_t) + z * sin_t, cos_t + y**2 * (1 - cos_t), y * z * (1 - cos_t) - x * sin_t, 0],
            [z * x * (1 - cos_t) - y * sin_t, z * y * (1 - cos_t) + x * sin_t, cos_t + z**2 * (1 - cos_t), 0],
            [0, 0, 0, 1]
        ])
        object_to_world_m = translation_m  @ rotation_m @ scale_m
        GL.transform_stack.append(object_to_world_m)

        GL.normal_transform_stack.append(rotation_m)


    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO. # Referência à variável global
        if len(GL.transform_stack)>0:
            GL.transform_stack.pop()  # Modificação da lista global

        if len(GL.normal_transform_stack)>0:
            GL.normal_transform_stack.pop()


    @staticmethod
    def triangleStripSet(point, stripCount, colors,vertex_colors=None):
        """Função usada para renderizar TriangleStripSet."""
        vertices = []                      
        for i in range(0,len(point)-6,3): #
            for u in range(0,9): # appending each vertex, 3 vertices
                vertices.append(point[i+u])
        
        GL.triangleSet(vertices,colors,vertex_colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors,vertex_colors = None,colorIndex = None,texCoord = None,texCoordIndex = None,normals=None):
        """Função usada para renderizar IndexedTriangleStripSet."""


        def appendVertices(points, vertices, idx):
            coord = idx * 3
            for u in range(3): 
                vertices.append(points[coord + u])

        def appendColors(vertex_colors, indexed_vertex_colors, idx):
            coord = idx * 3
            for u in range(3): 
                indexed_vertex_colors.append(vertex_colors[coord + u])

        def appendTex(vertex_tex_coord, indexed_vertex_tex_coord, idx):
            coord = idx * 2
            for u in range(2): 
                indexed_vertex_tex_coord.append(vertex_tex_coord[coord + u])

        vertices = []
        indexed_vertex_colors = []
        indexed_vertex_tex_coord = []
        indexed_normals = []
        
        
        
        i = 0
        while i < len(index) - 2:
            if index[i] == -1 or index[i + 1] == -1 or index[i + 2] == -1:
                i += 1 
                continue

            
            appendVertices(point, vertices, index[i])     # Vertex 1
            appendVertices(point, vertices, index[i + 1]) # Vertex 2
            appendVertices(point, vertices, index[i + 2]) # Vertex 3

            if normals is not None:
                appendVertices(normals,indexed_normals,index[i])
                appendVertices(normals,indexed_normals,index[i+1])
                appendVertices(normals,indexed_normals,index[i+2])
                


            if GL.colorPerVertex:
                appendColors(vertex_colors, indexed_vertex_colors, colorIndex[i])
                appendColors(vertex_colors, indexed_vertex_colors, colorIndex[i + 1])
                appendColors(vertex_colors, indexed_vertex_colors, colorIndex[i + 2])

            if GL.image is not None:
                appendTex(texCoord, indexed_vertex_tex_coord, texCoordIndex[i]) # TexCoord 1
                appendTex(texCoord, indexed_vertex_tex_coord, texCoordIndex[i + 1]) # TexCoord 2
                appendTex(texCoord, indexed_vertex_tex_coord, texCoordIndex[i + 2]) # TexCoord 3

            i += 1

        
        if normals is not None:
            normals = indexed_normals
        

        GL.triangleSet(vertices, colors, indexed_vertex_colors,texture_values=indexed_vertex_tex_coord,normals=normals)


    @staticmethod
    def indexedFaceSet(
        coord,
        coordIndex,
        colorPerVertex,
        color,
        colorIndex,
        texCoord,
        texCoordIndex,
        colors,
        current_texture,
        normals=None
    ):
        """Função usada para renderizar IndexedFaceSet com cores e texturas."""


        def splitFaces(indices):
            faces = [] 
            vertices = [] 
            for i in indices:
                if i == -1:
                    if vertices:
                        faces.append(vertices)
                    vertices = []
                else:
                    vertices.append(i)
            return faces

        def generateStrips(faces):
            strips = []
            for face in faces:
                if len(face) < 3:
                    continue 
                
                for i in range(1, len(face) - 1):
                    strip = [face[0], face[i], face[i + 1], -1]
                    strips.extend(strip)

            return strips
        

        if current_texture:
            GL.image = gpu.GPU.load_texture(current_texture[0])

        
        GL.colorPerVertex = colorPerVertex
        
        if len(colorIndex) == 0:
            GL.colorPerVertex = False
        
        vertex_colors = color
        faces = splitFaces(coordIndex)
        stripIndices = generateStrips(faces)


        texFaces = splitFaces(texCoordIndex)
        texStripIndices = generateStrips(texFaces)
     
        if normals is not None:
         
            # generating normal coordinates per vertex based on each face
            normals = helper.generateNormals(coord,stripIndices,GL.flat)

        

        GL.indexedTriangleStripSet(coord, stripIndices, colors,vertex_colors, colorIndex,texCoord,texStripIndices,normals)







    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.
        vertices = [
            (-1, 1, -1),
            (-1,  1, 1),
            (1,  1, 1),
            (1, 1, -1),
            (-1, -1,  -1),
            (-1,  -1,  1),
            (1,  -1,  1),
            (1, -1,  -1)
        ]

        faces = [
            [0, 1, 2, 3],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
            [4, 7, 6, 5]
        ]


        size = [size[0] / 2, size[1] / 2, size[2] / 2]
        scaled_vertices = [[v[0] * size[0], v[1] * size[1], v[2] * size[2]] for v in vertices]
    

        out_vertices = []
        for vertex in scaled_vertices:
            out_vertices.extend(vertex)

        indices = []

        for face in faces:
            indices.extend(face)
            indices.append(-1)
            


        rotation_matrix_y = np.array([
            [np.cos(np.pi/2), 0, np.sin(np.pi/2)],
            [0, 1, 0],
            [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]
        ])
        for light in GL.directionalLights:
            light['direction'] = (rotation_matrix_y @ np.array(light['direction'])).astype(np.int64)

        GL.indexedFaceSet(out_vertices, indices, False, [], [], [], [], colors, [])

 
        

        

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.
        # Número de divisões de latitude e longitude
        sectorCount = 64
        stackCount = 32
        points = helper.generateSphereVertices(radius*0.9, sectorCount, stackCount)
        coordIndex = helper.generateMeshIndices(sectorCount, stackCount)
        center = np.array([0.0,0.0,0.0])
        normals = helper.generateSphereNormals(center,points)
        GL.indexedFaceSet(points, coordIndex, False, [], [], [], [], colors, [],normals)


    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

        apex = [0, height/2, 0]
        center = [0, -height/2, 0]

        # Base
        sectorCount = 30
        base = helper.generateCircleVertices(center,bottomRadius, 40)
        
        vertices = []
        vertices.extend(apex)
        vertices.extend(base)

        # Faces
        indices = []

        for i in range(1, len(vertices)//3 - 1):
            indices.extend([0,i,i+1,-1])

        normals = helper.generateNormals(vertices,indices)
        
        GL.indexedFaceSet(vertices, indices, False, [], [], [], [], colors, [], normals=normals)

    @staticmethod
    def cylinder(radius, height, colors):
        """Function used to render cylinders."""
        topCenter = [0, height / 2, 0]
        bottomCenter = [0, -height / 2, 0]
        sectorCount = 80  
   
        base = helper.generateCircleVertices(bottomCenter, radius, sectorCount)
        top = helper.generateCircleVertices(topCenter, radius, sectorCount)

        vertices = base + top
        vertices.extend(bottomCenter) 
        vertices.extend(topCenter)     

        bottomCenterIndex = sectorCount * 2
        topCenterIndex = sectorCount * 2 + 1

        faces = []


        for i in range(sectorCount):
            next_i = (i + 1) % sectorCount
            faces.extend([i, next_i, sectorCount + next_i, sectorCount + i, -1])

        # Bottom cap faces
        for i in range(sectorCount):
            next_i = (i + 1) % sectorCount
            faces.extend([i, next_i, bottomCenterIndex, -1])

        # Top cap faces
        for i in range(sectorCount):
            next_i = (i + 1) % sectorCount
            faces.extend([sectorCount + next_i, sectorCount + i, topCenterIndex, -1])

        normals = helper.generateNormals(vertices, faces)

        GL.indexedFaceSet(vertices, faces, False, [], [], [], [], colors, [], normals=normals)



    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "NavigationInfo : headlight = {0}".format(headlight)
        )  # imprime no terminal

        # dlight =
        if headlight:
            GL.headlight = True
            GL.directionalLight(0.0, [1, 1, 1], 1.0, [0, 0, -1])

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color))  # imprime no terminal
        print(
            "DirectionalLight : intensity = {0}".format(intensity)
        )  # imprime no terminal
        print(
            "DirectionalLight : direction = {0}".format(direction)
        )  # imprime no terminal


        GL.directionalLights.append({"ambientIntensity": ambientIntensity, "color": color, "intensity": intensity, "direction": direction})

        

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color))  # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity))  # imprime no terminal
        print("PointLight : location = {0}".format(location))  # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color))  # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "TimeSensor : cycleInterval = {0}".format(cycleInterval)
        )  # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        GL.super_buffer = np.zeros((GL.width*2, GL.height*2, 3), dtype=np.uint8)
        GL.z_buffer =  - np.inf * np.ones((GL.width*2, GL.height*2)) 
        epoch = (
            time.time()
        )  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval



        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.

        def interpolatePosition(h0,h1,h2,h3,t): # h0 = p0, h1 = p1, h2 = p'0, h3 = p'1
            h0t = 2*t**3 - 3*t**2 + 1
            h1t = -2*t**3 + 3*t**2
            h2t = t**3 - 2*t**2 + t
            h3t = t**3 - t**2

            return h0*h0t + h1*h1t + h2*h2t + h3*h3t
        
        def getTangent(p0,p1):
            return (p1-p0)/2
        

        xs = [keyValue[i] for i in range(0, len(keyValue), 3)]
        ys = [keyValue[i] for i in range(1, len(keyValue), 3)]
        zs = [keyValue[i] for i in range(2, len(keyValue), 3)]

        t = set_fraction*(len(xs)-1)

        int_t = int(t)
        fract_t = t % 1

        h0 = [xs[int_t],ys[int_t],zs[int_t]]
        h1 = [xs[int_t+1],ys[int_t+1],zs[int_t+1]]

        h2 = [0,0,0]
        h3 = [0,0,0]

        if int_t > 0 and len(xs)-1 > int_t+1:
            h2x = getTangent(xs[int_t-1],xs[int_t+1])
            h2y = getTangent(ys[int_t-1],ys[int_t+1])
            h2z = getTangent(zs[int_t-1],zs[int_t+1])
            h2 = [h2x,h2y,h2z]
        
        if int_t+2 < len(xs):
            h3x = getTangent(xs[int_t],xs[int_t+2])
            h3y = getTangent(ys[int_t],ys[int_t+2])
            h3z = getTangent(zs[int_t],zs[int_t+2])
            h3 = [h3x,h3y,h3z]

        r_x = interpolatePosition(h0[0],h1[0],h2[0],h3[0],fract_t)
        r_y = interpolatePosition(h0[1],h1[1],h2[1],h3[1],fract_t)
        r_z = interpolatePosition(h0[2],h1[2],h2[2],h3[2],fract_t)

        value_changed = [r_x,r_y,r_z]



        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))


        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        rotations = [keyValue[i:i+4] for i in range(0, len(keyValue), 4)]
        num_keys = len(rotations)

        t = set_fraction * (num_keys - 1)
        int_t = int(t)
        if int_t >= num_keys - 1:
            int_t = num_keys - 2
        fract_t = t - int_t

        r0 = rotations[int_t]
        r1 = rotations[int_t + 1]

        def axis_angle_to_quaternion(axis_angle):
            x, y, z, angle = axis_angle
            norm = math.sqrt(x*x + y*y + z*z)
            if norm == 0:
                return [1, 0, 0, 0]
            x /= norm
            y /= norm
            z /= norm
            half_angle = angle / 2.0
            sin_half_angle = math.sin(half_angle)
            qx = x * sin_half_angle
            qy = y * sin_half_angle
            qz = z * sin_half_angle
            qw = math.cos(half_angle)
            return [qw, qx, qy, qz]

        def slerp(q0, q1, t):
            dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]
            if dot < 0.0:
                q1 = [-q1[0], -q1[1], -q1[2], -q1[3]]
                dot = -dot
            if dot > 0.9995:
                result = [q0[j] + t*(q1[j]-q0[j]) for j in range(4)]
                length = math.sqrt(sum([comp*comp for comp in result]))
                return [comp / length for comp in result]
            
            theta_0 = math.acos(dot)
            sin_theta_0 = math.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = math.sin(theta_t)
            s0 = math.cos(theta_t) - dot * sin_theta_t / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            return [s0 * q0[j] + s1 * q1[j] for j in range(4)]

        def quaternion_to_axis_angle(q):
            qw, qx, qy, qz = q
            angle = 2 * math.acos(qw)
            s = math.sqrt(1 - qw*qw)
            if s < 0.001:
                x = qx
                y = qy
                z = qz
            else:
                x = qx / s
                y = qy / s
                z = qz / s
            return [x, y, z, angle]

        q0 = axis_angle_to_quaternion(r0)
        q1 = axis_angle_to_quaternion(r1)
        qt = slerp(q0, q1, fract_t)
        value_changed = quaternion_to_axis_angle(qt)

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
