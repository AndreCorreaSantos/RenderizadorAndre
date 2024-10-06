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


class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    perspective_matrix = np.mat([])
    look_at = np.mat([])
    transform_stack = []
    vertex_colors = []
    vertex_tex_coord = []
    texture = []
    colorPerVertex = False
    

    width = 800  # largura da tela
    height = 600  # altura da tela
    near = 0.01  # plano de corte próximo
    far = 1000  # plano de corte distante

    super_buffer = None

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.super_buffer = np.zeros((GL.width*2, GL.height*2, 3), dtype=np.uint8)


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

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""

        def insideTri(tri: list[float], x: float, y: float) -> bool:
            def line_eq(x0, y0, x1, y1, px, py):
                return (y1 - y0) * px - (x1 - x0) * py + y0 * (x1 - x0) - x0 * (y1 - y0)

            # Extract x and y coordinates correctly
            p1 = [tri[0], tri[1]]      # x1, y1
            p2 = [tri[3], tri[4]]      # x2, y2
            p3 = [tri[6], tri[7]]      # x3, y3

            L1 = line_eq(p1[0], p1[1], p2[0], p2[1], x, y)
            L2 = line_eq(p2[0], p2[1], p3[0], p3[1], x, y)
            L3 = line_eq(p3[0], p3[1], p1[0], p1[1], x, y)

            return (L1 > 0 and L2 > 0 and L3 > 0) or (L1 < 0 and L2 < 0 and L3 < 0)
        
        def compute_barycentric_coordinates(tri, x, y):
            x1, y1 = tri[0], tri[1]
            x2, y2 = tri[3], tri[4]
            x3, y3 = tri[6], tri[7]
            denominator = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
            if denominator == 0:
                return 0, 0, 0  # Avoid division by zero
            alpha = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator
            beta = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator
            gamma = 1 - alpha - beta
            return alpha, beta, gamma

        if len(vertices) < 9:
            print("ERROR NO TRIANGLES SENT")
            return

        if GL.colorPerVertex:
            color_array = np.array(colors) * 255
        else:
            color = np.array(colors["emissiveColor"]) * 255

        for i in range(0, len(vertices), 9):
            tri = vertices[i : i + 9]
            if len(tri) != 9:
                return

            xs = [tri[j] for j in range(0, len(tri), 3)]  # x coordinates
            ys = [tri[j] for j in range(1, len(tri), 3)]  # y coordinates
            zs = [tri[j] for j in range(2, len(tri), 3)]  # z coordinates

            # Bounding Box
            box = [int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))]

            # Scale only x and y for super sampling
            super_box = [2 * v for v in box]
            super_tri = []
            for j in range(0, len(tri), 3):
                super_tri.extend([2 * tri[j], 2 * tri[j + 1], tri[j + 2]])  # Scale x and y, keep z

            if GL.colorPerVertex:
                # Extract per-triangle colors
                tri_colors = color_array[i : i + 9]
                if len(tri_colors) != 9:
                    return
                c1 = np.array(tri_colors[0:3])
                c2 = np.array(tri_colors[3:6])
                c3 = np.array(tri_colors[6:9])
                # Corresponding z-values at each vertex
                z1, z2, z3 = tri[2], tri[5], tri[8]

            # Iterating over the bounding box
            for x in range(super_box[0], super_box[1] + 1):
                for y in range(super_box[2], super_box[3] + 1):

                    if insideTri(super_tri, x + 0.5, y + 0.5):
                        if (0 <= x < GL.width * 2) and (0 <= y < GL.height * 2):
                            if GL.colorPerVertex:
                                # Compute barycentric coordinates
                                alpha, beta, gamma = compute_barycentric_coordinates(super_tri, x + 0.5, y + 0.5)
                                # Ensure weights are within [0,1]
                                alpha = max(0, min(alpha, 1))
                                beta = max(0, min(beta, 1))
                                gamma = max(0, min(gamma, 1))

                                # Perspective-correct interpolation
                                # Compute 1/z for each vertex
                                one_over_z1 = 1.0 / z1 if z1 != 0 else 0.0
                                one_over_z2 = 1.0 / z2 if z2 != 0 else 0.0
                                one_over_z3 = 1.0 / z3 if z3 != 0 else 0.0

                                # Interpolate 1/z at the current pixel
                                one_over_z = alpha * one_over_z1 + beta * one_over_z2 + gamma * one_over_z3
                                if one_over_z == 0:
                                    continue  # Avoid division by zero

                                # Interpolate color components with perspective correction
                                r = (alpha * c1[0] * one_over_z1 + beta * c2[0] * one_over_z2 + gamma * c3[0] * one_over_z3) / one_over_z
                                g = (alpha * c1[1] * one_over_z1 + beta * c2[1] * one_over_z2 + gamma * c3[1] * one_over_z3) / one_over_z
                                b = (alpha * c1[2] * one_over_z1 + beta * c2[2] * one_over_z2 + gamma * c3[2] * one_over_z3) / one_over_z

                                pixel_color = np.array([r, g, b])
                                pixel_color = np.clip(pixel_color, 0, 255).astype(np.uint8)

                                GL.super_buffer[x][y] = pixel_color
                            else:
                                GL.super_buffer[x][y] = color

            # Downsample and draw pixels
            for x in range(box[0], box[1] + 1):
                for y in range(box[2], box[3] + 1):
                    c1 = GL.super_buffer[2 * x][2 * y]
                    c2 = GL.super_buffer[2 * x][2 * y + 1]
                    c3 = GL.super_buffer[2 * x + 1][2 * y]
                    c4 = GL.super_buffer[2 * x + 1][2 * y + 1]
                    super_colors = np.array([c1, c2, c3, c4]).mean(axis=0).astype(np.uint8)
                    if (0 <= x < GL.width) and (0 <= y < GL.height):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, super_colors)


    @staticmethod
    def triangleSet(point, colors):
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
                print("z")
                print(z)

                p = GL.perspective_matrix @ GL.look_at @ transform_mat_res @ p
                # print("shape")
                # print(p.shape)
                # Step 2: Multiply the result by the perspective matrix
                # p = GL.perspective_matrix @ p

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

        vertices = transform_points(point, min(xs), min(ys), min(zs), max(zs))

        # Call triangleSet2D with the transformed 2D vertices

        GL.triangleSet2D(vertices, colors)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.

        #LÓGICA TRANSLAÇÕES E ROTAÇÕES LOOK AT
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


    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        vertices = []                      
        for i in range(0,len(point)-6,3): #
            for u in range(0,9): # appending each vertex, 3 vertices
                vertices.append(point[i+u])
        
        GL.triangleSet(vertices,colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors,colorIndex = None):
        """Função usada para renderizar IndexedTriangleStripSet."""


        def appendVertices(points, vertices, idx):
            coord = idx * 3
            for u in range(3): 
                vertices.append(points[coord + u])

        def appendColors(colors, vertex_colors, idx):
            coord = idx * 3
            for u in range(3): 
                vertex_colors.append(colors[coord + u])


        vertices = []
        vertex_colors = []
        i = 0
        while i < len(index) - 2:
            if index[i] == -1 or index[i + 1] == -1 or index[i + 2] == -1:
                i += 1 # pulando indices -1
                continue

            
            appendVertices(point, vertices, index[i])     # Vertex 1
            appendVertices(point, vertices, index[i + 1]) # Vertex 2
            appendVertices(point, vertices, index[i + 2]) # Vertex 3

            if GL.colorPerVertex:
                appendColors(colors, vertex_colors, colorIndex[i])
                appendColors(colors, vertex_colors, colorIndex[i + 1])
                appendColors(colors, vertex_colors, colorIndex[i + 2])

            i += 1 
            



            GL.triangleSet(vertices, vertex_colors)


    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size))  # imprime no terminal pontos
        print("Box : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

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
    ):
        """Função usada para renderizar IndexedFaceSet com cores e texturas."""

        def splitColors(colors):
            return [int(colors[i:i + 3]*255) for i in range(0, len(colors), 3)]


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
        
        # vert_colors = []

        # print("colors: ")
        # print(color)
        if colorPerVertex:
            colors = color
    
        # print("vert_colors: ")
        # print(vert_colors)

        faces = splitFaces(coordIndex)
        stripIndices = generateStrips(faces)
        GL.colorPerVertex = colorPerVertex

            
        GL.indexedTriangleStripSet(coord, stripIndices, colors, colorIndex)







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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print(
            "Sphere : radius = {0}".format(radius)
        )  # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors))  # imprime no terminal as cores

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
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print(
            "SplinePositionInterpolator : key = {0}".format(key)
        )  # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]

        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key))  # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
