"""
Steps for SIFT
1.- Multi-Scale extrema detection
    Tomar la imagen y pasarla por diferentes octavas
    (filtros gaussianos) y sacar la diferencia entre 
    estos filtros. No. impar de filtros gaussianos para 
    tener no. pares de laplacianos (se hace a dif escalas)
    DoG (Difference of Gaussian). Seleccionar 26 vecinos

2.- Keypoint localization
    2do order Taylor series apporximation of DoG scale space
    Take teh derivative and solve extrema
    (ver fórmulas en clase min 16:16)
3.- Orientation assignment
    Orientación es la tang inv del laplaciano (der en x y en y0
    ver fórmula en clase
    For a keypoint, L is the Gaussian-sm image with the closest scale
    Nos regresa la orientación y escala del punto clave 
    {x,y,scale,orient}
4.- Keypoint description
    Generar descriptor
    se hace generando *gradientes de imagen*
    (4 x 4 pixel per cell, 4 x 4 cells)
    Gaussian weigthing (sigma = half width)
    *SIFT descriptor*
    sumar los gradientes de cada celda
    16 cells x 8 directions = 128 dims

Generar al menos 3 funciones (ya son 5, ver canvas)
1.-Generar escalamientos y diferencias Gaussianas
2.-Encontrar extremos (x,y) en las diferentes escalas
   esta regresa {x,y,scale,orient}
3.-Gnerar orientación del punto característico

Gnerar descriptor de SIFT con las funciones pasadas
los keypoints deben de notarse las escalas

No utilizar funciones de opencv que no hayamos visto

"""