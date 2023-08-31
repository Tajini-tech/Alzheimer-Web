# Alzheimer-Web
Web application for 3D Visualization of Alzheimer's disease Signatures

##2D Plane Saliency Visualization:
The proposed approach involves visualizing saliency maps overlaid on reconstructed brain surfaces using the Marching cubes algorithm. A collection of preprocessed 2D saliency slices are used to create 2D saliency planes. The final color of the saliency on the 2D plane is calculated using a combination of color mapping and lighting calculations. The saliency values, which range from 0 to 1, are mapped to a Green Yellow Red (GYR) colorscale. The visible brain surface has a thickness defined as 10 MRI scans per frame
![2D saliency planes overlaid on the surface brain](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/44a11b64-6f2e-4bb2-adf2-eab98db293e1)

An alternative approach to saliency visualization as a 2D plane involves the construction of the brain mesh as a static entity, with the brain surface remaining visible throughout all frames. A slider component is incorporated to facilitate the movement of a 2D saliency plane withinthe brain volume. The brain mesh is represented with a low opacity, allowing the saliency visualization to be visible beneath it
![Saliency slicing](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/75596ffa-f48d-469f-a2bb-ca0607eb0308)

##Saliency Visualization Using Mesh Coloring:
The saliency information is represented now by painting the 3D surface mesh of the brain. This method involves identifying the coordinates of the saliency and finding the nearest mesh vertices to these coordinates. A k-d tree data structure is utilized to efficiently search for the nearest mesh vertices . By mapping the saliency values to a colorscale, the vertices of the mesh are colored based on the corresponding saliency values.

![meshpainting](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/9873b833-e7a4-41e5-85ac-50f24720c458)
##Visualization of Relevance Maps Using Indirect Volume Rendering:
The saliency is then converted into a set of polygons that represent a level surface or iso-surface. The idea is to create multiple iso- surfaces of the saliency with different iso levels. The rendering of these iso-surfaces involves iterating through the different iso-levels and drawing the surfaces accordingly. Each iso-level represents a different magnitude of saliency. Additionally, opacity adjustments can also be applied to enhance the visibility of different iso-level surfaces representing the saliency. By assigning varying opacity values to different iso levels, we can control the transparency of the surfaces. For example, a low opacity value can be assigned to surfaces corresponding to low iso levels, while a high opacity value can be assigned to surfaces representing larger iso levels.
![Saliency Indirect Volume rendering](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/e25eed4d-1fa6-46f4-bf06-9492bc45e54a)
##Visualization of Saliency inside the temporal lobe
![temporal mesh painting](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/a72bc520-5731-46ad-a1dd-e1b0c5297234)
