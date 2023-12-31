# Alzheimer-Web for 3D Visualization of Alzheimer's disease Signatures

The web application developed provides an intuitive user interface for exploring and analyzing Alzheimer’s saliency in the human brain. It utilizes Plotly Dash, a Python framework designed for creating web applications with interactive visualizations. The application offers a menu where users can choose different parameters and options to customize their analysis. One of the key features of the application is the ’choose mask for overlay’ dropdown menu. It provides users with a selection of specific regions of interest in the brain. By choosing a region of interest, the application acts as a filter, displaying only the saliency pertaining to those selected regions. This filtering mechanism allows for focused visualization. To further enhance the visual representation, the application includes a threshold slider. This slider enables users to adjust the saliency values and control the visibility of saliency outside of a specified range. By setting the threshold, users can fine-tune the visualization to highlight the most relevant areas and exclude irrelevant saliency information. When selecting a region of interest, a line graph is displayed showing the sum of the saliency scores within each slice of the saliency volume.


![axial_hippo_slices_scores (1)](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/3c6132ec-1bb1-4703-b98c-3eef355c7913)

The x-axis of the line graph represents the slices, ranging from 0 to 65 slices, while the y-axis represents the sum of saliency values within each slice




![Web_app](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/689596c9-c310-4ea5-ba0b-fe05ec42cfa9)

## 2D Plane Saliency Visualization

The proposed approach involves visualizing saliency maps overlaid on reconstructed brain surfaces using the Marching cubes algorithm. A collection of preprocessed 2D saliency slices are used to create 2D saliency planes. The final color of the saliency on the 2D plane is calculated using a combination of color mapping and lighting calculations. The saliency values, which range from 0 to 1, are mapped to a Green Yellow Red (GYR) colorscale. The visible brain surface has a thickness defined as 10 MRI scans per frame.


![2D saliency planes overlaid on the surface brain](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/10eb4da6-fae6-415e-858a-205e83f635b0)

An alternative approach to saliency visualization as a 2D plane involves the construction of the brain mesh as a static entity, with the brain surface remaining visible throughout all frames. A slider component is incorporated to facilitate the movement of a 2D saliency plane withinthe brain volume. The brain mesh is represented with a low opacity, allowing the saliency visualization to be visible beneath it.


![Saliency slicing](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/d2a6336c-b279-47af-a764-49db29bf4d91)

## Saliency Visualization Using Mesh Coloring

The saliency information is represented now by painting the 3D surface mesh of the brain. This method involves identifying the coordinates of the saliency and finding the nearest mesh vertices to these coordinates. A k-d tree data structure is utilized to efficiently search for the nearest mesh vertices . By mapping the saliency values to a colorscale, the vertices of the mesh are colored based on the corresponding saliency values.


![meshpainting](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/d18b9d45-ee0f-4f5d-8d72-37c5e08f8bfa)

## Visualization of Relevance Maps Using Indirect Volume Rendering

The saliency is converted into a set of polygons that represent a level surface or iso-surface. The idea is to create multiple iso-surfaces of the saliency with different iso levels. The rendering of these iso-surfaces involves iterating through the different iso-levels and drawing the surfaces accordingly. Each iso-level represents a different magnitude of saliency. Additionally, opacity adjustments can also be applied to enhance the visibility of different iso-level surfaces representing the saliency. By assigning varying opacity values to different iso levels, we can control the transparency of the surfaces. For example, a low opacity value can be assigned to surfaces corresponding to low iso levels, while a high opacity value can be assigned to surfaces representing larger iso levels.


![Saliency Indirect Volume rendering](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/01e59f4d-07df-4c3f-8d95-a0e4d0f16c08)

## Visualization of Saliency inside the temporal lobe


![temporal mesh painting](https://github.com/Tajini-tech/Alzheimer-Web/assets/143637408/c84f92c8-7eae-4ffe-8093-29df664e0953)
