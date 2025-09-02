# Motion Planning

This project explores **robotic motion planning** using Python, computer vision, and computational geometry techniques. It demonstrates how to represent environments with obstacles, define start/goal points, and generate paths through Voronoi diagrams.  

## Features
- **Environment Setup**:  
  - Loads a map image and resizes it to a configurable grid size.  
  - Defines rectangular obstacles (`A`–`F`) within the environment.  
  - Specifies start (`S`) and goal (`G`) positions.  

- **Geometric Processing**:  
  - Generates obstacle polygons using `numpy`.  
  - Collects all obstacle and goal points for path computation.  
  - Employs **Voronoi diagrams** (via `scipy.spatial.Voronoi`) to construct roadmap structures.  

- **Pathfinding Algorithms**:  
  - Implements **Dijkstra’s Algorithm** to compute the shortest path across the Voronoi roadmap.  
  - Implements **A*** for efficient pathfinding with heuristic guidance.  
  - Ensures obstacle avoidance by leveraging roadmap connectivity.  

- **Visualization**:  
  - Displays the map and planning process with **OpenCV** and **Matplotlib**.  
  - Overlays obstacles, start/goal points, Voronoi edges, and computed paths.  

## Technologies Used
- **Python 3.x**  
- **OpenCV** (`cv2`) for image processing and visualization  
- **NumPy** for geometric computations  
- **SciPy** for Voronoi diagram generation  
- **Matplotlib** for plotting  

## Project Structure
- `Motion Planing.ipynb` — Jupyter Notebook containing the code and demonstrations  
- **Sections include**:  
  - Imports  
  - Load Map  
  - Define Obstacles and Goals  
  - Generate Voronoi Diagram  
  - Pathfinding with Dijkstra’s Algorithm and A*  
  - Visualization and Results  

## How to Run
1. Clone or download the repository.  
2. Install dependencies:  
   ```bash
   pip install opencv-python numpy scipy matplotlib
   ```  
3. Open the notebook:  
   ```bash
   jupyter notebook "Motion Planing.ipynb"
   ```  
4. Run all cells to visualize the motion planning and pathfinding process.  

## Author
**Gal Schechter**
