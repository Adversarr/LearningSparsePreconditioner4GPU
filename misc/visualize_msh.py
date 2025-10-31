import argparse
import meshio
import pyvista as pv
import numpy as np

def load_and_visualize_msh(file_path):
    # 用meshio加载.msh文件
    mesh = meshio.read(file_path)
    
    if mesh.points is not None and len(mesh.points) > 0:
        # 准备pyvista的cells和cell_type
        cells = []
        cell_types = []
        
        for cell_block in mesh.cells:
            # meshio的cell类型映射到pyvista的VTK类型
            if cell_block.type == "triangle":
                vtk_type = pv.CellType.TRIANGLE
                n_nodes = 3
            elif cell_block.type == "quad":
                vtk_type = pv.CellType.QUAD
                n_nodes = 4
            elif cell_block.type == "tetra":
                vtk_type = pv.CellType.TETRA
                n_nodes = 4
            elif cell_block.type == "hexahedron":
                vtk_type = pv.CellType.HEXAHEDRON
                n_nodes = 8
            else:
                print(f"Warning: Unsupported cell type {cell_block.type}, skipping.")
                continue
            
            # 将单元数据添加到cells数组
            cells.append(np.hstack([np.full((len(cell_block.data), 1), n_nodes), cell_block.data]))
            cell_types.append(np.full(len(cell_block.data), vtk_type))
        
        if not cells:
            print("Error: No supported cell types found.")
            return
        
        # 转换为numpy数组
        cells = np.vstack(cells).astype(np.int64)
        cell_types = np.hstack(cell_types).astype(np.uint8)
        points = mesh.points.astype(np.float64)  # 确保是浮点型
        
        # 生成UnstructuredGrid
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        print(f"Loaded {len(points)} points and {len(cells)} cells from {file_path}.")
        # 可视化网格
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, color="lightblue")
        plotter.show()
    else:
        print("Error: No points found in the mesh.")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Visualize a .msh file using pyvista and meshio")
    parser.add_argument("msh_file", type=str, help="Path to the .msh file")
    args = parser.parse_args()

    # 加载并可视化.msh文件
    load_and_visualize_msh(args.msh_file)

if __name__ == "__main__":
    main()