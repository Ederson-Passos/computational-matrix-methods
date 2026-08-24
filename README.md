# Computational Matrix Methods

Welcome to the repository for **Computational Matrix Methods**, developed for the Applied Mathematics Master's program. This repository houses computational experiments, theoretical validations, and visual analyses of advanced matrix algebra techniques, focusing on rigor, reproducibility, and professional academic standards.

## 📂 Repository Structure

The project is organized into modular experiments and units. Each unit contains its own exploratory notebooks, source code, and generated results (high-resolution figures and LaTeX fragments).

- [`experiments/`](experiments/) - The core directory containing all course units.

---

## 📚 Course Units

### [Unit 1: Matrix Analysis & Singular Value Decomposition (SVD)](experiments/unit_01_matrix_analysis/)

This unit explores the mathematical properties and practical applications of the Singular Value Decomposition (SVD) on continuous 2D functions sampled over discrete grids. 

We generate a matrix $A$ from the interference pattern $A(X,Y) = \sin(aX) + \cos(bY)$ and use SVD to compute highly compressed, low-rank approximations ($A_k$), analyzing the spectral energy distribution and the Frobenius error norm decay ($\|A - A_k\|_F$). A key highlight of this unit is the practical demonstration of the **Rank-2 phenomenon**, where mathematical exactness is achieved at $k=2$.

**🔗 Quick Links for Unit 1:**
- 📄 **[Course Material & Assignment](experiments/unit_01_matrix_analysis/course_material/TC1%20(6).pdf)**: The original theoretical and computational requirements.
- 💻 **[Computational Notebook](experiments/unit_01_matrix_analysis/notebooks/tc1_resolution.ipynb)**: The main Jupyter Notebook containing the data generation, SVD algorithms, and analysis.
- 📈 **[Generated Figures](experiments/unit_01_matrix_analysis/results/figures/)**: High-resolution (300 DPI) plots following Edward Tufte's principles of minimal ink. Includes the singular value spectrum, error decay, and visual comparisons.
- 📝 **[LaTeX Tables](experiments/unit_01_matrix_analysis/results/tex_tables/)**: Auto-generated LaTeX files containing the singular values and error tables, structured for direct import into Overleaf.

---
*Developed for the Master's Program in Applied Mathematics.*
