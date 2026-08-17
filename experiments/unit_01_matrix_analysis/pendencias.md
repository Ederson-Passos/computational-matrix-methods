Trabalho Computacional 1:
1. Normas Matriciais (Matrizes como Operadores Lineares):
* Normas Induzidas ($p$-normas): Compreender como a norma de um vetor induz a norma de uma matriz. A norma $\vert{}\vert{}A\vert{}\vert{}_2$ (norma espectral) não é calculada apenas somando os elementos, mas é definida pelo maior valor singular de $A$.
* Norma de Frobenius ($\vert{}\vert{}A\vert{}\vert{}_F$): A raiz quadrada da soma dos quadrados de todos os elementos. Fisicamente, mede a "energia" total da matriz. É computacionalmente mais barata que a norma espectral, e você precisará entender a desigualdade que as relaciona: $\vert{}\vert{}A\vert{}\vert{}_2 \leq \vert{}\vert{}A\vert{}\vert{}_F \leq \sqrt{r} \vert{}\vert{}A\vert{}\vert{}_2$, onde $r$ é o posto da matriz.

2. Decomposição em Valores Singulares (SVD - Singular Value Decomposition):
* O teorema fundamental afirma que qualquer matriz real $A \in \mathbb{R}^{m \times n}$ pode ser fatorada como $A = U \Sigma V^T$, onde $U$ e $V$ são matrizes ortogonais e $\Sigma$ é uma matriz diagonal contendo os valores singulares $\sigma_i$.

* O Teorema de Eckart-Young-Mirsky: Este é o coração da Questão 1 (item e) e de toda a Questão 2. Ele prova que a melhor aproximação de posto $k$ para uma matriz $A$ (minimizando tanto a norma $L_2$ quanto a de Frobenius) é obtida truncando a SVD para os $k$ maiores valores singulares. Na sua área de Machine Learning, isso é a base teórica para compressão de imagens, redução de dimensionalidade (PCA) e aproximação de operadores estruturados.

Questão 1:
1. Lax, Peter D. - Linear Algebra and Its Applications:Consulta:
* No livro de Peter Lax (arquivo lax.pdf), consulte o Capítulo intitulado "Matrix Norms" (geralmente Capítulo 11 ou seções correlatas sobre operadores limitados). Lax introduz as normas de matrizes sob a ótica clássica da análise funcional, excelente para fundamentar o rigor da definição de normas de operadores induzidos e a submultiplicatividade ($\vert{}\vert{}AB\vert{}\vert{} \leq \vert{}\vert{}A\vert{}\vert{} \cdot \vert{}\vert{}B\vert{}\vert{}$).

2. Golub, Gene H. & Van Loan, Charles F. - Matrix Computations:Consulta:
* Consulte o Capítulo 2 ("Matrix Analysis"), especificamente a Seção 2.3 ("Norms"). Esta é a bíblia da Álgebra Linear Numérica. Golub e Van Loan discutem detalhadamente os aspectos computacionais e as relações de equivalência entre as $p$-normas e a norma de Frobenius, fornecendo a base exata para a justificativa de algoritmos numéricos.

