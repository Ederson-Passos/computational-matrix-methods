# Relatório de Validação e Auditoria — `TC4_apresentacao.html`

**Disciplina:** Métodos Matriciais Computacionais (PPGMAP / UFRGS)  
**Docente:** Prof.ª Dr.ª Cibele Aparecida Ladeia  
**Autor:** Éderson Passos Silva (Matrícula: `318375`)  
**Arquivo Principal Entregue:** `TC4_apresentacao.html`

---

## 1. Materiais de Referência Examinados

1. **`docs/TC4.pdf` (Enunciado Oficial):**
   - Verificados os requisitos acadêmicos, identificação institucional (Universidade Federal do Rio Grande do Sul — PPGMAP), nome da docente (Prof.ª Dr.ª Cibele Aparecida Ladeia) e definição paramétrica das matrizes $A_1 \in \mathbb{R}^{4 \times 4}$ (Problema 1: itens a, b, c) e $A_2 \in \mathbb{R}^{5 \times 5}$ (Problema 2: itens a, b) a partir dos dígitos $m = d_1 d_2 d_3 d_4 d_5 d_6$.
2. **`scripts/problem_1_2.ipynb` (Notebook Computacional):**
   - Examinadas todas as 44 células (código, fundamentação matemática e saídas numéricas/gráficas) referentes à matrícula `318375` $\implies (d_1, d_2, d_3, d_4, d_5, d_6) = (3, 1, 8, 3, 7, 5)$.
3. **`scripts/iteracoes_metodo_potencia.csv` e Relatório LaTeX (`main.tex` e `sections/*.tex`):**
   - Conferida a consistência integral das 818 iterações do Método da Potência e dos 99 pontos da tabela amostrada granulada (`tabela_potencia_granulada`).

---

## 2. Principais Decisões de Implementação

1. **Autonomia Absoluta (`file://` Offline):**
   - Todo o código HTML5 semântico, folhas de estilo CSS3, lógica de controle em JavaScript Vanilla, equações matemáticas, matrizes e gráficos vetoriais SVG foram incorporados em um único arquivo (`TC4_apresentacao.html`, $\approx 135\text{ KB}$).
   - **Zero dependências externas:** não há chamadas a CDNs, Google Fonts, MathJax externo, imagens externas ou servidores locais.
2. **Identidade Visual (LaTeX + Edward Tufte):**
   - **Tipografia LaTeX:** hierarquia serifada acadêmica (`Latin Modern Roman` / `Computer Modern Serif` / `Cambria` / `Times New Roman`, com fallback nativo universal), código e números tabulares em fonte monoespaçada (`font-variant-numeric: tabular-nums`), tabelas no padrão `\toprule`, `\midrule` e `\bottomrule` (`booktabs`) e matrizes com colchetes estruturais rigorosos (`\begin{bmatrix}`).
   - **Princípios de Edward Tufte (*The Visual Display of Quantitative Information*):** alto *data-ink ratio*, ausência completa de *chartjunk* (sem sombras decorativas, gradientes ornamentais ou 3D), *direct labeling* diretamente sobre barras e curvas, *small multiples* para comparação dos números de condição $\operatorname{cond}(\lambda_i)$ e anotações analíticas concisas no próprio espaço gráfico.
3. **Gráficos Vetoriais Reais em SVG Nativo:**
   - Os gráficos da Etapa 03 (espectro de $A_1$), Etapa 04 (fatores de amplificação $\operatorname{cond}(\lambda_i)$) e Etapa 06 (evolução de $\lambda^{(k)} \to \lambda_1$ e decaimento logarítmico do erro relativo até $10^{-8}$) são gerados vetorialmente em SVG a partir do conjunto exato de dados produzido pelo notebook (incluindo tanto a série de 99 pontos da tabela granulada quanto o histórico completo das 818 iterações).
4. **Interatividade Operacional Simples e Previsível:**
   - Navegação por teclado (`←`, `→`, `Home`, `End`, `F` para Tela Cheia, `N` para Notas do Apresentador, `Esc` para recolher painéis) e por botões discretos no cabeçalho.
   - Notas do apresentador ocultas por padrão e estruturadas nos 4 eixos exigidos (*O que explicar*, *Resultado a destacar*, *Interpretação matemática* e *Transição*), com aviso explícito de que são visíveis no projetor caso abertas.

---

## 3. Confirmação de Funcionamento Offline e Compatibilidade

- **Auditoria de rede:** verificado via inspeção automatizada por expressões regulares que o arquivo contém **0 URLs externas** (`http://` ou `https://`).
- **Protocolo `file://`:** testado para abertura direta via duplo clique em qualquer diretório local (ex.: Área de Trabalho de computador institucional sem internet).
- **Navegadores compatíveis:** Google Chrome, Microsoft Edge, Mozilla Firefox e Safari (baseados em padrões W3C HTML5/SVG/CSS3 nativos).

---

## 4. Verificações Matemáticas Realizadas

Todos os números exibidos na apresentação foram auditados diretamente contra as saídas do notebook `problem_1_2.ipynb`:

| Grandeza / Experimento | Valor no Notebook | Valor em `TC4_apresentacao.html` | Status |
| :--- | :--- | :--- | :---: |
| Matrícula $(d_1, d_2, d_3, d_4, d_5, d_6)$ | `318375` $\to (3,1,8,3,7,5)$ | `318375` $\to (3,1,8,3,7,5)$ | Conferido |
| Redução de Hessenberg ($\|U^T U - I\|_\infty$ / $\|U H U^T - A_1\|_\infty$) | $8.31 \times 10^{-18}$ / $1.78 \times 10^{-15}$ | $8.31 \times 10^{-18}$ / $1.78 \times 10^{-15}$ | Conferido |
| Iterações QR até convergência de Schur ($A_1$) | `68` ($\text{tol} = 10^{-12}$) | `68` ($\text{tol} = 10^{-12}$) | Conferido |
| Autovalores $\lambda(A_1)$ | `{26.68556904, 18.42664022, 2.88779074, 34.00000000}` | `{26.68556904, 18.42664022, 2.88779074, 34.00000000}` | Conferido |
| Ortogonalidade $\|Q^T Q - I\|_{\max}$ e Fatoração $\|Q T Q^T - A_1\|_{\max}$ | $1.89 \times 10^{-15}$ e $3.19 \times 10^{-11}$ | $1.89 \times 10^{-15}$ e $3.19 \times 10^{-11}$ | Conferido |
| Invariância do Traço ($\operatorname{tr}(A_1) = \sum \lambda_i$) | $82.000000 = 82.000000$ | $82.000000 = 82.000000$ | Conferido |
| Determinante ($34 \times \det(A_{1,3\times 3})$ vs. $\prod \lambda_i$) | $48280.00000000$ vs. $48280.00000017$ (erro rel. $3.59 \times 10^{-12}$) | $48280.00000000$ vs. $48280.00000017$ (erro rel. $3.59 \times 10^{-12}$) | Conferido |
| Desvio de normalidade $\|A_1^T A_1 - A_1 A_1^T\|_{\max}$ | `2294.00` | `2294.00` | Conferido |
| Números de condição $\operatorname{cond}(\lambda_1, \lambda_2, \lambda_3, \lambda_4)$ | `21.222548`, `11.680828`, `3.298015`, `12.642448` | `21.222548`, `11.680828`, `3.298015`, `12.642448` | Conferido |
| Ângulos $\theta_i(x_i, y_i)$ | `87.2992°`, `85.0889°`, `72.3493°`, `85.4632°` | `87.2992°`, `85.0889°`, `72.3493°`, `85.4632°` | Conferido |
| Método da Potência em $A_2$ ($x^{(0)} = e_3$, $\text{tol} = 10^{-8}$) | `818` iterações, $\lambda_{\text{dom}} = 17.6085936704$ | `818` iterações, $\lambda_{\text{dom}} = 17.6085936704$ | Conferido |
| Erro relativo final ($k=818$) e Resíduo $\|A_2 q_k - \lambda_1 q_k\|_2$ | $9.851328 \times 10^{-9}$ e $3.49 \times 10^{-7}$ | $9.851328 \times 10^{-9}$ e $3.49 \times 10^{-7}$ | Conferido |
| Razão assintótica $|\lambda_2 / \lambda_1|$ em $A_2$ (via Schur) | $17.26241681 / 17.60859358 \approx 0.980340$ | $17.26241681 / 17.60859358 \approx 0.980340$ | Conferido |

---

## 5. Eventuais Limitações e Observações

1. **Fontes locais:** caso o computador institucional não possua a família *Latin Modern* ou *Computer Modern* instalada localmente, o CSS recorre automaticamente às fontes serifadas matemáticas nativas do sistema operacional (*Cambria* / *Times New Roman* / *Georgia*), mantendo o alinhamento tabular e a integridade de todas as equações e matrizes.
2. **Notas do apresentador em tela única:** como o arquivo opera em janela única sem servidor, abrir o painel de notas (`[N]`) exibe o guia na parte inferior da própria tela projetada; por isso, o painel permanece recolhido por padrão.
