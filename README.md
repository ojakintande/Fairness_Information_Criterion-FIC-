# Fairness Information Criterion (FIC) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Active%20Research-blue)](https://github.com/ojakintande/LLM_Who_Audit_Reviewers)

This repository contains the implementation, diagnostic framework, and experimental results for the Fairness Information Criterion (FIC) framework. FIC provides a unified method for statistically rigorous and context-aware fairness auditing in machine learning, offering an actionable bridge between algorithmic outputs and regulatory requirements.

---

## 1. Abstract
The FIC framework moves beyond rigid, binary fairness determinations by integrating performance evaluation with multi-criteria assessments of Independence, Separation, and Sufficiency under an adjustable significance threshold ($\alpha_F$). The framework outputs an interpretable four-tier benchmark—ranging from "Optimal" to "Unacceptable"—alongside visual diagnostics to support practitioner decision-making.

## 2. Key Features & Theoretical Contributions
*   **Uncertainty-Aware Auditing:** Leverages stratified bootstrap confidence intervals for maximum pairwise disparity estimates to ensure statistical rigor.
*   **Statistical Properties:** Theoretically proven to satisfy scale invariance, estimation consistency, fairness–performance tradeoff bounds, threshold monotonicity, and bootstrap consistency.
*   **Context-Aware Certification:** Enables domain-tailored compliance certification through an adjustable significance threshold ($\alpha_F$).
*   **Cross-Modality Robustness:** Validated across both tabular benchmarks (COMPAS and Adult datasets) and unstructured computer vision data (FHIBE).

## 3. Repository Structure
*   `/scripts`: Implementation of the FIC calculation engine, bootstrap inference, and visualization modules.
*   `/data`: Pre-processed datasets (COMPAS, Adult, FHIBE) used for empirical validation.
*   `/results`: Visual diagnostics and fairness certification benchmarks.

## 4. Quick Start
To apply the FIC framework to your model:

1.  **Environment:** Ensure R (v4.x) is installed.
2.  **Dependencies:** Install the required statistical packages:
```r
    install.packages(c("tidyverse", "boot", "ggplot2"))
    ```
3.  **Implementation:** Load the FIC module and define your audit parameters:
```r
    # Example usage:
    source("scripts/fic_engine.R")
    audit_results <- calculate_fic(model_outputs, alpha_f = 0.05)
    print(audit_results$tier_benchmark)
    ```

## 5. Citation
If this framework contributes to your research or auditing practice, please cite our manuscript

## 6. License
This project is licensed under the **MIT License**. See the `LICENSE` file for more details.
