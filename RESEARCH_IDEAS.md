# Candidate research ideas for The Journal of Economic Inequality

Selection criteria: fit with JEI scope (measurement, statistical and data issues, inequality of opportunity, poverty), a contribution that uses a machine-learning background, data that can be obtained free or by registration, and a gap confirmed by a literature check (26 Nov 2026).

---

## Idea 1 (recommended): ML poverty maps understate inequality

**Title:** *Predicted welfare is too equal: inequality and poverty measurement with machine-learning poverty maps*

**Motivation.** Governments and agencies use satellite and phone-based poverty maps for targeting, and researchers now compute inequality from them, for example from the Meta Relative Wealth Index. A prediction ŷ = E[y | x] is a mean-preserving contraction of true welfare y, because y = ŷ + e with E[e | ŷ] = 0. Any Lorenz-consistent index therefore satisfies I(ŷ) ≤ I(y). This is the same logic as the ex-ante lower bound in inequality of opportunity. The "inequality of the predicted map" is really the inequality explained by satellite features, not total inequality. Regularised ML shrinks predictions further. Current work only mentions this as a caveat (Gini "may be underpredicted") and does not quantify or correct it.

**Research questions**
1. How large is the downward bias in Gini, MLD, Theil and poverty headcounts computed from ML-predicted welfare, compared with survey ground truth? How does it depend on predictive R², the aggregation level (household, cluster, district), and the index used?
2. Formally, under what conditions is the bias signed and bounded, and how does it map to the R² of the first-stage model? This links to the IOp lower-bound literature.
3. Can a distribution-recovering estimator restore unbiased inequality and poverty estimates with valid coverage? Candidates are quantile regression forests, conformal predictive distributions, or ELL-style residual simulation.
4. Do between-region inequality rankings and poverty-targeting decisions change after the correction?

**Data.** Two publicly available sources:
- World Bank LSMS-ISA household consumption surveys (Malawi, Nigeria, Tanzania, Uganda, Ethiopia; free with registration).
- Public satellite features: MOSAIKS, SustainBench, and the Meta Relative Wealth Index.

**Contribution.** A measurement result (a sign and bound on the bias), a practical correction, and empirical evidence across several countries. This fits JEI's "related statistical and data issues" and "poverty".

**Closest literature and the gap.**
- The PNAS Nexus (2025) fine-scale inequality maps paper notes possible Gini underprediction but does not quantify or correct it.
- Aiken et al. (IJCAI 2023) document urban–rural bias in poverty maps.
- The World Bank "Poverty Mapping in the Age of Machine Learning" paper says R² hides systematic bias.
- A 2026 arXiv paper on uncertainty-aware poverty mapping proposes prediction intervals, but not inequality indices.

**Risk.** Medium. Survey consumption is noisy (measurement error inflates I(y)), so the analysis must separate model shrinkage from measurement error.

---

## Idea 2: Inequality of opportunity in Vietnam, 2010–2022, with debiased ML

**Research questions**
- How has ex-ante IOp in household per-capita income/consumption in Vietnam evolved from 2010 to 2022?
- Which circumstances drive it: ethnicity, birthplace (province and urban/rural), gender, parental background?
- Is the trend robust to the choice of ML algorithm once the debiased estimator of Escanciano and Terschuur is used?

**Data.** VHLSS. It must be purchased from the GSO, and parental background is limited.

**Gap and risk.** Vietnam mobility is already studied (ADB; Hoang-Duc, IARIW 2025), and the Monash GEOM Asia-Pacific ML-IOp work may already cover Vietnam. The novelty is moderate and depends on the debiased inference plus the 2020–2022 COVID period. The JEI special issue on mobility in LMICs closed on 30 April 2026.

---

## Idea 3: Inequality of opportunity in algorithmic credit decisions

**Research question.** Using the public HMDA mortgage data (2018–2024), how much inequality in approval and pricing is explained by circumstances (race, ethnicity, sex, age)? How much is explained by "effort-like" financial characteristics (income, LTV, DTI, credit score band)? How did this change with the spread of automated underwriting?

**Contribution.** It connects the IOp framework (ex-ante and ex-post, Roemer) with algorithmic-fairness metrics.

**Risk.** The novelty has not been checked yet. The fairness and discrimination literature on HMDA is large, so the IOp framing must be the main contribution.

---

## Sources

- JEI special issue call (deadline 30 Apr 2026): https://www.ecineq.org/category/announcements/
- Mapping fine-scale socioeconomic inequality using ML and remotely sensed data, PNAS Nexus (2025): https://academic.oup.com/pnasnexus/article/4/2/pgaf040/8005621
- Aiken et al., Fairness and representation in satellite-based poverty maps, IJCAI 2023: https://ijcai.org/proceedings/2023/653
- Poverty Mapping in the Age of Machine Learning, World Bank: https://openknowledge.worldbank.org/server/api/core/bitstreams/628af49e-8f46-4679-a04a-0a9f00c8684c/content
- Beyond point predictions: uncertainty-aware satellite poverty mapping (arXiv 2026): https://arxiv.org/pdf/2608.23322
- Escanciano and Terschuur, Debiased machine learning U-statistics / ML inference on IOp: https://arxiv.org/abs/2206.05235
- Terschuur, Educational inequality of opportunity and mobility in Europe: https://arxiv.org/pdf/2212.02407
- Brunori, Hufe and Mahler, The roots of inequality, Scand. J. Econ. (2023): https://researchonline.lse.ac.uk/id/eprint/118220/1/Scandinavian_J_Economics_2023_Brunori.pdf
- ADB, Intragenerational and intergenerational mobility in Viet Nam: https://www.adb.org/publications/intragenerational-and-intergenerational-mobility-viet-nam
- Hoang-Duc, IARIW 2025: https://iariw.org/wp-content/uploads/2025/09/Duc-Hanoi.pdf
- Monash GEOM Asia-Pacific IOp working paper: https://www.monash.edu/__data/assets/pdf_file/0004/4296046/wp06-26.pdf
