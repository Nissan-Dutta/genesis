# Coverage of 95% rank intervals (1000 simulated benchmarks, 1000 bootstrap draws each)

Monte Carlo standard error of a coverage near 95%: 0.007. Run time 97s.

| method | mean per-model coverage | worst model | top-20 models: mean (min) | all 97 at once | true #1 kept in 'could be #1' set | mean width, top 20 |
|---|---|---|---|---|---|---|
| per-model (marginal) | 1.000 | 0.996 (MSA Transformer (ensemble), true rank 36) | 1.000 (0.998) | 0.975 | 1.000 | 17.6 ranks |
| per-model, small-n fix | 1.000 | 0.996 (MSA Transformer (ensemble), true rank 36) | 1.000 (0.998) | 0.983 | 1.000 | 18.1 ranks |
| simultaneous | 1.000 | 1.000 (AIDO Protein-RAG (16B), true rank 1) | 1.000 (1.000) | 1.000 | 1.000 | 23.2 ranks |
| naive: no multiple-comparison correction | 0.987 | 0.946 (MSA Transformer (ensemble), true rank 36) | 0.985 (0.958) | 0.505 | 0.974 | 10.6 ranks |

## Are the bootstrap standard errors the right size?

Ratio of the average bootstrap SE of each top-20 model's gap to the true #1, to the actual spread of that gap across simulated benchmarks (1.00 = right size, below 1 = too narrow):

- plain bootstrap: median 0.965, range 0.947 to 1.000
- with small-n fix: median 0.997, range 0.968 to 1.030
