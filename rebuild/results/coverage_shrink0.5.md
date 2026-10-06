# Coverage of 95% rank intervals (1000 simulated benchmarks, 1000 bootstrap draws each, true gaps shrunk to 0.5x the observed gaps)

Monte Carlo standard error of a coverage near 95%: 0.007. Run time 107s.

| method | mean per-model coverage | worst model | top-20 models: mean (min) | all 97 at once | true #1 kept in 'could be #1' set | mean width, top 20 |
|---|---|---|---|---|---|---|
| per-model (marginal) | 1.000 | 0.997 (Unirep, true rank 94) | 1.000 (0.998) | 0.977 | 1.000 | 41.2 ranks |
| per-model, small-n fix | 1.000 | 0.997 (Unirep, true rank 94) | 1.000 (0.999) | 0.981 | 1.000 | 42.5 ranks |
| simultaneous | 1.000 | 0.999 (ProSST (K=1024), true rank 7) | 1.000 (0.999) | 0.998 | 1.000 | 53.3 ranks |
| naive: no multiple-comparison correction | 0.987 | 0.948 (MSA Transformer (ensemble), true rank 36) | 0.985 (0.959) | 0.522 | 0.973 | 23.4 ranks |

## Are the bootstrap standard errors the right size?

Ratio of the average bootstrap SE of each top-20 model's gap to the true #1, to the actual spread of that gap across simulated benchmarks (1.00 = right size, below 1 = too narrow):

- plain bootstrap: median 0.965, range 0.947 to 1.000
- with small-n fix: median 0.997, range 0.968 to 1.030
