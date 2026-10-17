# Gate 2: model quality (validation only)

**Verdict: PASS**. mean best validation Dice of conventional models (3 seeds) >= 0.60.

Conventional mean best val Dice: 0.624 (SD 0.0099)

| Run | Best val Dice | Last | Evaluations |
|---|---|---|---|
| v2_conv_s0 | 0.6296 | 0.6258 | 100 |
| v2_maml_s0 | 0.6621 | 0.6139 | 16 |
| v2_conv_s1 | 0.6125 | 0.5831 | 100 |
| v2_maml_s1 | 0.6080 | 0.5909 | 16 |
| v2_conv_s2 | 0.6298 | 0.5905 | 100 |
| v2_maml_s2 | 0.6034 | 0.5652 | 16 |
| v2_lovo_GE_MEDICAL_SYSTEMS_conv | 0.5462 | 0.4851 | 100 |
| v2_lovo_GE_MEDICAL_SYSTEMS_maml | 0.5908 | 0.5815 | 16 |
| v2_lovo_SIEMENS_conv | 0.5839 | 0.5422 | 100 |
| v2_lovo_SIEMENS_maml | 0.6141 | 0.5474 | 16 |
| v2_lovo_TOSHIBA_conv | 0.6352 | 0.6155 | 100 |
| v2_lovo_TOSHIBA_maml | 0.6260 | 0.5595 | 16 |
| v2_lovo_Philips_conv | 0.6278 | 0.6146 | 100 |
| v2_lovo_Philips_maml | 0.6352 | 0.5820 | 16 |

MAML values are validation Dice after adapting on k=4 annotated slices; conventional values are without adaptation, so the two columns are not directly comparable (that comparison is Aim 3).