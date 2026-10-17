# Gate 2: model quality (validation only)

**Verdict: FAIL**. mean best validation Dice of conventional models (3 seeds) >= 0.60.

Conventional mean best val Dice: 0.5829 (SD 0.034)

| Run | Best val Dice | Last | Evaluations |
|---|---|---|---|
| conv_s0 | 0.6221 | 0.5909 | 40 |
| maml_s0 | 0.6696 | 0.6325 | 16 |
| conv_s1 | 0.5651 | 0.5455 | 40 |
| maml_s1 | 0.5938 | 0.5796 | 16 |
| conv_s2 | 0.5616 | 0.5432 | 40 |
| maml_s2 | 0.5898 | 0.5247 | 16 |
| lovo_GE_MEDICAL_SYSTEMS_conv | 0.5707 | 0.5624 | 40 |
| lovo_GE_MEDICAL_SYSTEMS_maml | 0.5773 | 0.5479 | 16 |
| lovo_SIEMENS_conv | 0.5284 | 0.5141 | 40 |
| lovo_SIEMENS_maml | 0.6643 | 0.5756 | 16 |
| lovo_TOSHIBA_conv | 0.6286 | 0.5971 | 40 |
| lovo_TOSHIBA_maml | 0.6416 | 0.6130 | 16 |
| lovo_Philips_conv | 0.6313 | 0.6148 | 40 |
| lovo_Philips_maml | 0.6631 | 0.6357 | 16 |

MAML values are validation Dice after adapting on k=4 annotated slices; conventional values are without adaptation, so the two columns are not directly comparable (that comparison is Aim 3).