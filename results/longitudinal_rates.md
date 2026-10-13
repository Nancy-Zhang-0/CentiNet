# Longitudinal validity: predicted vs true annual CL rate

Split: test; subjects with >= 2 scans and >= 0.9 y span. Rate = OLS slope of CL vs time (CL/year). ICC = ICC(2,1), absolute agreement.

## cnn_transformer_hybrid_0_20260910-190030

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.947 | 0.941 | 0.861 | +0.47 | 0.98 | +0.29 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.950 | 0.946 | 0.871 | +0.40 | 0.83 | +0.22 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.915 | 0.912 | 0.835 | +0.25 | 0.99 | -0.04 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 90.6% | 86.5% | 93.8% | 0.809 |
| CL_WC | 2 | 43 | 92.3% | 93.0% | 91.9% | 0.837 |
| CL_WC | 3 | 37 | 91.5% | 83.8% | 95.0% | 0.799 |
| CL_WCB | 1.5 | 52 | 94.0% | 92.3% | 95.4% | 0.879 |
| CL_WCB | 2 | 44 | 93.2% | 90.9% | 94.5% | 0.854 |
| CL_WCB | 3 | 32 | 92.3% | 90.6% | 92.9% | 0.812 |
| CL_pons | 1.5 | 61 | 89.7% | 88.5% | 91.1% | 0.795 |
| CL_pons | 2 | 50 | 89.7% | 88.0% | 91.0% | 0.790 |
| CL_pons | 3 | 39 | 90.6% | 84.6% | 93.6% | 0.787 |

## resnet3d_18_0_20260910-190023

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.926 | 0.925 | 0.878 | +0.29 | 1.06 | +0.13 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.925 | 0.924 | 0.877 | +0.29 | 0.94 | +0.12 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.889 | 0.886 | 0.803 | +0.36 | 1.12 | +0.02 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 93.2% | 90.4% | 95.4% | 0.861 |
| CL_WC | 2 | 43 | 91.5% | 86.0% | 94.6% | 0.814 |
| CL_WC | 3 | 37 | 90.6% | 83.8% | 93.8% | 0.781 |
| CL_WCB | 1.5 | 52 | 95.7% | 92.3% | 98.5% | 0.913 |
| CL_WCB | 2 | 44 | 94.0% | 90.9% | 95.9% | 0.872 |
| CL_WCB | 3 | 32 | 93.2% | 87.5% | 95.3% | 0.828 |
| CL_pons | 1.5 | 61 | 92.3% | 93.4% | 91.1% | 0.846 |
| CL_pons | 2 | 50 | 88.0% | 84.0% | 91.0% | 0.754 |
| CL_pons | 3 | 39 | 88.9% | 84.6% | 91.0% | 0.752 |

## resnet3d_34_0_20260910-190026

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.921 | 0.919 | 0.910 | +0.46 | 1.09 | +0.34 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.919 | 0.917 | 0.899 | +0.41 | 0.98 | +0.28 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.892 | 0.887 | 0.800 | +0.37 | 1.06 | +0.02 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 93.2% | 90.4% | 95.4% | 0.861 |
| CL_WC | 2 | 43 | 93.2% | 90.7% | 94.6% | 0.853 |
| CL_WC | 3 | 37 | 91.5% | 83.8% | 95.0% | 0.799 |
| CL_WCB | 1.5 | 52 | 92.3% | 88.5% | 95.4% | 0.843 |
| CL_WCB | 2 | 44 | 92.3% | 86.4% | 95.9% | 0.834 |
| CL_WCB | 3 | 32 | 94.0% | 90.6% | 95.3% | 0.851 |
| CL_pons | 1.5 | 61 | 89.7% | 88.5% | 91.1% | 0.795 |
| CL_pons | 2 | 50 | 88.0% | 84.0% | 91.0% | 0.754 |
| CL_pons | 3 | 39 | 87.2% | 82.1% | 89.7% | 0.713 |

## resnet_transformer_hybrid_0_20260910-233046

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.943 | 0.941 | 0.902 | +0.30 | 1.01 | +0.18 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.941 | 0.940 | 0.905 | +0.30 | 0.89 | +0.17 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.922 | 0.920 | 0.868 | +0.36 | 0.97 | +0.13 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 94.0% | 92.3% | 95.4% | 0.879 |
| CL_WC | 2 | 43 | 91.5% | 93.0% | 90.5% | 0.820 |
| CL_WC | 3 | 37 | 94.0% | 86.5% | 97.5% | 0.859 |
| CL_WCB | 1.5 | 52 | 94.9% | 92.3% | 96.9% | 0.896 |
| CL_WCB | 2 | 44 | 91.5% | 88.6% | 93.2% | 0.818 |
| CL_WCB | 3 | 32 | 92.3% | 90.6% | 92.9% | 0.812 |
| CL_pons | 1.5 | 61 | 92.3% | 88.5% | 96.4% | 0.846 |
| CL_pons | 2 | 50 | 90.6% | 92.0% | 89.6% | 0.809 |
| CL_pons | 3 | 39 | 93.2% | 97.4% | 91.0% | 0.852 |

## swinunetr_encoder_0_20260910-190154

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.895 | 0.893 | 0.829 | +0.36 | 1.34 | +0.14 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.902 | 0.902 | 0.875 | +0.28 | 1.18 | +0.11 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.859 | 0.859 | 0.884 | +0.19 | 1.34 | -0.01 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 82.1% | 76.9% | 86.2% | 0.634 |
| CL_WC | 2 | 43 | 88.0% | 83.7% | 90.5% | 0.743 |
| CL_WC | 3 | 37 | 91.5% | 86.5% | 93.8% | 0.802 |
| CL_WCB | 1.5 | 52 | 85.5% | 80.8% | 89.2% | 0.704 |
| CL_WCB | 2 | 44 | 87.2% | 81.8% | 90.4% | 0.726 |
| CL_WCB | 3 | 32 | 94.0% | 90.6% | 95.3% | 0.851 |
| CL_pons | 1.5 | 61 | 85.5% | 80.3% | 91.1% | 0.710 |
| CL_pons | 2 | 50 | 83.8% | 76.0% | 89.6% | 0.664 |
| CL_pons | 3 | 39 | 92.3% | 84.6% | 96.2% | 0.824 |

## swinunetr_encoder_0_20260910-222956

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.859 | 0.852 | 0.754 | +0.20 | 1.46 | -0.12 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.849 | 0.846 | 0.774 | +0.20 | 1.31 | -0.11 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.791 | 0.790 | 0.741 | +0.32 | 1.48 | -0.14 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 91.5% | 84.6% | 96.9% | 0.825 |
| CL_WC | 2 | 43 | 88.0% | 79.1% | 93.2% | 0.737 |
| CL_WC | 3 | 37 | 89.7% | 73.0% | 97.5% | 0.748 |
| CL_WCB | 1.5 | 52 | 89.7% | 80.8% | 96.9% | 0.789 |
| CL_WCB | 2 | 44 | 88.9% | 79.5% | 94.5% | 0.758 |
| CL_WCB | 3 | 32 | 91.5% | 84.4% | 94.1% | 0.785 |
| CL_pons | 1.5 | 61 | 81.2% | 75.4% | 87.5% | 0.625 |
| CL_pons | 2 | 50 | 84.6% | 78.0% | 89.6% | 0.682 |
| CL_pons | 3 | 39 | 83.8% | 71.8% | 89.7% | 0.627 |

## vit3d_small_0_20260910-190145

| target | n subj | median span (y) | r | ICC(2,1) | slope | intercept | MAE rate | bias rate | mean true rate |
|---|---|---|---|---|---|---|---|---|---|
| CL_WC | 117 | 4.1 | 0.822 | 0.821 | 0.888 | +0.14 | 1.92 | -0.01 | +1.31 ± 4.67 |
| CL_WCB | 117 | 4.1 | 0.846 | 0.838 | 0.981 | +0.01 | 1.61 | -0.02 | +1.35 ± 4.02 |
| CL_pons | 117 | 4.1 | 0.779 | 0.772 | 0.893 | +0.02 | 1.64 | -0.16 | +1.74 ± 3.58 |

### Accumulator classification (true rate >= thr vs predicted)

| target | thr (CL/y) | n true accum. | accuracy | sensitivity | specificity | kappa |
|---|---|---|---|---|---|---|
| CL_WC | 1.5 | 52 | 81.2% | 73.1% | 87.7% | 0.615 |
| CL_WC | 2 | 43 | 82.9% | 74.4% | 87.8% | 0.629 |
| CL_WC | 3 | 37 | 90.6% | 81.1% | 95.0% | 0.778 |
| CL_WCB | 1.5 | 52 | 82.1% | 73.1% | 89.2% | 0.632 |
| CL_WCB | 2 | 44 | 85.5% | 77.3% | 90.4% | 0.686 |
| CL_WCB | 3 | 32 | 91.5% | 84.4% | 94.1% | 0.785 |
| CL_pons | 1.5 | 61 | 86.3% | 77.0% | 96.4% | 0.728 |
| CL_pons | 2 | 50 | 84.6% | 78.0% | 89.6% | 0.682 |
| CL_pons | 3 | 39 | 83.8% | 69.2% | 91.0% | 0.623 |
