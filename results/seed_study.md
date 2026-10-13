# Multi-seed study (test split, default configuration runs)

## CL_WC

| model | seeds | MAE mean ± std | RMSE | R2 | kappa@20 |
|---|---|---|---|---|---|
| resnet_transformer_hybrid | 10 | 4.32 ± 0.10 | 5.50 | 0.983 | 0.937 |
| cnn_transformer_hybrid | 10 | 4.54 ± 0.07 | 5.87 | 0.981 | 0.933 |
| resnet3d_34 | 10 | 4.91 ± 0.19 | 6.39 | 0.977 | 0.928 |
| resnet3d_18 | 10 | 5.17 ± 0.69 | 6.66 | 0.975 | 0.918 |
| cnn_conv_control | 10 | 5.37 ± 1.20 | 6.91 | 0.973 | 0.917 |
| swinunetr_encoder | 10 | 6.49 ± 0.80 | 8.30 | 0.961 | 0.891 |
| swinunetr_encoder (pretrained) | 10 | 6.96 ± 0.49 | 8.95 | 0.955 | 0.881 |
| vit3d_small | 10 | 8.27 ± 1.11 | 11.02 | 0.933 | 0.845 |

## CL_WCB

| model | seeds | MAE mean ± std | RMSE | R2 | kappa@20 |
|---|---|---|---|---|---|
| resnet_transformer_hybrid | 10 | 3.75 ± 0.11 | 4.75 | 0.987 | 0.931 |
| cnn_transformer_hybrid | 10 | 3.98 ± 0.08 | 5.13 | 0.985 | 0.930 |
| resnet3d_34 | 10 | 4.32 ± 0.16 | 5.61 | 0.982 | 0.921 |
| resnet3d_18 | 10 | 4.50 ± 0.64 | 5.79 | 0.980 | 0.923 |
| cnn_conv_control | 10 | 4.79 ± 1.15 | 6.16 | 0.977 | 0.923 |
| swinunetr_encoder | 10 | 5.59 ± 0.68 | 7.18 | 0.969 | 0.901 |
| swinunetr_encoder (pretrained) | 10 | 5.99 ± 0.41 | 7.82 | 0.964 | 0.912 |
| vit3d_small | 10 | 7.37 ± 1.26 | 9.87 | 0.943 | 0.869 |

## CL_pons

| model | seeds | MAE mean ± std | RMSE | R2 | kappa@20 |
|---|---|---|---|---|---|
| resnet_transformer_hybrid | 10 | 4.09 ± 0.17 | 5.36 | 0.985 | 0.951 |
| cnn_transformer_hybrid | 10 | 4.59 ± 0.18 | 6.03 | 0.981 | 0.954 |
| resnet3d_34 | 10 | 4.76 ± 0.28 | 6.26 | 0.980 | 0.949 |
| resnet3d_18 | 10 | 5.19 ± 1.40 | 6.90 | 0.976 | 0.944 |
| cnn_conv_control | 10 | 6.46 ± 2.96 | 8.37 | 0.972 | 0.931 |
| swinunetr_encoder | 10 | 6.70 ± 1.02 | 8.76 | 0.964 | 0.912 |
| swinunetr_encoder (pretrained) | 10 | 6.81 ± 0.55 | 9.05 | 0.957 | 0.916 |
| vit3d_small | 10 | 9.42 ± 1.38 | 12.25 | 0.919 | 0.881 |
