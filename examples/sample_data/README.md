# Fixed sample data

`periodic_large_test_samples.npz` contains four entries from the completed
`mnist_large_v1` periodic test split. The archive stores initial-pressure images,
k-Wave target measurements, analytical Fourier predictions, fixed 25% masks,
masked measurements, MNIST labels, and original source indices.

The masks use seed `20260802` and are copied from the completed reconstruction
experiment. All pressure arrays use the same `64 x 64` image grid and `64 x 91`
sensor-time grid as the thesis experiments.
