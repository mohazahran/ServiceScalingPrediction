Usage
=====

## Modules
- `function.py`: DC-BPTT and $\infty$-SGD implementation as callable functions.
  ```python
import function as F
  import torch
  import numpy as np
  
  # generate a random matrix
  X = torch.randn(5, 5)
  for i in range(5):
      X.data[i, i] = 0
  
  # get differentiable steady-state distribution by infty-SGD
  pi = F.stdy_dist_rrx(X, None, trick='rrinf')
  
  # get differentiable steady-state distribution by DC-BPTT t^*=7
  pi = F.stdy_dist_pow(X, None, trick='7')
  ```

- `module.py`: Queue model modules.
- `data.py`: Data generation process as callable functions.

## Experiments

- `mm1k_all_cond_all_in.py`: Run all reported M/M/1/$K$ experiments on given prior regularization strength, and save results to current directory.

  ```bash
  # run with alpha = 100
  python mm1k_all_cond_all_in.py 100
  ```

- `mmmmr_all_cond_all_in.py`: Run all reported M/M/m/m+r experiments on given prior regularization strength, and save results to current directory.

- `lbwb_all_cond_all_in.py`: Run all reported web-browsing queue experiments on given prior regularization strength, and save results to current directory.

- `real.py`: Run all reported real-world experiments on given data and prior regularization strength, and save results to current directory.

  ```python
  # run on RTT collection with upper triangular prior
  python real.py rtt -1
  ```

  Here, "-1" means using upper triangular prior, and other positive values means using M/M/1/$K$ prior.

## Utils
- `mix.py`: Examples of slow mixing M/M/1/$K$.
- `viz_all.py`: Visualize all results.
- `viz_E.py`: Visualize learned matrix using upper triangular prior.