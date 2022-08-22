# Prioritizing samples in Reinforcement Learning based on Reducible Loss

The codebases for each benchmark are present in a separate folder. Each folder contains a `scripts` folder that contains the files to reproduce the results in the paper.

To start the runs for each benchmark, you can use the `run.sh` file in the `scripts` folder. It will start the runs for each environment in the benchmark.

In the run script you can choose which algorithm to run. Whether it is the baseline version, the PER verion or the ReLo version.

The MinAtar codebase is modified from the official [codebase](https://github.com/kenjyoung/MinAtar) for the MinAtar environment.

The SAC codebase is modified from [here](https://github.com/denisyarats/pytorch_sac).

The Rainbow codebase is modified from [here](https://github.com/Kaixhin/Rainbow).

Each codebase contains the instructions required to set it up from the original authors. No additional setup is required to run the code.