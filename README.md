# Read Me

### Goals:
- I wanted to create code that helped filter and prioritize essential data to compute parameters from summary statistics. My simulate-and-recover code mainly focused on testing the forward and inverse diffusion models, with the forward equation giving summary statistics in terms of parameters and the inverse equation giving the parameters in terms of summary statistics.

### Methods:
- **SRC: forward and inverse model**
  - The forward model turns parameters of the EZ diffusion model to compute different summary statistics.
- **SRC: forward and inverse model**
  - The forward model turns parameters of the EZ diffusion model to compute different summary statistics.
- **Bash:**
  - I also included two bash scripts, one to run src, which runs the entire simulation exercise, and one to run the tests!
  
### Conclusions:
- Simulate_and_recover outputs the used sample size, the average biases of the drift rate, boundary separation, and non-decision time.
- It also calls on the forward and inverse models, which helps compare the two to see how accurate the inverse process's outputs are. A low bias and small squared error show that the method is reliable. The sample size is different because, with a larger sample, the inverse model will be more similar (in this case, showing a lower number in average bias and average squared error) to the real values.
- **Example:** 
  Sample Size: 10
  Average Bias [v, a, t]: [ 6.02438831e-01 -8.81974945e+02  1.49593905e+09]
  Average Squared Error [v, a, t]: [9.31816845e-01 4.14895832e+06 1.70479404e+19]

  Sample Size: 40
  Average Bias [v, a, t]: [ 4.11228210e-01 -1.29357930e+02  2.40029536e+08]
  Average Squared Error [v, a, t]: [6.33484395e-01 6.43511497e+05 2.71605765e+18]

  Sample Size: 4000
  Average Bias [v, a, t]: [ 0.3490102  -1.36608652 25.98392593]
  Average Squared Error [v, a, t]: [5.33704403e-01 6.48759392e+00 5.40928069e+03]
- **Tests:**
  - I incorporated five tests to guarantee that the results are not due to confounding issues.
      - Test_forward_inverse_consist: This test looks for when the observed and predicted statistics are the same.
      - Test-forward-returns-valid-accuracy: This test ensures that the forward model's output falls between 0 and 1. If not, it is not realistic.
      - Test_fixed_seed_reproducibility: This test checks if the results coincide with the seed.
      - Test_sample_size_effect: this checks that the squared error gets smaller as the sample size gets bigger.
      - integration_test_fixed_params + Test_integration_parameter_set: these tests test with specific values if average bias appears.
  

### Results seen:
  - Since the average bias and average squared error values decreased as the sample size increased, and the data adequately passed my five tests, I can conclude that the inverse model correctly finds the actual parameters with little to no bias. This means that in most contexts, the inverse model is equivalent in usage to the forward model.
 
### Sources:
- **Visual Studio Code Shell Integration:**  
  [https://code.visualstudio.com/docs/terminal/shell-integration](https://code.visualstudio.com/docs/terminal/shell-integration)
- **AskUbuntu: How to Run and Debug Bash Scripts from VSCode:**  
  [https://askubuntu.com/questions/1228213/how-to-run-and-debug-bash-script-from-vscode](https://askubuntu.com/questions/1228213/how-to-run-and-debug-bash-script-from-vscode)
- **StackOverFlow:**
  [https://stackoverflow.com/questions/40185437/no-module-named-numpy-visual-studio-code](https://stackoverflow.com/questions/40185437/no-module-named-numpy-visual-studio-code)

### Artificial Intelligence Usage:
- **ChatGPT:**  
  - Used as a resource to understand how to create and implement the equations (such as the Forward and Inverse EZ diffusion equations) and to create the necessary tests.
  - Also used to understand Bash Scripts
