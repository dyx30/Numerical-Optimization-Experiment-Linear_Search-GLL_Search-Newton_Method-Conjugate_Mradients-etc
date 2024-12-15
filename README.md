# Numerical-Optimization-Experiment-Linear_Search-GLL_Search-Newton_Method-Conjugate_Mradients-etc
数值最优化方法实验，包括线搜索，GLL搜索，修正Cholesky分解，阻尼Newton，拟Newton，共轭梯度法等算法，以及常见测试函数Extended Rosenbrock等。

# 实验说明
该项目代码主要面向两次数值最优化方法大作业。<br />
experiment1：<br />
<img src="https://github.com/user-attachments/assets/ba5e4a85-d728-49a7-aa34-4872551e64d2" width="400" /><br />
experiment2：<br />
<img src="https://github.com/user-attachments/assets/38b7c7cd-52a5-493b-8810-36961974e4c7" width="450" /><br />
### 文件说明：


- `damped_newton.py`: 阻尼Newton方法，可以指定搜索准则;
- `quasi_newton.py`:拟Newton法，L_BFGS;
- `conjugate_gradients.py`:共轭梯度法，PRP+和FR;
- `linear_search.py`: 强Wofle准则的单调线搜索;
- `GLL_linear_search.py`:GLL搜索;
- `utils.py`:Cholesky分解;
- `main.py`:第一次实验所用的测试函数和命令。该部分中测试函数使用自动微分，大规模问题下建议手动微分;
- `variably_dimensioned_function.py`、`discrete_boundary_function.py`、`extended_rosenbrock_function.py`，第二次实验使用的三个测试函数，手动微分;
- `more-testing`:*Testing unconstrained optimization software*,包含了更多的测试样例。



# Reference：
高立.数值最优化方法[M]. 北京,北京大学出版社,2014.8.<br />
刘浩洋, 户将, 李勇锋, 文再文.最优化计算方法[M].北京，高等教育出版社, 2023  <br />
Moré J J, Garbow B S, Hillstrom K E. Testing unconstrained optimization software[J]. ACM transactions on mathematical software (TOMS), 1981, 7(1): 17-41.
