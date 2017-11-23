# Capstone_Project_Rossmann_Sales_Prediction
Udacity machine learning (advanced) Nanodegree final project
English version is simplied, because I did the project report in Chinese. The final score on the Kaggle LB was top 7.6% with my solution.

CN:
1. 项目报告名为 'mlnd_capstone_report_Li.pdf'.
2. Python代码在'main_folder'文件夹中，先运行rossmann_main.py。model_aver.py是用来计算模型预测结果平均值的。
3. 项目所用到的软件是Python，库分别是pandas, seaborn, operator还有xgboost。其中特别注意的是Xgboost采用的是GPU版本，具体的下载安装方法由如下两个网页提供:http://www.picnet.com.au/blogs/guido/post/2016/09/22/how-to-build-xgboost-on-windows/以及https://github.com/dmlc/xgboost/blob/master/doc/build.md。使用的显卡是Geforce GTX1070。安装GPU Xgboost之前需要安装CUDA库。总体来说安装过程比较复杂，导师也可以使用CPU版本的来运行，但是运算速度会很慢。
4. 额外材料都放在appendix文件夹中。其中，在prediction result文件夹中，所有上传Kaggle的.csv文件都在里面，导师请查阅或者尝试上传；上传的结果保存在Rossmann_Submission.pdf中(直接网页截图)。
5. 在我的机器上，使用GTX1070，并行Xgboost的情况下，跑一组大概5000次迭代大约要10分钟左右。如果使用CPU版本的话，我没有测试，但是时间应该会很长。
6. 运行代码的视频演示名为:'video demo'，也收录在appendix文件夹中。
7. 两个jupyter文件在JupyterNotebooks文件夹中，EDA_and_Visualization_Notebook.ipynb是对数据做分析和可视化作用的，Visualization_RMSPE.ipynb是用作对最终结果做可视化的。

EN:
1. The project report is named 'mlnd_capstone_report_Li.pdf'.
2. Python code In the main_folder folder, run rossmann_main.py first. model_aver.py is used to calculate the average of the model predictions.
3. The project used Python, and packages such as pandas, seaborn, operator and Xgboost. One of the special note is Xgboost uses the GPU version, the specific download and installation method is provided by the following two pages: http://www.picnet.com.au/blogs/guido/post/2016/09/22/how-to-build-xgboost-on-windows / and https://github.com/dmlc/xgboost/blob/master/doc/build.md. The graphics card used is the Geforce GTX1070. You need to install the CUDA library before installing GPU Xgboost. Overall the installation process is more complicated, mentors can also use the CPU version to run, but the operation speed will be very slow.
4. Additional materials are kept in the appendix. Among them, in the prediction result folder, all the .csv files for uploading Kaggle are included. The instructor can check or try the upload. The uploaded result is saved in Rossmann_Submission.pdf (screenshot of the direct webpage).
On my PC, using GTX1070, in parallel with Xgboost, running a set of about 5000 iterations takes about 10 minutes to do so. I did not test if using the CPU version, much more time should be required for the training process.
6. The video demo of the running code is named 'video demo' and is also included in the appendix folder.
7. Two jupyter files In the JupyterNotebooks folder, EDA_and_Visualization_Notebook.ipynb is used to analyze and visualize the data. Visualization_RMSPE.ipynb is used to visualize the final result.
