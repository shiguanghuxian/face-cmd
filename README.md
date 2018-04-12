# 人脸识别命令行
功能

1.识别人脸数 2.比对人脸

## 编译步骤
1.编译程序

编译之前需要修改CMakeLists.txt文件，将INCLUDE_DIRECTORIES中引入的[SeetaFaceEngine](https://github.com/seetaface/SeetaFaceEngine)头文件地址改成正确的。

```
mkdir build
cmake ..
make

复制外层model到build目录
```

2.请查看model和libs目录的README.md文件将相关文件考入
