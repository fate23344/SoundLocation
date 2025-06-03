# SoundLocation
基于Matlab and Python的四麦声源定位。
项目使用ReSpeaker Mic Array v2.0平面四麦，利用TDOA算法，在Matlab上实现静态二维定位，在Python上实现实时二维方向定位。两个代码是独立运行的。
受麦克风数量与位置分布影响，无法实现三维的定位。
Python代码使用远场模型，无法精确确定距离，故没有实现此功能的代码。
使用Python代码时，注意输出的麦克风序号，将麦克风阵列的序号填入相应参数中。
