# custom_PPO_with_arena-ros


arena-ros 패키지에서는 기본적으로 stable-baselines3의 model들을 사용하지만 이후 논문화를 원활하게 하기위해서 stable-baselines3와 분리하여 PPO model을 직접 구현하였고, 
시뮬레이션 환경이 Flatland로 이루어진 arena-ros 패키지와 엮어 학습할 수 있는 환경을 구현함.

arena-rosnav/training/scripts/ 위치의 train.py와 PPO.py를 제작함.
