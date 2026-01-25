@echo off
echo ========================================
echo DCEvo 依赖安装脚本 (CUDA版本)
echo ========================================
echo.

REM 第一步: 安装PyTorch (CUDA 11.7版本)
echo [1/3] 正在安装 PyTorch (CUDA 11.7)...
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117
if %errorlevel% neq 0 (
    echo 错误: PyTorch 安装失败!
    pause
    exit /b 1
)
echo PyTorch 安装成功!
echo.

REM 第二步: 安装其他依赖
echo [2/3] 正在安装其他依赖包...
pip install timm==1.0.15
pip install einops==0.8.1
pip install opencv-python==4.11.0.86
pip install scikit-image==0.21.0
pip install seaborn==0.13.2
pip install kornia==0.7.0
pip install pygad==3.4.0
pip install ultralytics>=8.0.0
if %errorlevel% neq 0 (
    echo 警告: 部分依赖安装可能失败
)
echo 依赖包安装完成!
echo.

REM 第三步: 验证安装
echo [3/3] 验证安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
if %errorlevel% neq 0 (
    echo 警告: PyTorch验证失败
)
echo.

echo ========================================
echo 安装完成!
echo ========================================
echo 如果CUDA可用显示为False,请检查:
echo 1. NVIDIA驱动是否正确安装
echo 2. CUDA版本是否匹配 (需要CUDA 11.7)
echo.
pause
