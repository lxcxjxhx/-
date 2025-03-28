import os
import sys
import subprocess
import pkg_resources
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dependency_check.log'),
        logging.StreamHandler()
    ]
)

# 定义核心依赖包
REQUIRED_PACKAGES = {
    'torch': '>=2.0.0',
    'transformers': '>=4.30.0',
    'peft': '>=0.4.0',
    'accelerate': '>=0.20.0',
    'bitsandbytes': '>=0.41.0',
    'optimum': '>=1.16.0',
}

def check_python_version():
    """检查Python版本"""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logging.error(f"Python版本不满足要求: 需要 {required_version[0]}.{required_version[1]} 或更高版本")
        return False
    return True

def check_cuda_availability():
    """检查CUDA是否可用"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logging.info(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA版本: {torch.version.cuda}")
        else:
            logging.warning("CUDA不可用，将使用CPU进行训练")
        return cuda_available
    except ImportError:
        logging.warning("无法导入torch，无法检查CUDA状态")
        return False

def install_package(package_name, version_spec):
    """安装单个包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}{version_spec}"])
        logging.info(f"成功安装 {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"安装 {package_name} 失败: {str(e)}")
        return False

def check_and_install_dependencies():
    """检查并安装所有依赖"""
    logging.info("开始检查依赖...")
    
    # 检查Python版本
    if not check_python_version():
        return False
    
    # 检查CUDA可用性
    check_cuda_availability()
    
    # 检查并安装依赖
    for package, version in REQUIRED_PACKAGES.items():
        try:
            pkg_resources.require(f"{package}{version}")
            logging.info(f"{package} 已安装且版本满足要求")
        except (pkg_resources.VersionConflict, pkg_resources.DistributionNotFound):
            logging.warning(f"缺少依赖包: {package}{version}")
            if not install_package(package, version):
                logging.error(f"安装依赖失败: {package}")
                return False
    
    logging.info("所有依赖检查完成")
    return True

def main():
    """主函数"""
    try:
        if check_and_install_dependencies():
            logging.info("依赖检查通过，可以启动主程序")
            return True
        else:
            logging.error("依赖检查失败，请检查错误日志")
            return False
    except Exception as e:
        logging.error(f"依赖检查过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    main() 