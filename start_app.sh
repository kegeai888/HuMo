#!/bin/bash
# -*- coding: utf-8 -*-
# HuMo视频生成器启动脚本
# 自动检查端口占用，清理GPU缓存，启动应用

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 应用配置
APP_NAME="HuMo视频生成器"
APP_PORT=7860
CONDA_ENV="py310"
APP_SCRIPT="app.py"
PROJECT_DIR="/root/HuMo"

# 输出带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}  🎬 $APP_NAME 启动脚本${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# 检查是否在正确的目录
check_directory() {
    print_info "检查项目目录..."
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "项目目录不存在: $PROJECT_DIR"
        exit 1
    fi

    cd "$PROJECT_DIR" || {
        print_error "无法进入项目目录: $PROJECT_DIR"
        exit 1
    }

    if [ ! -f "$APP_SCRIPT" ]; then
        print_error "应用脚本不存在: $APP_SCRIPT"
        exit 1
    fi

    print_success "项目目录检查通过"
}

# 检查conda环境
check_conda_env() {
    print_info "检查conda环境: $CONDA_ENV"

    # 检查conda是否可用
    if ! command -v conda &> /dev/null; then
        print_error "conda未安装或不在PATH中"
        exit 1
    fi

    # 检查环境是否存在
    if ! conda info --envs | grep -q "^$CONDA_ENV "; then
        print_error "conda环境不存在: $CONDA_ENV"
        print_info "请先创建环境: conda create -n $CONDA_ENV python=3.11"
        exit 1
    fi

    print_success "conda环境检查通过"
}

# 检查端口占用
check_port() {
    print_info "检查端口 $APP_PORT 是否被占用..."

    # 检查端口是否被占用
    if lsof -Pi :$APP_PORT -sTCP:LISTEN -t >/dev/null ; then
        print_warning "端口 $APP_PORT 已被占用"

        # 获取占用进程的PID
        PID=$(lsof -Pi :$APP_PORT -sTCP:LISTEN -t)
        print_info "占用进程PID: $PID"

        # 获取进程信息
        PROCESS_INFO=$(ps -p $PID -o pid,ppid,cmd --no-headers 2>/dev/null)
        if [ -n "$PROCESS_INFO" ]; then
            print_info "进程信息: $PROCESS_INFO"
        fi

        # 询问是否杀死占用进程
        read -p "是否杀死占用端口的进程? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "正在杀死进程 $PID..."
            if kill -9 $PID 2>/dev/null; then
                print_success "进程已杀死"
                sleep 2  # 等待端口释放
            else
                print_error "无法杀死进程 $PID，可能需要管理员权限"
                exit 1
            fi
        else
            print_error "端口被占用，无法启动应用"
            exit 1
        fi
    else
        print_success "端口 $APP_PORT 可用"
    fi
}

# 检查GPU状态
check_gpu() {
    print_info "检查GPU状态..."

    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | while read line; do
            echo -e "${CYAN}  📊 $line${NC}"
        done

        # 清理GPU缓存
        print_info "清理GPU缓存..."
        python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU缓存已清理')
else:
    print('CUDA不可用')
" 2>/dev/null || print_warning "清理GPU缓存失败"

    else
        print_warning "nvidia-smi未找到，可能没有GPU或驱动未安装"
    fi
}

# 检查Python依赖
check_dependencies() {
    print_info "检查Python依赖..."

    # 激活conda环境并检查关键依赖
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV

    # 检查关键包
    REQUIRED_PACKAGES=("torch" "gradio" "PIL" "numpy")
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_success "✓ $package 已安装"
        else
            print_error "✗ $package 未安装"
            print_info "请安装依赖: pip install -r requirements.txt"
            exit 1
        fi
    done
}

# 检查模型文件
check_models() {
    print_info "检查模型文件..."

    MODEL_DIR="weights/HuMo"
    if [ -d "$MODEL_DIR" ]; then
        if [ -d "$MODEL_DIR/HuMo-1.7B" ]; then
            print_success "✓ HuMo-1.7B 模型文件存在"
        else
            print_warning "✗ HuMo-1.7B 模型文件不存在"
        fi

        if [ -d "$MODEL_DIR/HuMo-17B" ]; then
            print_success "✓ HuMo-17B 模型文件存在"
        else
            print_warning "✗ HuMo-17B 模型文件不存在"
        fi
    else
        print_warning "模型目录不存在: $MODEL_DIR"
        print_info "请下载模型文件到对应目录"
    fi

    # 检查其他依赖模型
    if [ -d "weights/Wan2.1-T2V-1.3B" ]; then
        print_success "✓ Wan2.1-T2V-1.3B 模型文件存在"
    else
        print_warning "✗ Wan2.1-T2V-1.3B 模型文件不存在"
    fi
}

# 启动应用
start_app() {
    print_info "启动应用..."

    # 激活conda环境
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate $CONDA_ENV

    print_success "已激活conda环境: $CONDA_ENV"
    print_info "启动 $APP_NAME ..."
    print_info "访问地址: http://localhost:$APP_PORT"
    print_info "按 Ctrl+C 停止应用"

    echo -e "${CYAN}================================${NC}"

    # 启动应用
    python3 $APP_SCRIPT
}

# 清理函数
cleanup() {
    print_info "正在清理..."

    # 杀死可能残留的进程
    pkill -f "$APP_SCRIPT" 2>/dev/null || true

    # 清理GPU缓存
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
" 2>/dev/null || true

    print_success "清理完成"
}

# 信号处理
trap cleanup EXIT
trap 'print_info "收到中断信号，正在退出..."; exit 1' INT TERM

# 主函数
main() {
    print_header

    # 执行检查步骤
    check_directory
    check_conda_env
    check_port
    check_gpu
    check_dependencies
    check_models

    echo -e "${CYAN}================================${NC}"
    print_success "所有检查通过，准备启动应用"
    echo -e "${CYAN}================================${NC}"

    # 启动应用
    start_app
}

# 显示帮助信息
show_help() {
    echo "HuMo视频生成器启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -p, --port     指定端口号 (默认: 7860)"
    echo "  -e, --env      指定conda环境名 (默认: py310)"
    echo "  --force-kill   强制杀死占用端口的进程"
    echo ""
    echo "示例:"
    echo "  $0                    # 使用默认设置启动"
    echo "  $0 -p 8080            # 使用端口8080启动"
    echo "  $0 -e myenv           # 使用myenv环境启动"
    echo "  $0 --force-kill       # 强制杀死占用进程并启动"
}

# 解析命令行参数
FORCE_KILL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            APP_PORT="$2"
            shift 2
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --force-kill)
            FORCE_KILL=true
            shift
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果指定了强制杀死选项
if [ "$FORCE_KILL" = true ]; then
    if lsof -Pi :$APP_PORT -sTCP:LISTEN -t >/dev/null ; then
        PID=$(lsof -Pi :$APP_PORT -sTCP:LISTEN -t)
        print_info "强制杀死进程 $PID..."
        kill -9 $PID 2>/dev/null && print_success "进程已杀死"
        sleep 2
    fi
fi

# 运行主函数
main
