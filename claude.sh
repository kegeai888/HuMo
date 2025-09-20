#!/bin/bash

# 检查并安装 @anthropic-ai/claude-code
if ! command -v node &> /dev/null; then
    echo "正在安装 nodejs..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo bash -
sudo apt-get install -y nodejs
else
    echo "claude 已安装，跳过安装"
fi



# 检查并安装 @anthropic-ai/claude-code
if ! command -v claude &> /dev/null; then
    echo "正在安装 @anthropic-ai/claude-code..."
    npm install -g @anthropic-ai/claude-code
else
    echo "claude 已安装，跳过安装"
fi

# 显示版本信息
claude --version

# 设置环境变量

export ANTHROPIC_AUTH_TOKEN="sk-MMO93xLQ3MFtdiZwjfAclearGD7vlkohLlOCdcwd8pk5v025"
export ANTHROPIC_BASE_URL="https://pmpjfbhq.cn-nb1.rainapp.top"

# 创建上级目录（如果不存在）
mkdir -p ~/.claude

mkdir -p ~/.config/claude-code

cat > ~/.config/claude-code/mcp_config.json << 'EOF'
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
EOF

# 以沙盒模式运行并跳过权限检查
IS_SANDBOX=1 claude --dangerously-skip-permissions

