# GitHub CLI 和 SSH 配置指南

## 1. GitHub CLI 安装状态
✅ **已成功安装 GitHub CLI (gh)**
- 版本：gh version 2.4.0+dfsg1
- 安装时间：2025-09-20

## 2. SSH Key 配置

### 2.1 生成的 SSH Key
- 类型：ED25519 (推荐)
- 位置：`~/.ssh/id_ed25519` (私钥) 和 `~/.ssh/id_ed25519.pub` (公钥)
- 指纹：SHA256:SDH2OBeaj0fuYc9evqzwvdrcwl5/d3/NDEG9/g6zwVM

### 2.2 公钥内容
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICBPospDTYUWlJe/FPVJjf9dc0QgUVjfsEjjIc/mkzhZ github-key
```

## 3. GitHub 配置步骤

### 方法一：通过 GitHub 网页界面
1. 登录 GitHub 账户
2. 进入 Settings → SSH and GPG keys
3. 点击 "New SSH key"
4. 标题填写：`HuMo-Server`
5. 粘贴上面的公钥内容
6. 点击 "Add SSH key"

### 方法二：使用 GitHub CLI (推荐)
```bash
# 1. 使用 token 登录（需要 Personal Access Token）
gh auth login --with-token < your_token.txt

# 2. 或使用网页登录
gh auth login --web

# 3. 验证登录状态
gh auth status

# 4. 添加 SSH key
gh ssh-key add ~/.ssh/id_ed25519.pub --title "HuMo-Server"
```

## 4. 测试 SSH 连接
```bash
# 测试 SSH 连接到 GitHub
ssh -T git@github.com

# 如果连接成功，会看到类似信息：
# Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## 5. 配置 Git 使用 SSH
```bash
# 检查当前仓库的远程 URL
git remote -v

# 如果是 HTTPS，修改为 SSH
git remote set-url origin git@github.com:username/HuMo.git

# 或添加新的远程仓库
git remote add origin git@github.com:username/HuMo.git
```

## 6. 常用的 GitHub CLI 命令
```bash
# 创建新仓库
gh repo create HuMo --public --description "声音克隆生成项目"

# 推送代码到 GitHub
git push -u origin main

# 创建 Pull Request
gh pr create --title "新功能" --body "功能描述"

# 查看仓库信息
gh repo view

# 克隆仓库
gh repo clone username/repository-name
```

## 7. 安全注意事项
- ✅ 私钥文件权限已设置为 600
- ✅ 公钥文件权限已设置为 644
- ✅ SSH 目录权限已设置为 700
- ⚠️ 请妥善保管私钥文件，不要泄露给他人
- ⚠️ 建议定期轮换 SSH keys

## 8. 故障排除
如果遇到连接问题：
```bash
# 1. 检查 SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 2. 详细测试连接
ssh -vT git@github.com

# 3. 检查防火墙/代理设置
```