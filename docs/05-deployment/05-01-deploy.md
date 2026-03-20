# 文档部署说明

本文档说明如何将 Nano-vLLM 项目文档部署到 GitHub Pages。

## 📁 文件结构

```
nano-vllm/
├── docs/                           # 文档目录
│   ├── index.md                    # 首页
│   ├── 01-intro.md                 # 项目介绍
│   ├── 02-architecture.md          # 架构概览
│   ├── ...                         # 其他文档
│   ├── includes/                   # 全局包含文件
│   │   └── mkdocs.md
│   └── javascripts/                # JavaScript 文件
│       └── mathjax.js              # MathJax 配置
├── mkdocs.yml                      # MkDocs 配置文件
├── requirements-docs.txt           # 文档依赖
└── .github/
    └── workflows/
        └── deploy-docs.yml         # GitHub Actions 工作流
```

## 🚀 部署方式

### 方式 1：自动部署（推荐）

GitHub Actions 会自动部署文档到 GitHub Pages：

1. **推送代码到 main 分支**
   ```bash
   git add docs/ mkdocs.yml .github/workflows/deploy-docs.yml
   git commit -m "docs: 添加项目文档"
   git push origin main
   ```

2. **等待 GitHub Actions 执行**
   - 访问 https://github.com/Xjg-0216/nano-vllm/actions
   - 查看 "Deploy MkDocs to GitHub Pages" 工作流
   - 等待构建完成（约 2-5 分钟）

3. **访问文档网站**
   - URL: `https://xjg-0216.github.io/nano-vllm/`

### 方式 2：手动部署

如果需要手动部署到本地测试：

#### 1. 安装依赖

```bash
pip install -r requirements-docs.txt
```

#### 2. 本地预览

```bash
# 启动本地服务器
mkdocs serve

# 访问 http://localhost:8000 查看文档
```

#### 3. 构建文档

```bash
# 构建静态文件到 site/ 目录
mkdocs build

# 查看生成的文件
ls -la site/
```

#### 4. 手动部署到 GitHub Pages

```bash
# 使用 ghp-import 工具
pip install ghp-import

# 构建并部署
mkdocs build
ghp-import -n -p -f site

# 或者使用 mkdocs gh-deploy
mkdocs gh-deploy
```

## ⚙️ 配置说明

### mkdocs.yml

主要配置项：

```yaml
site_name: Nano-vLLM 文档
site_url: https://xjg-0216.github.io/nano-vllm/

theme:
  name: material
  language: zh
  # ... 其他主题配置

nav:
  - 首页：index.md
  - 入门指南：
    - 项目介绍：01-intro.md
    # ... 其他导航项

plugins:
  - search
  - mermaid2:        # Mermaid 图表支持
  - git-revision-date-localized:  # 显示最后更新时间
```

### GitHub Actions 工作流

触发条件：
- 推送到 `main` 分支
- 修改 `docs/**`、`mkdocs.yml` 或 `.github/workflows/deploy-docs.yml`
- 手动触发（workflow_dispatch）

部署流程：
1. 检出代码
2. 安装 Python 和依赖
3. 构建 MkDocs 文档
4. 部署到 `gh-pages` 分支

## 🔧 故障排查

### 问题 1：GitHub Pages 404

**原因**：GitHub Pages 未启用或部署失败

**解决方法**：
1. 访问 https://github.com/Xjg-0216/nano-vllm/settings/pages
2. 确保 Source 设置为 "Deploy from a branch"
3. 确保 Branch 设置为 "gh-pages" 和 "/"
4. 检查 GitHub Actions 是否成功执行

### 问题 2：Mermaid 图表不显示

**原因**：mermaid2 插件未安装或配置错误

**解决方法**：
```bash
# 确保安装插件
pip install mkdocs-mermaid2-plugin

# 检查 mkdocs.yml 配置
plugins:
  - mermaid2:
      version: 10.6.1
```

### 问题 3：数学公式不渲染

**原因**：MathJax 配置缺失

**解决方法**：
1. 确保 `docs/javascripts/mathjax.js` 存在
2. 检查 `mkdocs.yml` 中的 `extra_javascript` 配置
3. 使用正确的数学公式语法：
   - 行内公式：`\( E = mc^2 \)`
   - 块级公式：`\[ E = mc^2 \]`

### 问题 4：中文显示乱码

**原因**：文件编码问题

**解决方法**：
```bash
# 确保文件使用 UTF-8 编码
file -i docs/*.md

# 如有问题，转换编码
iconv -f gbk -t utf-8 docs/file.md > docs/file_utf8.md
```

## 📝 添加新文档

1. **在 docs/ 目录创建新文件**
   ```bash
   touch docs/17-new-feature.md
   ```

2. **编辑文件内容**
   ```markdown
   # 新功能说明
   
   这里是新功能的详细说明...
   ```

3. **更新 mkdocs.yml 导航**
   ```yaml
   nav:
     - ...
     - 新功能：17-new-feature.md
   ```

4. **提交并推送**
   ```bash
   git add docs/17-new-feature.md mkdocs.yml
   git commit -m "docs: 添加新功能文档"
   git push origin main
   ```

## 🎨 主题定制

### 修改颜色主题

编辑 `mkdocs.yml`：

```yaml
theme:
  palette:
    - scheme: default  # 浅色模式
      primary: indigo  # 主色调
      accent: indigo   # 强调色
    - scheme: slate    # 深色模式
      primary: indigo
      accent: indigo
```

可用颜色：`red`, `pink`, `purple`, `deep-purple`, `indigo`, `blue`, `light-blue`, `cyan`, `teal`, `green`, `light-green`, `lime`, `yellow`, `amber`, `orange`, `deep-orange`

### 添加自定义 CSS

1. 创建 `docs/stylesheets/extra.css`
2. 在 `mkdocs.yml` 中添加：
   ```yaml
   extra_css:
     - stylesheets/extra.css
   ```

## 📊 查看部署状态

1. **GitHub Actions**
   - 访问：https://github.com/Xjg-0216/nano-vllm/actions
   - 查看 "Deploy MkDocs to GitHub Pages" 工作流

2. **GitHub Pages 设置**
   - 访问：https://github.com/Xjg-0216/nano-vllm/settings/pages
   - 查看部署状态和访问 URL

3. **文档网站**
   - 访问：https://xjg-0216.github.io/nano-vllm/

## 🔗 相关资源

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material for MkDocs 主题](https://squidfunk.github.io/mkdocs-material/)
- [Mermaid 图表文档](https://mermaid.js.org/)
- [GitHub Pages 文档](https://pages.github.com/)

---

**最后更新**: 2024 年
