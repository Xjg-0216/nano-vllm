# GitHub Pages 文档部署说明

## ✅ 已完成的配置

已经为你的 Nano-vLLM 项目配置了完整的 MkDocs 文档系统，可以自动部署到 GitHub Pages。

### 创建的文件

```
nano-vllm/
├── mkdocs.yml                      # MkDocs 配置文件 (Material 主题)
├── requirements-docs.txt           # 文档依赖
├── docs/
│   ├── index.md                    # 文档首页
│   ├── DEPLOY.md                   # 详细部署说明
│   ├── includes/
│   │   └── mkdocs.md               # 全局包含文件
│   └── javascripts/
│       └── mathjax.js              # MathJax 配置
└── .github/
    └── workflows/
        └── deploy-docs.yml         # GitHub Actions 自动部署工作流
```

## 🚀 下一步操作

### 1. 启用 GitHub Pages

访问：https://github.com/Xjg-0216/nano-vllm/settings/pages

设置：
- **Source**: Deploy from a branch
- **Branch**: gh-pages
- **Folder**: /

保存后等待几分钟即可访问文档网站。

### 2. 查看自动部署状态

访问：https://github.com/Xjg-0216/nano-vllm/actions

查看 "Deploy MkDocs to GitHub Pages" 工作流的执行状态。

### 3. 访问文档网站

部署成功后，访问：
```
https://xjg-0216.github.io/nano-vllm/
```

## 📝 本地测试

### 安装依赖

```bash
conda activate base
pip install -r requirements-docs.txt
```

### 本地预览

```bash
mkdocs serve
# 访问 http://localhost:8000
```

### 构建文档

```bash
mkdocs build
# 生成的文件在 site/ 目录
```

## 🔄 自动部署触发条件

以下情况会自动触发部署：

1. **推送到 main 分支**
   ```bash
   git push origin main
   ```

2. **修改文档相关文件**
   - `docs/**` - 任何文档文件
   - `mkdocs.yml` - 配置文件
   - `.github/workflows/deploy-docs.yml` - 工作流文件

3. **手动触发**
   - 访问 Actions 页面
   - 选择 "Deploy MkDocs to GitHub Pages"
   - 点击 "Run workflow"

## 📚 文档导航结构

```
首页
├── 入门指南
│   ├── 项目介绍
│   ├── 架构概览
│   └── 学习顺序
├── Engine 模块
│   ├── Sequence 设计
│   ├── BlockManager 设计
│   ├── BlockManager 流程
│   └── Scheduler 流程
├── 模型架构
│   ├── Qwen3 模型
│   └── 模型注意力
├── 核心组件
│   ├── 张量并行基础
│   ├── 嵌入层与输出头
│   ├── 采样器
│   ├── 注意力机制
│   ├── 并行线性层
│   ├── 并行决策
│   └── RoPE 与 LRU 缓存
└── vLLM 创新
```

## 🎨 主题特性

- ✅ 中文界面
- ✅ 深色/浅色模式切换
- ✅ 响应式设计（支持移动端）
- ✅ 全文搜索
- ✅ Mermaid 图表支持
- ✅ MathJax 数学公式支持
- ✅ 代码复制按钮
- ✅ 最后更新时间显示

## 🔧 常用命令

```bash
# 本地预览
mkdocs serve

# 构建文档
mkdocs build

# 清理构建
mkdocs build --clean

# 检查配置
mkdocs --verbose
```

## 📖 添加新文档

1. 在 `docs/` 目录创建新文件
2. 编辑 `mkdocs.yml` 添加导航项
3. 提交并推送

```bash
git add docs/new-doc.md mkdocs.yml
git commit -m "docs: 添加新文档"
git push origin main
```

## ⚠️ 注意事项

1. **不要提交 `site/` 目录** - 已添加到 `.gitignore`
2. **确保文档使用 UTF-8 编码**
3. **Mermaid 图表使用 mermaid 代码块**
4. **数学公式使用 `\\(` 和 `\\)` 包裹**

## 🔗 相关链接

- [MkDocs 官方文档](https://www.mkdocs.org/)
- [Material 主题文档](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages 文档](https://pages.github.com/)

---

**配置完成时间**: 2024 年
**配置者**: Xingkai Yu
