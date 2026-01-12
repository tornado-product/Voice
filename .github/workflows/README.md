# GitHub Actions 工作流说明

本文档说明 Voice 项目中使用的 GitHub Actions 工作流配置。

## 工作流概览

项目包含三个主要工作流：

1. **CI 工作流** (`ci.yml`) - 持续集成，用于代码检查和测试
2. **Release 工作流** (`release.yml`) - 持续部署，用于创建发布版本和发布到 crates.io
3. **Publish 工作流** (`publish.yml`) - 手动发布到 crates.io

## CI 工作流 (ci.yml)

### 触发条件

- 推送到 `main`、`master` 或 `develop` 分支
- 创建针对上述分支的 Pull Request
- 手动触发（workflow_dispatch）

### 包含的作业

#### 1. 代码格式检查 (fmt)
- **运行平台**: Ubuntu Latest
- **功能**: 使用 `cargo fmt` 检查代码格式
- **失败条件**: 代码格式不符合标准

#### 2. Clippy 静态分析 (clippy)
- **运行平台**: Ubuntu Latest
- **功能**: 使用 `cargo clippy` 进行代码质量检查
- **失败条件**: 发现警告或错误

#### 3. 构建测试
- **Linux 构建** (`build-linux`)
  - 运行平台: Ubuntu Latest
  - 构建 Debug 和 Release 版本
  
- **Windows 构建** (`build-windows`)
  - 运行平台: Windows Latest
  - 构建 Debug 和 Release 版本
  
- **macOS 构建** (`build-macos`)
  - 运行平台: macOS Latest
  - 构建 Debug 和 Release 版本

#### 4. 运行测试 (test)
- **运行平台**: Ubuntu Latest
- **功能**: 
  - 运行所有单元测试
  - 构建示例程序（允许失败）

#### 5. 文档生成 (doc)
- **运行平台**: Ubuntu Latest
- **功能**: 生成 API 文档
- **输出**: 上传文档到 Artifacts（保留 7 天）

#### 6. 安全检查 (security)
- **运行平台**: Ubuntu Latest
- **功能**: 使用 `cargo audit` 检查依赖漏洞
- **失败处理**: 允许失败（仅警告）

#### 7. 依赖更新检查 (outdated)
- **运行平台**: Ubuntu Latest
- **功能**: 使用 `cargo outdated` 检查过时依赖
- **失败处理**: 允许失败（仅信息）

### 缓存策略

所有作业都使用 Cargo 缓存来加速构建：

- 缓存路径：
  - `~/.cargo/bin/`
  - `~/.cargo/registry/index/`
  - `~/.cargo/registry/cache/`
  - `~/.cargo/git/db/`
  - `target/`
- 缓存键：基于 `Cargo.lock` 的哈希值
- 恢复键：使用操作系统前缀的通用缓存

## Release 工作流 (release.yml)

### 触发条件

- 推送版本标签（格式：`v*.*.*`，如 `v1.0.0`）
- 手动触发（workflow_dispatch）

### 包含的作业

#### 1. 创建发布 (create-release)
- **运行平台**: Ubuntu Latest
- **功能**: 
  - 从标签提取版本号
  - 创建 GitHub Release
  - 生成发布说明
- **输出**: 版本号（供后续作业使用）

#### 2. 构建发布版本
- **Linux 发布版** (`build-linux`)
  - 运行平台: Ubuntu Latest
  - 输出: `voice-{version}-x86_64-unknown-linux-gnu.tar.gz`
  
- **Windows 发布版** (`build-windows`)
  - 运行平台: Windows Latest
  - 输出: `voice-{version}-x86_64-pc-windows-msvc.zip`
  
- **macOS 发布版** (`build-macos`)
  - 运行平台: macOS Latest
  - 输出: `voice-{version}-x86_64-apple-darwin.tar.gz`

每个发布包包含：
- 可执行文件（`voice` 或 `voice.exe`）
- README.md
- CHANGELOG.md

#### 3. 发布到 crates.io (publish-crates-io)
- **运行平台**: Ubuntu Latest
- **功能**: 
  - 验证版本号和 Cargo.toml 配置
  - 运行代码检查和测试
  - 发布到 crates.io
- **条件**: 
  - 仅在 `Cargo.toml` 中 `publish != false` 时执行
  - 需要 `CRATES_IO_TOKEN` secret
- **依赖**: 需要 create-release 作业完成

#### 4. 上传到 GitHub Release (upload-release)
- **运行平台**: Ubuntu Latest
- **功能**: 
  - 下载所有平台的发布包
  - 上传到 GitHub Release
  - 显示 crates.io 发布状态
- **依赖**: 需要所有构建作业和发布作业完成

### 创建新版本

#### 方法 1: 使用 Git 标签

```bash
# 1. 更新版本号（在 Cargo.toml 中）
# 2. 提交更改
git add Cargo.toml CHANGELOG.md
git commit -m "Release version 1.0.1"
git push

# 3. 创建并推送标签
git tag -a v1.0.1 -m "Release version 1.0.1"
git push origin v1.0.1
```

#### 方法 2: 使用 GitHub Actions UI

1. 进入 GitHub 仓库的 Actions 页面
2. 选择 "Release" 工作流
3. 点击 "Run workflow"
4. 输入版本号（如：1.0.1）
5. 点击 "Run workflow"

### 发布到 crates.io

Release 工作流会自动尝试发布到 crates.io（如果 `Cargo.toml` 中 `publish != false`）。

**前置条件**：
1. 在 GitHub Secrets 中配置 `CRATES_IO_TOKEN`
   - 获取 Token: https://crates.io/settings/tokens
   - 设置 Secret: 仓库 Settings → Secrets and variables → Actions → New repository secret
2. 确保 `Cargo.toml` 中 `publish != false`（或删除 `publish` 字段）

## Publish 工作流 (publish.yml)

### 触发条件

- 手动触发（workflow_dispatch）

### 功能

用于手动发布到 crates.io，支持：

- **版本验证** - 验证版本号格式和 Cargo.toml 配置
- **代码检查** - 运行 Clippy、测试、格式检查
- **Dry-run 模式** - 可以仅验证而不实际发布
- **安全发布** - 多重验证确保发布质量

### 使用方法

#### 方法 1: 使用 GitHub Actions UI

1. 进入 GitHub 仓库的 Actions 页面
2. 选择 "Publish to crates.io" 工作流
3. 点击 "Run workflow"
4. 输入版本号（如：1.0.1）
5. （可选）勾选 "仅验证，不实际发布" 进行 dry-run
6. 点击 "Run workflow"

#### 方法 2: 本地验证后发布

```bash
# 1. 更新 Cargo.toml 中的版本号
# 2. 本地验证
cargo publish --dry-run

# 3. 如果验证通过，在 GitHub Actions 中运行发布工作流
```

### 前置条件

1. **配置 CRATES_IO_TOKEN**
   - 访问 https://crates.io/settings/tokens
   - 创建新的 API Token（需要 `publish` 权限）
   - 在 GitHub 仓库中添加 Secret：
     - 名称: `CRATES_IO_TOKEN`
     - 值: 你的 crates.io API Token

2. **检查 Cargo.toml**
   - 确保版本号正确
   - 如果不想发布，设置 `publish = false`
   - 确保所有必要的元数据已填写（description、license 等）

### 发布流程

1. **版本验证** - 检查版本号格式和匹配性
2. **配置检查** - 验证 Cargo.toml 配置
3. **代码检查** - 运行 Clippy、测试、格式检查
4. **构建验证** - 确保包可以正常构建
5. **发布** - 执行 `cargo publish`（或 dry-run）

### Dry-run 模式

在发布前，可以使用 dry-run 模式验证：

- 勾选 "仅验证，不实际发布" 选项
- 工作流会运行所有检查，但不会实际发布
- 用于验证配置和代码是否正确

## 环境变量和 Secrets

### 环境变量

工作流使用以下环境变量：

- `CARGO_TERM_COLOR`: 始终启用颜色输出
- `RUST_BACKTRACE`: 设置为 1，用于调试

### GitHub Secrets

发布到 crates.io 需要配置以下 Secret：

- **CRATES_IO_TOKEN**: crates.io API Token
  - 获取方式: https://crates.io/settings/tokens
  - 权限: 需要 `publish` 权限
  - 配置位置: 仓库 Settings → Secrets and variables → Actions

## 依赖安装

工作流会自动安装以下工具：

- Rust 工具链（stable 版本）
- rustfmt（用于格式检查）
- clippy（用于静态分析）
- cargo-audit（用于安全检查，可选）
- cargo-outdated（用于依赖检查，可选）

## 故障排除

### CI 构建失败

1. **格式检查失败**
   ```bash
   cargo fmt --all
   git add .
   git commit -m "Format code"
   ```

2. **Clippy 检查失败**
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings
   # 修复警告后重新提交
   ```

3. **测试失败**
   ```bash
   cargo test --verbose
   # 查看详细错误信息并修复
   ```

### Release 构建失败

1. **检查标签格式**
   - 确保标签格式为 `v*.*.*`（如 `v1.0.0`）
   - 不要使用其他格式（如 `1.0.0` 或 `version-1.0.0`）

2. **检查 Cargo.toml**
   - 确保版本号与标签匹配
   - 确保所有依赖都可用

3. **检查权限**
   - 确保有创建 Release 的权限
   - 检查 GitHub Token 是否有效

### crates.io 发布失败

1. **检查 CRATES_IO_TOKEN**
   - 确保 Secret 已正确配置
   - 验证 Token 是否有效且有 `publish` 权限
   - Token 可以在 https://crates.io/settings/tokens 管理

2. **检查 Cargo.toml**
   - 确保版本号与标签匹配
   - 如果 `publish = false`，发布会被跳过（这是正常的）
   - 确保所有必要的元数据已填写

3. **检查包名可用性**
   - 确保包名在 crates.io 上可用
   - 如果包已存在，确保版本号是新的

4. **检查依赖**
   - 确保所有依赖都可以在 crates.io 上找到
   - 检查是否有私有依赖或 Git 依赖需要特殊处理

## 自定义工作流

### 添加新的检查

在 `ci.yml` 中添加新作业：

```yaml
new-check:
  name: 新检查
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo your-command
```

### 添加新的构建平台

在 `release.yml` 中添加新作业：

```yaml
build-new-platform:
  name: 构建新平台
  needs: create-release
  runs-on: your-runner
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo build --release
    # ... 其他步骤
```

## 最佳实践

1. **提交前检查**
   - 在本地运行 `cargo fmt`、`cargo clippy` 和 `cargo test`
   - 确保所有检查通过后再推送

2. **版本管理**
   - 遵循语义化版本控制（Semantic Versioning）
   - 在 CHANGELOG.md 中记录所有变更

3. **标签管理**
   - 使用带注释的标签（`git tag -a`）
   - 标签信息应该清晰描述版本变更

4. **缓存优化**
   - 充分利用 Cargo 缓存
   - 避免频繁更改 `Cargo.lock`（除非必要）

## 相关资源

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [Rust 工具链文档](https://rust-lang.github.io/rustup/)
- [Cargo 文档](https://doc.rust-lang.org/cargo/)
- [语义化版本控制](https://semver.org/lang/zh-CN/)
