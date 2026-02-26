# Travel Diary 部署到 Render（免费版）

## 1) 准备代码仓库

1. 把 `Travel diary` 文件夹上传到 GitHub（新建一个仓库即可）。
2. 确认仓库里包含：
   - `server.js`
   - `package.json`
   - `public/`
   - `render.yaml`
   - `.env.example`

## 2) 在 Render 创建服务

1. 打开 Render，选择 `New +` -> `Blueprint`（会自动识别 `render.yaml`）。
2. 连接你的 GitHub 仓库并创建服务。
3. 部署时保持默认 `Free` 套餐。

## 3) 配置环境变量（重点）

在 Render 的服务设置中添加：

- `GEMINI_API_KEY` = 你的 Gemini API Key
- `DEEPSEEK_API_KEY` = 你的 DeepSeek API Key（可选，配置后文本聊天优先走国内可达接口）

不要把真实密钥写进仓库，也不要提交 `.env` 文件。

## 4) 验证服务是否正常

部署完成后，用 Render 分配的域名访问：

- 首页：`https://你的域名/`
- 健康检查：`https://你的域名/health`

如果 `/health` 返回 `{"ok": true}`，说明服务启动成功。

## 5) 发给朋友测试前的注意事项

- 免费实例会冷启动，首次打开可能较慢。
- 目前数据和上传文件在免费实例上不是强持久化，重启或重新部署可能丢失。

## 6) 本地与线上的运行差异

- 本地使用 `.env`
- 线上使用 Render 环境变量
- 端口由平台注入（代码已支持 `process.env.PORT`）
