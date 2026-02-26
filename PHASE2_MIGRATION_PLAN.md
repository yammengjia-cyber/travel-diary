# Phase 2 迁移方案（数据库 + 对象存储）

目标：解决演示版的数据易丢失问题，支持长期稳定使用。

## 目标架构

- 应用层：保留现有 `Express` 接口形态
- 数据层：`db.json` -> `Postgres`（建议 Supabase）
- 文件层：`public/uploads` -> 对象存储（建议 Supabase Storage 或 Cloudinary）

## 迁移范围

- `records` 数据（旅行记录）
- `plans` 数据（旅行计划）
- `imagePaths/chibiImagePaths` 文件地址改为外链 URL

## 分步迁移

1. **建表**
   - 创建 `records`、`plans` 两张主表
   - 按需增加 `chat_log`、`keywords`、`character_styles` 等 JSON 字段

2. **抽象数据访问层**
   - 新增 `storage` 适配层（JSON 模式 / DB 模式）
   - 先保证 API 响应结构不变，前端无需重写

3. **图片上传改造**
   - 上传后直接存对象存储
   - 数据库存储公开 URL，而不是本地路径

4. **历史数据迁移脚本**
   - 读取 `db.json`
   - 批量写入数据库
   - 扫描本地 `uploads` 并上传对象存储
   - 回填 URL

5. **回归测试与切换**
   - 核对 `/api/records`、`/api/plans`、聊天相关接口
   - 小流量验证后，再移除 `db.json` 读写依赖

## 建议新增环境变量

- `DATABASE_URL`
- `STORAGE_BUCKET`
- `STORAGE_PUBLIC_BASE_URL`
- `STORAGE_ACCESS_KEY`
- `STORAGE_SECRET_KEY`

## 验收标准

- 服务重启后数据不丢失
- 重新部署后历史图片可访问
- 原有接口路径与前端交互保持兼容
