require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { GoogleGenAI } = require('@google/genai');

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const DB_PATH = path.join(__dirname, 'db.json');
const UPLOADS_DIR = path.join(__dirname, 'public', 'uploads');

// ========== Gemini AI 配置 ==========
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const chatSessions = new Map(); // 会话存储 sessionId -> { chat, history, keywords }
const SESSION_TTL = 15 * 60 * 1000; // 15分钟过期
// 主模型 → 轻量降级 → Gemma 兜底（Gemma 3 支持图片理解，且免费额度独立）
const MODELS = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemma-3-27b-it', 'gemma-3-12b-it'];

// 带自动降级的 Gemini/Gemma 调用
async function callGemini(contents) {
  let lastErr = null;
  for (const model of MODELS) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const response = await genAI.models.generateContent({ model, contents });
        return response;
      } catch (err) {
        lastErr = err;
        if (err.status === 429) {
          if (attempt === 0) {
            console.warn(`[AI] ${model} 频率限制，2秒后重试...`);
            await new Promise(r => setTimeout(r, 2000));
          } else {
            console.warn(`[AI] ${model} 仍然受限，尝试下一个模型...`);
            break; // 跳到下一个模型
          }
        } else {
          console.warn(`[AI] ${model} 错误(${err.status || 'unknown'}): ${(err.message || '').substring(0, 80)}`);
          break; // 非429错误也尝试下一个模型
        }
      }
    }
  }
  throw lastErr || new Error('所有模型均不可用，请稍后再试');
}

// 根据文件扩展名获取正确的 MIME 类型（支持 HEIC 等特殊格式）
const MIME_MAP = {'.png':'image/png','.gif':'image/gif','.webp':'image/webp','.heic':'image/heic','.heif':'image/heif','.bmp':'image/bmp','.tiff':'image/tiff','.tif':'image/tiff'};
function getMimeFromExt(filePath) {
  return MIME_MAP[path.extname(filePath).toLowerCase()] || 'image/jpeg';
}

// 从 AI 回复中移除泄露的 ","keywords":[...] 等 JSON 片段
function stripKeywordsLeak(text) {
  if (!text || typeof text !== 'string') return text || '';
  return text.replace(/["']?\s*,\s*["']?keywords["']?\s*:\s*\[[\s\S]*\]\s*$/i, '').trim();
}

// ========== AI Q版角色图片生成（Gemini 多人检测 + 图片生成 + 去背景） ==========
const sharp = require('sharp');

// 图片生成模型
const IMAGE_MODEL = 'gemini-2.5-flash-image';

// Step 1: 扫描所有照片，识别不同人物（最多3人），返回 [{photoIndex, personLabel, description}]
async function scanPersonsInPhotos(imagePaths) {
  const MAX_PERSONS = 3;
  const validPhotos = []; // {idx, data, mime}

  // 读取所有照片
  for (let i = 0; i < imagePaths.length; i++) {
    const fPath = imagePaths[i].startsWith('/') ? imagePaths[i].slice(1) : imagePaths[i];
    const fullPath = path.join(__dirname, 'public', fPath);
    if (!fs.existsSync(fullPath)) continue;
    const data = fs.readFileSync(fullPath);
    const mime = getMimeFromExt(fullPath);
    validPhotos.push({ idx: i, data, mime, path: imagePaths[i] });
  }
  if (validPhotos.length === 0) return [];

  // 如果只有一张照片，简化流程
  if (validPhotos.length === 1) {
    const p = validPhotos[0];
    const desc = await _describePersons(p.data, p.mime);
    if (!desc || desc.length === 0) return [];
    return desc.slice(0, MAX_PERSONS).map((d, i) => ({ photoIndex: p.idx, photoPath: p.path, personLabel: `person${i + 1}`, description: d }));
  }

  // 多张照片：逐张扫描，合并去重
  const allPersons = [];
  for (const p of validPhotos) {
    if (allPersons.length >= MAX_PERSONS) break;
    const desc = await _describePersons(p.data, p.mime);
    if (!desc || desc.length === 0) continue;
    for (const d of desc) {
      if (allPersons.length >= MAX_PERSONS) break;
      // 简单去重：如果已有描述与新描述前50字相似度高，跳过
      const isDuplicate = allPersons.some(existing => {
        const a = existing.description.substring(0, 60).toLowerCase();
        const b = d.substring(0, 60).toLowerCase();
        const overlap = a.split(' ').filter(w => b.includes(w)).length;
        return overlap > 5;
      });
      if (!isDuplicate) {
        allPersons.push({ photoIndex: p.idx, photoPath: p.path, personLabel: `person${allPersons.length + 1}`, description: d });
      }
    }
    // 避免频率限制
    if (validPhotos.indexOf(p) < validPhotos.length - 1) await new Promise(r => setTimeout(r, 1500));
  }

  return allPersons;
}

// 辅助：分析单张照片中的人物，返回描述数组
async function _describePersons(photoData, photoMime) {
  const prompt = `Analyze this photo carefully.
If there are NO people visible (only scenery/animals/buildings/diagrams), reply with exactly: NO_PERSON

If there ARE people visible, describe EACH person separately. For each person, write one paragraph with these details:
1. Hair: exact style (straight/wavy/curly, length, bangs), exact color
2. Clothing: specific type, exact colors, patterns, accessories
3. Skin tone (fair/light/medium/tan/dark)
4. Distinctive features: glasses, hat, scarf, jewelry, etc.
5. Position in photo (left, right, center)

IMPORTANT: Separate each person's description with the delimiter: ---PERSON---
Maximum 3 people. Be very specific about colors and styles. Output ONLY descriptions, no other text.`;

  try {
    const res = await callGemini([{ role: 'user', parts: [{ text: prompt }, { inlineData: { data: photoData.toString('base64'), mimeType: photoMime } }] }]);
    const text = (res.text || '').trim();
    if (text.includes('NO_PERSON') || text.length < 20) return [];
    // 分割多人描述
    const persons = text.split('---PERSON---').map(s => s.trim()).filter(s => s.length > 20);
    if (persons.length === 0 && text.length > 20) return [text]; // 单人时可能没有分隔符
    return persons;
  } catch (e) {
    console.warn('[Q版扫描] 人物分析失败:', (e.message || '').substring(0, 80));
    return [];
  }
}

// Step 2: 为单个人物生成透明底 chibi 图片（每人不同姿势）
const CHIBI_POSES = [
  'cute standing pose with one hand waving hello, slight head tilt to the right',
  'playful pose with both hands behind back, leaning forward slightly with a wink',
  'cheerful pose doing a peace sign with one hand, other hand on hip',
  'adorable pose holding the hem of their clothes, looking up shyly',
  'confident pose with arms crossed, smiling brightly',
  'sweet pose with one hand touching their cheek, gentle smile',
];

async function generateSingleChibi(photoPath, personDescription, recordId, personIndex) {
  const MAX_RETRIES = 2;

  const fPath = photoPath.startsWith('/') ? photoPath.slice(1) : photoPath;
  const fullPath = path.join(__dirname, 'public', fPath);
  if (!fs.existsSync(fullPath)) return null;

  const photoData = fs.readFileSync(fullPath);
  const photoMime = getMimeFromExt(fullPath);

  // 为每个角色选择不同的姿势
  const poseIdx = (personIndex - 1 + parseInt(recordId) % CHIBI_POSES.length) % CHIBI_POSES.length;
  const pose = CHIBI_POSES[poseIdx];

  // Prompt: 纯白背景、水彩风格、只画指定的人、独特姿势
  const prompt = `Look at this photo. I need you to create a cute chibi (Q-version) character based on THIS SPECIFIC PERSON: ${personDescription}

CRITICAL REQUIREMENTS:
- PURE WHITE background (#FFFFFF), absolutely nothing else in the background
- Soft watercolor painting style with gentle brush strokes and smooth color blending
- NO hard outlines or sharp edges, soft color boundaries
- 2-head body proportion (oversized cute head, small body)
- Big round sparkly anime eyes with highlights
- Precisely match this person's: hair style & color, clothing colors & style, skin tone, accessories
- POSE: ${pose}
- ONLY this ONE character, no other elements, no text, no shadow, no decorations
- Full body visible head to toe, centered in frame
- Professional quality, clear and sharp, no ghosting or blur`;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`[Q版图片] person${personIndex} 第${attempt + 1}次生成...`);
      const response = await genAI.models.generateContent({
        model: IMAGE_MODEL,
        contents: [{ role: 'user', parts: [
          { text: prompt },
          { inlineData: { data: photoData.toString('base64'), mimeType: photoMime } }
        ]}],
        config: { responseModalities: ['TEXT', 'IMAGE'] }
      });

      if (!response || !response.candidates || !response.candidates[0] || !response.candidates[0].content) {
        console.warn('[Q版图片] 无候选结果'); continue;
      }

      const parts = response.candidates[0].content.parts || [];
      for (const part of parts) {
        if (part.inlineData && part.inlineData.data) {
          const rawBuffer = Buffer.from(part.inlineData.data, 'base64');
          if (rawBuffer.length < 5000) continue;

          // 去除白色背景，输出透明底 PNG
          const transparentBuffer = await removeWhiteBackground(rawBuffer);

          const filename = `chibi-${recordId}-${personIndex}.png`;
          const savePath = path.join(UPLOADS_DIR, filename);
          fs.writeFileSync(savePath, transparentBuffer);
          console.log(`[Q版图片] 生成成功: ${filename} (${(transparentBuffer.length / 1024).toFixed(1)} KB)`);
          return `/uploads/${filename}`;
        }
      }
      const textParts = parts.filter(p => p.text).map(p => p.text).join('');
      console.warn('[Q版图片] 无图片数据:', textParts.substring(0, 80));
    } catch (e) {
      console.warn(`[Q版图片] person${personIndex} 失败:`, (e.message || '').substring(0, 80));
      if (e.status === 429) await new Promise(r => setTimeout(r, 4000));
    }
    if (attempt < MAX_RETRIES) await new Promise(r => setTimeout(r, 3000));
  }
  return null;
}

// Step 3: 白色背景去除 → 透明 PNG（边缘洪泛填充法，只去掉外围白色，保留角色内部浅色）
async function removeWhiteBackground(imgBuffer) {
  try {
    const image = sharp(imgBuffer).ensureAlpha();
    const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
    const { width, height, channels } = info;

    // 判断像素是否为"白色/近白色"
    const WHITE_THRESHOLD = 42; // 与纯白的欧氏距离阈值
    function isWhitish(idx) {
      const r = data[idx], g = data[idx + 1], b = data[idx + 2];
      const dist = Math.sqrt((255 - r) ** 2 + (255 - g) ** 2 + (255 - b) ** 2);
      return dist < WHITE_THRESHOLD;
    }

    // 标记数组：0=未访问，1=是背景白色，2=已确认非背景
    const mask = new Uint8Array(width * height); // 默认0

    // 从图像四条边缘开始洪泛填充（BFS）
    const queue = [];
    // 上下两条边
    for (let x = 0; x < width; x++) {
      queue.push(x); // 第一行
      queue.push((height - 1) * width + x); // 最后一行
    }
    // 左右两条边
    for (let y = 0; y < height; y++) {
      queue.push(y * width); // 第一列
      queue.push(y * width + width - 1); // 最后一列
    }

    // BFS 洪泛：从边缘向内扩展，只在白色像素间传播
    let head = 0;
    while (head < queue.length) {
      const pos = queue[head++];
      if (pos < 0 || pos >= width * height) continue;
      if (mask[pos] !== 0) continue; // 已处理过

      const pixIdx = pos * channels;
      if (!isWhitish(pixIdx)) {
        mask[pos] = 2; // 非白色，不是背景
        continue;
      }

      mask[pos] = 1; // 标记为背景白色
      const x = pos % width, y = Math.floor(pos / width);
      // 4邻域扩展
      if (x > 0 && mask[pos - 1] === 0) queue.push(pos - 1);
      if (x < width - 1 && mask[pos + 1] === 0) queue.push(pos + 1);
      if (y > 0 && mask[pos - width] === 0) queue.push(pos - width);
      if (y < height - 1 && mask[pos + width] === 0) queue.push(pos + width);
    }

    // 应用透明度：只对标记为背景(1)的像素做透明处理
    for (let pos = 0; pos < width * height; pos++) {
      const pixIdx = pos * channels;
      if (mask[pos] === 1) {
        // 是从边缘连通的背景白色 → 完全透明
        data[pixIdx + 3] = 0;
      } else if (mask[pos] === 0 || mask[pos] === 2) {
        // 角色内部（包括白色部分）→ 保持完全不透明
        // 不做任何修改
      }
    }

    // 对背景与角色的边界做1px柔和过渡（抗锯齿）
    for (let pos = 0; pos < width * height; pos++) {
      if (mask[pos] !== 1) continue; // 只处理背景像素
      const x = pos % width, y = Math.floor(pos / width);
      // 检查是否紧邻非背景像素
      const neighbors = [];
      if (x > 0) neighbors.push(pos - 1);
      if (x < width - 1) neighbors.push(pos + 1);
      if (y > 0) neighbors.push(pos - width);
      if (y < height - 1) neighbors.push(pos + width);
      const hasNonBg = neighbors.some(n => mask[n] !== 1);
      if (hasNonBg) {
        // 边界像素：半透明过渡
        data[pos * channels + 3] = 60;
      }
    }

    // 裁切透明边缘 + 统一高度，让所有角色大小一致
    const trimmed = await sharp(data, { raw: { width, height, channels } })
      .png()
      .trim({ threshold: 5 }) // 裁掉四周透明/近透明像素
      .toBuffer();
    // 统一高度为 800px（保持宽高比），确保角色显示大小一致
    const resized = await sharp(trimmed)
      .resize({ height: 800, fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
      .png()
      .toBuffer();
    return resized;
  } catch (e) {
    console.warn('[去背景] 处理失败，返回原图:', (e.message || '').substring(0, 60));
    return imgBuffer;
  }
}

// 组合入口：为一条记录生成所有 chibi 角色
async function generateAllChibis(imagePaths, recordId) {
  console.log(`[Q版] 开始处理记录 ${recordId}，共 ${imagePaths.length} 张照片`);

  // Step 1: 扫描所有照片中的人物
  const persons = await scanPersonsInPhotos(imagePaths);
  if (persons.length === 0) {
    console.log('[Q版] 未检测到人物，跳过');
    return [];
  }
  console.log(`[Q版] 检测到 ${persons.length} 个不同人物`);

  // Step 2: 为每个人物生成透明底 chibi
  const results = [];
  for (let i = 0; i < persons.length; i++) {
    const p = persons[i];
    console.log(`[Q版] 生成角色 ${i + 1}/${persons.length}: ${p.description.substring(0, 60)}...`);
    const imgPath = await generateSingleChibi(p.photoPath, p.description, recordId, i + 1);
    if (imgPath) results.push(imgPath);
    // 间隔避免频率限制
    if (i < persons.length - 1) await new Promise(r => setTimeout(r, 3000));
  }

  return results;
}

// 定期清理过期会话
setInterval(() => {
  const now = Date.now();
  for (const [id, session] of chatSessions) {
    if (now - session.lastActive > SESSION_TTL) chatSessions.delete(id);
  }
}, 60 * 1000);

// 确保目录存在
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR, { recursive: true });

// ========== Multer 配置：支持多图上传（最多20张） ==========
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + '-' + Math.round(Math.random() * 1e9) + path.extname(file.originalname);
    cb(null, uniqueName);
  }
});
const upload = multer({
  storage,
  limits: { fileSize: 30 * 1024 * 1024 }, // 30MB，适配高分辨率手机照片
  fileFilter: (req, file, cb) => {
    const allowed = /\.(jpg|jpeg|png|gif|webp|heic|heif|bmp|tiff|tif)$/i;
    if (allowed.test(file.originalname)) cb(null, true);
    else cb(new Error('不支持的图片格式：' + path.extname(file.originalname)));
  }
});

// ========== 中间件 ==========
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// 云平台健康检查接口
app.get('/health', (req, res) => {
  res.status(200).json({ ok: true });
});

// ========== HEIC 转 JPEG 服务端接口（macOS sips 转换，带缓存） ==========
const { execSync } = require('child_process');
app.get('/api/heic-preview/:filename', (req, res) => {
  const filename = req.params.filename;
  if (!/\.(heic|heif)$/i.test(filename)) return res.status(400).send('Not a HEIC file');
  const heicPath = path.join(UPLOADS_DIR, filename);
  if (!fs.existsSync(heicPath)) return res.status(404).send('File not found');
  // 缓存：如果已有转换后的 JPEG 就直接返回
  const jpgName = filename.replace(/\.(heic|heif)$/i, '_heic.jpg');
  const jpgPath = path.join(UPLOADS_DIR, jpgName);
  if (!fs.existsSync(jpgPath)) {
    try {
      execSync(`sips -s format jpeg "${heicPath}" --out "${jpgPath}"`, { timeout: 10000, stdio: 'pipe' });
    } catch (e) {
      console.warn('[HEIC] 转换失败:', filename, (e.message || '').substring(0, 60));
      return res.status(500).send('HEIC conversion failed');
    }
  }
  res.setHeader('Content-Type', 'image/jpeg');
  res.setHeader('Cache-Control', 'public, max-age=86400');
  res.sendFile(jpgPath);
});

// ========== GeoJSON 文件 ==========
app.get('/ne_110m_admin_0_countries.geojson', (req, res) => {
  const geojsonPath = path.join(__dirname, 'ne_110m_admin_0_countries.geojson');
  if (fs.existsSync(geojsonPath)) res.sendFile(geojsonPath);
  else res.status(404).json({ error: 'GeoJSON 文件不存在' });
});

// ========== 逆地理编码（经纬度→国家+城市，用于手动选位） ==========
app.get('/api/reverse-geocode', async (req, res) => {
  const lat = parseFloat(req.query.lat);
  const lon = parseFloat(req.query.lon);
  if (isNaN(lat) || isNaN(lon)) return res.status(400).json({ success: false });
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json&accept-language=zh`;
    const r = await fetch(url, { headers: { 'User-Agent': 'TravelDiary/1.0' } });
    const data = await r.json();
    if (!data || !data.address) return res.json({ success: false });
    const ad = data.address;
    const displayParts = (data.display_name || '').split(',').map(s => s.trim());
    const country = (ad.country || '').split(';')[0].split('/')[0].trim(); // 处理"法国;法國"格式

    // 智能提取城市名（只到城市级别，不要乡镇/街道/区等更小单位）
    let city = '';
    const cc = ad.country_code || '';
    if (['cn', 'tw', 'hk', 'mo'].includes(cc)) {
      // 中国/港澳台：ad.city 通常是区名，从 display_name 中提取"xx市"
      const cityMatch = displayParts.find(p => /市$/.test(p.trim()));
      city = cityMatch ? cityMatch.trim() : (ad.city || '');
    } else if (cc === 'jp') {
      // 日本：从 display_name 中提取"xx都/府/市"（如"东京都"、"大阪市"）
      const jpMatch = displayParts.find(p => /[都府市]$/.test(p.trim()) || /都\//.test(p.trim()));
      city = jpMatch ? jpMatch.trim().split('/')[0] : (ad.city || '');
    } else if (cc === 'kr') {
      // 韩国：从 display_name 中提取"xx시/광역시/특별시"或使用 city 字段
      const krMatch = displayParts.find(p => /[시도]$/.test(p.trim()));
      city = krMatch ? krMatch.trim() : (ad.city || '');
    } else {
      // 其他国家：直接取 city 字段（Nominatim 对欧美城市通常准确）
      city = ad.city || ad.municipality || ad.county || '';
    }

    // 组合显示：城市 · 国家（去重）
    const parts = [city, country].filter(Boolean);
    const displayName = [...new Set(parts)].join(' · ');
    const state = (ad.state || ad.region || '').split('/')[0].trim();
    res.json({
      success: true,
      city,
      state,
      country,
      displayName: displayName || state || country || '未知位置'
    });
  } catch (err) {
    res.status(500).json({ success: false });
  }
});

// ========== 城市地理编码（用于搜索定位） ==========
app.get('/api/geocode', async (req, res) => {
  const query = (req.query.query || '').toString().trim();
  if (!query) return res.status(400).json({ success: false, error: '缺少 query' });
  try {
    const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query)}&count=10&language=zh&format=json`;
    const r = await fetch(url);
    const data = await r.json();
    if (!data || !data.results || data.results.length === 0) {
      return res.json({ success: false, error: '未找到城市' });
    }
    // 优先选人口最多或名称最匹配的（南京、Nanjing 等）
    const q = query.toLowerCase().replace(/\s/g, '');
    const first = data.results.sort((a, b) => {
      const nameA = (a.name || '').toLowerCase();
      const nameB = (b.name || '').toLowerCase();
      const matchA = nameA.includes(q) || q.includes(nameA) ? 1 : 0;
      const matchB = nameB.includes(q) || q.includes(nameB) ? 1 : 0;
      if (matchA !== matchB) return matchB - matchA;
      return (b.population || 0) - (a.population || 0);
    })[0];
    res.json({
      success: true,
      location: {
        name: first.name,
        country: first.country,
        latitude: first.latitude,
        longitude: first.longitude
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, error: '地理编码失败' });
  }
});

// ========== 上传旅行记录图片（支持多图） ==========
app.post('/api/upload', (req, res) => {
  upload.array('images', 20)(req, res, (err) => {
    if (err) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(400).json({ error: '图片太大（单张最大30MB），请压缩后重试' });
      }
      return res.status(400).json({ error: '上传失败：' + err.message });
    }
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: '请选择图片' });
    const paths = req.files.map(f => '/uploads/' + f.filename);
    res.json({ success: true, paths });
  });
});

// ========== 读取旅行记录 ==========
app.get('/api/records', (req, res) => {
  try {
    const data = fs.readFileSync(DB_PATH, 'utf-8');
    const json = JSON.parse(data);
    res.json(json.records || []);
  } catch (err) {
    if (err.code === 'ENOENT') res.json([]);
    else res.status(500).json({ error: '读取失败' });
  }
});

// ========== 保存旅行记录（自动获取天气） ==========
app.post('/api/records', async (req, res) => {
  const record = req.body;
  if (record.lat == null || record.lon == null || !record.description) {
    return res.status(400).json({ error: '缺少 lat、lon 或 description' });
  }
  try {
    // 获取天气：如果有拍摄日期则查历史天气，否则查当前天气
    try {
      if (record.photoDate) {
        // 尝试查询历史天气（Open-Meteo Archive API）
        let gotHistorical = false;
        try {
          const archiveUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${record.lat}&longitude=${record.lon}&start_date=${record.photoDate}&end_date=${record.photoDate}&daily=temperature_2m_mean,weathercode,windspeed_10m_max&timezone=auto`;
          const archiveRes = await fetch(archiveUrl);
          const archiveData = await archiveRes.json();
          if (archiveData.daily && archiveData.daily.temperature_2m_mean && archiveData.daily.temperature_2m_mean[0] != null) {
            record.weather = {
              temperature: archiveData.daily.temperature_2m_mean[0],
              weathercode: archiveData.daily.weathercode ? archiveData.daily.weathercode[0] : 0,
              windspeed: archiveData.daily.windspeed_10m_max ? archiveData.daily.windspeed_10m_max[0] : 0
            };
            gotHistorical = true;
          }
        } catch (e) {
          console.warn('历史天气查询失败:', e.message);
        }
        // 如果历史天气查不到（超出范围），根据月份和纬度推测
        if (!gotHistorical) {
          const month = parseInt(record.photoDate.split('-')[1]);
          const isNorth = record.lat >= 0;
          let season, tempEst, codeEst;
          // 根据月份和半球判断季节
          if (isNorth) {
            if (month >= 12 || month <= 2) { season = '冬季'; tempEst = -2; codeEst = 71; }
            else if (month >= 3 && month <= 5) { season = '春季'; tempEst = 15; codeEst = 2; }
            else if (month >= 6 && month <= 8) { season = '夏季'; tempEst = 28; codeEst = 1; }
            else { season = '秋季'; tempEst = 12; codeEst = 2; }
          } else {
            if (month >= 12 || month <= 2) { season = '夏季'; tempEst = 25; codeEst = 1; }
            else if (month >= 3 && month <= 5) { season = '秋季'; tempEst = 15; codeEst = 2; }
            else if (month >= 6 && month <= 8) { season = '冬季'; tempEst = 5; codeEst = 3; }
            else { season = '春季'; tempEst = 18; codeEst = 2; }
          }
          // 纬度越高温度越低
          const latAbs = Math.abs(record.lat);
          if (latAbs > 50) tempEst -= 8;
          else if (latAbs > 35) tempEst -= 3;
          // 赤道附近温差小
          if (latAbs < 15) tempEst = 26;
          record.weather = {
            temperature: tempEst,
            weathercode: codeEst,
            windspeed: 10,
            estimated: true,
            season
          };
        }
      } else {
        // 无拍摄日期：获取当前天气
        const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${record.lat}&longitude=${record.lon}&current_weather=true`;
        const weatherRes = await fetch(weatherUrl);
        const weatherData = await weatherRes.json();
        if (weatherData.current_weather) {
          record.weather = {
            temperature: weatherData.current_weather.temperature,
            weathercode: weatherData.current_weather.weathercode,
            windspeed: weatherData.current_weather.windspeed
          };
        }
      }
    } catch (e) {
      console.warn('天气获取失败:', e.message);
    }

    let db = { records: [] };
    if (fs.existsSync(DB_PATH)) {
      db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
    }
    record.id = Date.now().toString();
    record.createdAt = new Date().toISOString();
    // 同步保存聊天数据（关键词 + 聊天记录），但 AI 总结放后台
    let pendingSession = null;
    if (record.chatSessionId) {
      const session = chatSessions.get(record.chatSessionId);
      if (session) {
        record.keywords = session.keywords || [];
        record.chatLog = session.chatLog || [];
        pendingSession = session; // 留给后台生成总结
        chatSessions.delete(record.chatSessionId);
      }
      delete record.chatSessionId;
    }
    db.records.push(record);
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    // 立即返回响应，不等待 AI
    res.json({ success: true, record });

    // ---- 后台异步：人物特征提取 ----
    const imagePaths = record.imagePaths || (record.imagePath ? [record.imagePath] : []);
    if (imagePaths.length > 0) {
      (async () => {
        try {
          const firstPath = imagePaths[0].startsWith('/') ? imagePaths[0].slice(1) : imagePaths[0];
          const fullPath = path.join(__dirname, 'public', firstPath);
          if (fs.existsSync(fullPath)) {
            const data = fs.readFileSync(fullPath);
            const mime = getMimeFromExt(fullPath);
            console.log('[Q版] 后台人物特征提取，图片:', firstPath);
            const charPrompt = `看这张照片。重要：如是纯风景（无人、只有景色/动物/建筑），必须返回{"characters":[]}。
如有1-3个清晰可见的人物，为每人提取：1)上衣主色hex 2)下装主色hex 3)头发short|medium|long 4)发色hex如#3d2c1e 5)肤色hint:light|medium|warm。最多3人。
返回JSON：{"characters":[{"topColor":"#hex","bottomColor":"#hex","hair":"short|medium|long","hairColor":"#hex","skin":"light|medium|warm"},...]}。纯风景或无人必须{"characters":[]}。`;
            const charRes = await callGemini([{ role: 'user', parts: [{ text: charPrompt }, { inlineData: { data: data.toString('base64'), mimeType: mime } }] }]);
            const charText = (charRes.text || '').replace(/```/g, '').replace(/```json/g, '').trim();
            const jsonMatch = charText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              const parsed = JSON.parse(jsonMatch[0]);
              const arr = parsed.characters || (parsed.noPerson ? [] : [parsed]);
              const list = Array.isArray(arr) ? arr.slice(0, 3) : [];
              // 异步回写 db.json
              const curDb = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
              const rec = curDb.records.find(r => r.id === record.id);
              if (rec) { rec.characterStyles = list; fs.writeFileSync(DB_PATH, JSON.stringify(curDb, null, 2)); }
              console.log('[Q版] 特征提取完成:', list.length, '个角色');
            }
          }
        } catch (e) {
          console.warn('[Q版] 后台特征提取失败:', e.message);
        }
      })();
    }

    // ---- 后台异步：AI 总结生成 ----
    if (pendingSession && pendingSession.chatLog && pendingSession.chatLog.length > 0) {
      (async () => {
        try {
          const chatContent = pendingSession.chatLog.map(m => `${m.role === 'ai' ? 'AI' : '用户'}：${m.text}`).join('\n');
          const summaryPrompt = `根据以下旅行对话内容，写一段温暖、有画面感的旅行日记（80-150字），用第一人称"我"来写，像散文一样优美，不需要标题，不要使用**星号标记：\n\n${chatContent}`;
          const summaryRes = await callGemini([{ role: 'user', parts: [{ text: summaryPrompt }] }]);
          const summaryText = (summaryRes.text || '').replace(/```/g, '').replace(/\*\*/g, '').trim();
          if (summaryText) {
            const curDb = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
            const rec = curDb.records.find(r => r.id === record.id);
            if (rec) { rec.aiSummary = summaryText; fs.writeFileSync(DB_PATH, JSON.stringify(curDb, null, 2)); }
            console.log('[保存] AI总结已后台生成');
          }
        } catch (e) {
          console.warn('[保存] 后台AI总结失败:', e.message);
        }
      })();
    }

    // ---- 后台异步：为有人物的照片生成 AI Q版角色透明底图片 ----
    const imgs = record.imagePaths || [];
    if (imgs.length > 0) {
      generateAllChibis(imgs, record.id).then(imagePaths => {
        if (imagePaths.length > 0) {
          try {
            const freshDb = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
            const rec = freshDb.records.find(r => r.id === record.id);
            if (rec) {
              rec.chibiImagePaths = imagePaths;
              rec.chibiImagePath = imagePaths[0]; // 兼容旧字段
              fs.writeFileSync(DB_PATH, JSON.stringify(freshDb, null, 2));
              console.log('[Q版图片] 已更新到记录:', record.id, '共', imagePaths.length, '个角色');
            }
          } catch (e) { console.warn('[Q版图片] 更新记录失败:', e.message); }
        }
      }).catch(e => console.warn('[Q版图片] 后台生成失败:', e.message));
    }
  } catch (err) {
    res.status(500).json({ error: '保存失败' });
  }
});

// ========== 批量生成 Q版角色透明底图片 ==========
app.post('/api/records/refresh-characters', async (req, res) => {
  try {
    let db = { records: [] };
    if (fs.existsSync(DB_PATH)) db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
    let updated = 0;
    for (const rec of db.records) {
      const imgs = rec.imagePaths || (rec.imagePath ? [rec.imagePath] : []);
      if (imgs.length === 0) continue;
      // 跳过已有新版多人图片的记录
      if (rec.chibiImagePaths && rec.chibiImagePaths.length > 0) continue;
      // 跳过已确认为纯风景的照片
      if (rec.characterStyles && rec.characterStyles.length === 0) continue;
      try {
        console.log('[Q版补提] 处理记录:', rec.id);
        const imagePaths = await generateAllChibis(imgs, rec.id);
        if (imagePaths.length > 0) {
          rec.chibiImagePaths = imagePaths;
          rec.chibiImagePath = imagePaths[0]; // 兼容旧字段
          updated++;
          console.log('[Q版补提]', rec.id, '→', imagePaths.length, '个角色已生成');
        }
        // 间隔5秒避免频率限制
        await new Promise(r => setTimeout(r, 5000));
      } catch (e) {
        console.warn('[Q版补提] 失败:', rec.id, (e.message || '').substring(0, 80));
      }
    }
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    res.json({ success: true, updated });
  } catch (err) {
    res.status(500).json({ error: '批量更新失败' });
  }
});

// ========== 更新记录/计划的聊天数据（继续聊天后保存） ==========
app.patch('/api/records/:id/chat', async (req, res) => {
  const { chatSessionId } = req.body;
  try {
    let db = { records: [] };
    if (fs.existsSync(DB_PATH)) db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
    const rec = db.records.find(r => r.id === req.params.id);
    if (!rec) return res.status(404).json({ error: '记录不存在' });
    if (chatSessionId) {
      const session = chatSessions.get(chatSessionId);
      if (session) {
        rec.keywords = session.keywords || rec.keywords || [];
        rec.chatLog = session.chatLog || rec.chatLog || [];
        // 重新生成AI总结
        try {
          const chatContent = session.chatLog.map(m => `${m.role === 'ai' ? 'AI' : '用户'}：${m.text}`).join('\n');
          const summaryPrompt = `根据以下旅行对话内容，写一段温暖、有画面感的旅行日记（80-150字），用第一人称"我"来写，不要使用**星号标记：\n\n${chatContent}`;
          const summaryRes = await callGemini([{ role: 'user', parts: [{ text: summaryPrompt }] }]);
          const summaryText = (summaryRes.text || '').replace(/```/g, '').replace(/\*\*/g, '').trim();
          if (summaryText) rec.aiSummary = summaryText;
        } catch (e) { console.warn('AI总结更新失败:', e.message); }
        chatSessions.delete(chatSessionId);
      }
    }
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    res.json({ success: true, record: rec });
  } catch (err) { res.status(500).json({ error: '更新失败' }); }
});

app.patch('/api/plans/:id/chat', async (req, res) => {
  const { chatSessionId } = req.body;
  try {
    let db = { records: [], plans: [] };
    if (fs.existsSync(DB_PATH)) db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
    if (!db.plans) db.plans = [];
    const plan = db.plans.find(p => p.id === req.params.id);
    if (!plan) return res.status(404).json({ error: '计划不存在' });
    if (chatSessionId) {
      const session = chatSessions.get(chatSessionId);
      if (session) {
        plan.keywords = session.keywords || plan.keywords || [];
        plan.chatLog = session.chatLog || plan.chatLog || [];
        try {
          const chatContent = session.chatLog.map(m => `${m.role === 'ai' ? 'AI' : '用户'}：${m.text}`).join('\n');
          const summaryPrompt = `根据以下旅行规划对话，写一段简洁的旅行计划摘要（80-120字），不要使用**星号标记：\n\n${chatContent}`;
          const summaryRes = await callGemini([{ role: 'user', parts: [{ text: summaryPrompt }] }]);
          const summaryText = (summaryRes.text || '').replace(/```/g, '').replace(/\*\*/g, '').trim();
          if (summaryText) plan.aiSummary = summaryText;
        } catch (e) { console.warn('计划AI总结更新失败:', e.message); }
        chatSessions.delete(chatSessionId);
      }
    }
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    res.json({ success: true, plan });
  } catch (err) { res.status(500).json({ error: '更新失败' }); }
});

// ========== 删除旅行记录 ==========
app.delete('/api/records/:id', (req, res) => {
  try {
    let db = { records: [] };
    if (fs.existsSync(DB_PATH)) {
      db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
    }
    const idx = db.records.findIndex(r => r.id === req.params.id);
    if (idx === -1) return res.status(404).json({ error: '记录不存在' });
    db.records.splice(idx, 1);
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: '删除失败' });
  }
});

// ========== 读取旅行计划 ==========
app.get('/api/plans', (req, res) => {
  try {
    const data = fs.readFileSync(DB_PATH, 'utf-8');
    const json = JSON.parse(data);
    res.json(json.plans || []);
  } catch (err) {
    if (err.code === 'ENOENT') res.json([]);
    else res.status(500).json({ error: '读取失败' });
  }
});

// ========== 保存旅行计划（含AI聊天数据） ==========
app.post('/api/plans', async (req, res) => {
  const plan = req.body;
  if (plan.lat == null || plan.lon == null || !plan.description) {
    return res.status(400).json({ error: '缺少 lat、lon 或 description' });
  }
  try {
    let db = { records: [], plans: [] };
    if (fs.existsSync(DB_PATH)) {
      db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
      if (!db.plans) db.plans = [];
    }
    plan.id = Date.now().toString();
    plan.createdAt = new Date().toISOString();

    // 同步保存聊天数据（关键词 + 聊天记录），AI 总结放后台
    let pendingPlanSession = null;
    if (plan.chatSessionId) {
      const session = chatSessions.get(plan.chatSessionId);
      if (session) {
        plan.keywords = session.keywords || [];
        plan.chatLog = session.chatLog || [];
        pendingPlanSession = session;
        chatSessions.delete(plan.chatSessionId);
      }
      delete plan.chatSessionId;
    }

    db.plans.push(plan);
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    // 立即返回响应，不等待 AI
    res.json({ success: true, plan });

    // ---- 后台异步：AI 计划总结 ----
    if (pendingPlanSession && pendingPlanSession.chatLog && pendingPlanSession.chatLog.length > 0) {
      (async () => {
        try {
          const chatContent = pendingPlanSession.chatLog.map(m => `${m.role === 'ai' ? 'AI' : '用户'}：${m.text}`).join('\n');
          const summaryPrompt = `根据以下旅行规划对话，写一段简洁的旅行计划摘要（80-120字），包含目的地亮点、推荐时间、关键建议，像旅行手册一样实用：\n\n${chatContent}`;
          const summaryRes = await callGemini([{ role: 'user', parts: [{ text: summaryPrompt }] }]);
          const summaryText = (summaryRes.text || '').replace(/```/g, '').replace(/\*\*/g, '').trim();
          if (summaryText) {
            const curDb = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
            const rec = (curDb.plans || []).find(p => p.id === plan.id);
            if (rec) { rec.aiSummary = summaryText; fs.writeFileSync(DB_PATH, JSON.stringify(curDb, null, 2)); }
            console.log('[计划] AI总结已后台生成');
          }
        } catch (e) {
          console.warn('[计划] 后台AI总结失败:', e.message);
        }
      })();
    }
  } catch (err) {
    res.status(500).json({ error: '保存失败' });
  }
});

// ========== 删除旅行计划 ==========
app.delete('/api/plans/:id', (req, res) => {
  try {
    let db = { records: [], plans: [] };
    if (fs.existsSync(DB_PATH)) {
      db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
      if (!db.plans) db.plans = [];
    }
    const idx = db.plans.findIndex(p => p.id === req.params.id);
    if (idx === -1) return res.status(404).json({ error: '计划不存在' });
    db.plans.splice(idx, 1);
    fs.writeFileSync(DB_PATH, JSON.stringify(db, null, 2));
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: '删除失败' });
  }
});

// ========== AI 聊天：开始对话（发送图片+位置信息给 Gemini） ==========
app.post('/api/chat/start', async (req, res) => {
  const { imagePaths, location } = req.body; // imagePaths: ['/uploads/xxx.jpg', ...], location: '中国杭州'
  try {
    const sessionId = Date.now().toString() + '-' + Math.random().toString(36).slice(2, 8);

    // 构建图片 parts（读取图片文件并转为 base64）
    const imageParts = [];
    if (imagePaths && imagePaths.length > 0) {
      for (const imgPath of imagePaths.slice(0, 3)) { // 最多发3张图片节省额度
        const fullPath = path.join(__dirname, 'public', imgPath);
        if (fs.existsSync(fullPath)) {
          const data = fs.readFileSync(fullPath);
          const mime = getMimeFromExt(fullPath);
          imageParts.push({
            inlineData: { data: data.toString('base64'), mimeType: mime }
          });
        }
      }
    }

    const systemPrompt = `你是「拾光鹿」，一位温暖又有见识的旅行回忆助手。你的风格是像一个靠谱的朋友——温和、真诚、偶尔幽默，但不会过度卖萌或夸张。

职责：帮用户回忆旅行故事，也能回答旅行相关的知识问题。

规则：
1. 用户提问时认真回答，可以穿插一两个有趣的小知识（50字内）
2. 用户分享回忆时，简洁温暖地回应，然后问一个有深度的递进问题（30字内）
3. 仔细观察照片中的细节来互动
4. 语气自然真诚，不要用太多感叹号和emoji，偶尔一个就好
5. 绝对不要使用 ** 星号标记，用自然的语言组织重点，需要分段时用换行符
6. 提取关键词
7. 回复格式必须是 JSON：{"reply": "你的回复文字", "keywords": [{"text":"关键词","type":"类型"}]}
8. 关键词简短（2-4字），type：scene/place/feature/mood/other
9. 关键词累积不重复
10. reply 中用 \\n 来分段，让信息层次清晰`;

    const locationInfo = location ? `用户当前在：${location}。` : '';
    const userFirstMsg = locationInfo + (imageParts.length > 0
      ? '这是我旅行时拍的照片，帮我回忆一下这段旅行吧！'
      : '我想记录一段旅行回忆，帮我聊聊吧！');

    // 调用 Gemini
    const contents = [
      { role: 'user', parts: [{ text: systemPrompt + '\n\n' + userFirstMsg }, ...imageParts] }
    ];

    // 调用 Gemini（含自动降级）
    const response = await callGemini(contents);

    let reply = '', keywords = [];
    try {
      // 尝试解析 JSON 回复
      const text = response.text || '';
      const jsonMatch = text.match(/\{[\s\S]*"reply"[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        reply = parsed.reply || text;
        keywords = parsed.keywords || [];
      } else {
        reply = text.replace(/```json\s*/g, '').replace(/```/g, '').trim();
      }
    } catch (e) {
      reply = response.text || '你好！跟我聊聊这段旅行吧～';
    }
    reply = stripKeywordsLeak(reply);

    // 保存会话
    const history = [
      { role: 'user', parts: [{ text: systemPrompt + '\n\n' + userFirstMsg }, ...imageParts] },
      { role: 'model', parts: [{ text: response.text || reply }] }
    ];

    // 标准化关键词格式
    const normalizedKw = (keywords || []).map(kw => typeof kw === 'string' ? { text: kw, type: 'other' } : kw);

    chatSessions.set(sessionId, {
      history,
      keywords: normalizedKw,
      chatLog: [{ role: 'ai', text: reply }],
      lastActive: Date.now()
    });

    res.json({ success: true, sessionId, reply, keywords: normalizedKw });
  } catch (err) {
    console.error('AI chat start error:', err);
    res.status(500).json({ error: 'AI 对话启动失败：' + err.message });
  }
});

// ========== AI 聊天：开始旅行计划对话 ==========
app.post('/api/chat/start-plan', async (req, res) => {
  const { location } = req.body; // location: '意大利'
  try {
    const sessionId = Date.now().toString() + '-' + Math.random().toString(36).slice(2, 8);

    const systemPrompt = `你是「漫游喵」，一位经验丰富又有点幽默感的旅行规划师。你的风格是专业但亲切——像一个去过很多地方的朋友给你中肯的建议，不会过度热情或卖萌。

职责：帮用户规划旅行，提供实用的目的地信息和建议。

重要——地名相关：
- 当用户提到某个地名（城市、景点、区域、国家等）时，除了回答当前问题，还要自然地穿插 1-2 句关于该地的基本介绍（如特色、人文、为什么值得去），让聊天既有规划又涨知识。

重要——对话节奏：
- 第一轮回复：简短确认目的地，然后自然地提到 1-2 个这个地方最有特色的亮点（比如标志性景点、当季特色、美食名片等），让用户感受到你对这里很了解，最后问 1-2 个关键问题（什么时间去、几天、偏好什么风格）
- 后续介绍城市/景点时：在推荐的同时主动提到实用的注意事项（比如当地文化禁忌、治安提醒、交通坑、天气穿搭、签证/货币小贴士等），自然融入建议中，不要单独列一大段注意事项
- 每次聚焦一个方面展开，信息分段呈现，不要一口气全部说完

规则：
1. 建议要具体实用，推荐景点/餐厅说出名字和理由
2. 知道近期有活动/节日/赛事的话自然提到
3. 适当分享当地文化习惯和实用小贴士
4. 语气专业友善，偶尔幽默，不滥用emoji和感叹号
5. 绝对不要使用 ** 星号标记，用自然语言表达重点，需要分段时用换行符
6. 提取关键词
7. 回复格式必须是 JSON：{"reply": "你的回复文字", "keywords": [{"text":"关键词","type":"类型"}]}
8. 关键词 type：scene/place/feature/mood/tip/other
9. 关键词累积不重复
10. reply 中用 \\n 来分段，让信息层次清晰
11. 需要详细介绍时不要吝啬篇幅，把信息说清楚`;

    const locationInfo = location ? `用户选择了目的地：${location}。` : '';
    const userFirstMsg = locationInfo + '我在考虑去这个地方旅行。';

    const contents = [
      { role: 'user', parts: [{ text: systemPrompt + '\n\n' + userFirstMsg }] }
    ];

    // 调用 Gemini（含自动降级）
    const response = await callGemini(contents);

    let reply = '', keywords = [];
    try {
      const text = response.text || '';
      const jsonMatch = text.match(/\{[\s\S]*"reply"[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        reply = parsed.reply || text;
        keywords = parsed.keywords || [];
      } else {
        reply = text.replace(/```json\s*/g, '').replace(/```/g, '').trim();
      }
    } catch (e) {
      reply = response.text || '让我来帮你规划旅行吧～';
    }
    reply = stripKeywordsLeak(reply);

    const history = [
      { role: 'user', parts: [{ text: systemPrompt + '\n\n' + userFirstMsg }] },
      { role: 'model', parts: [{ text: response.text || reply }] }
    ];

    const normalizedKw = (keywords || []).map(kw => typeof kw === 'string' ? { text: kw, type: 'other' } : kw);

    chatSessions.set(sessionId, {
      history,
      keywords: normalizedKw,
      chatLog: [{ role: 'ai', text: reply }],
      lastActive: Date.now()
    });

    res.json({ success: true, sessionId, reply, keywords: normalizedKw });
  } catch (err) {
    console.error('AI plan chat start error:', err);
    res.status(500).json({ error: 'AI 对话启动失败：' + err.message });
  }
});

// ========== AI 聊天：继续对话 ==========
app.post('/api/chat/message', async (req, res) => {
  const { sessionId, message } = req.body;
  if (!sessionId || !message) return res.status(400).json({ error: '缺少 sessionId 或 message' });

  const session = chatSessions.get(sessionId);
  if (!session) return res.status(404).json({ error: '会话已过期，请重新开始' });

  try {
    session.lastActive = Date.now();
    session.chatLog.push({ role: 'user', text: message });

    // 追加用户消息到历史
    session.history.push({ role: 'user', parts: [{ text: message }] });

    // 调用 Gemini 继续对话（含自动降级）
    const response = await callGemini(session.history);

    let reply = '', newKeywords = [];
    try {
      const text = response.text || '';
      const jsonMatch = text.match(/\{[\s\S]*"reply"[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        reply = parsed.reply || text;
        newKeywords = parsed.keywords || [];
      } else {
        reply = text.replace(/```json\s*/g, '').replace(/```/g, '').trim();
      }
    } catch (e) {
      reply = response.text || '嗯嗯，继续跟我说说～';
    }
    reply = stripKeywordsLeak(reply);

    // 追加 AI 回复到历史
    session.history.push({ role: 'model', parts: [{ text: response.text || reply }] });
    session.chatLog.push({ role: 'ai', text: reply });

    // 合并关键词（按 text 去重，支持新格式 {text,type} 和旧格式 string）
    const normalize = kw => typeof kw === 'string' ? { text: kw, type: 'other' } : kw;
    const existing = session.keywords.map(normalize);
    const incoming = newKeywords.map(normalize);
    const seen = new Set(existing.map(k => k.text));
    for (const k of incoming) {
        if (!seen.has(k.text)) { existing.push(k); seen.add(k.text); }
    }
    session.keywords = existing;

    res.json({ success: true, reply, keywords: session.keywords });
  } catch (err) {
    console.error('AI chat message error:', err);
    res.status(500).json({ error: 'AI 回复失败：' + err.message });
  }
});

// ========== 恢复聊天会话（从已保存的 chatLog 重建） ==========
app.post('/api/chat/resume', (req, res) => {
  const { chatLog, keywords, type } = req.body; // type: 'record' | 'plan'
  if (!chatLog || chatLog.length === 0) return res.status(400).json({ error: '无聊天记录可恢复' });

  const sessionId = Date.now().toString() + '-' + Math.random().toString(36).slice(2, 8);

  // 根据类型选择系统提示词
  const systemPrompt = type === 'plan'
    ? '你是「漫游喵」，一位经验丰富的旅行规划师。请继续之前的对话，帮用户补充和完善旅行计划。当用户提到地名时，自然地加 1-2 句该地的基本介绍。回复格式：{"reply":"回复","keywords":[{"text":"关键词","type":"类型"}]}，不要使用**星号，用\\n分段。'
    : '你是「拾光鹿」，一位温暖的旅行回忆助手。请继续之前的对话，帮用户补充更多旅行回忆。回复格式：{"reply":"回复","keywords":[{"text":"关键词","type":"类型"}]}，不要使用**星号，用\\n分段。';

  // 重建 Gemini 对话历史
  const history = [];
  // 首条包含系统提示
  const firstUserMsg = chatLog.find(m => m.role === 'user');
  history.push({ role: 'user', parts: [{ text: systemPrompt + '\n\n（以下是之前的对话记录，请基于这些内容继续）\n' + chatLog.map(m => `${m.role === 'ai' ? 'AI' : '用户'}：${m.text}`).join('\n') }] });
  history.push({ role: 'model', parts: [{ text: '好的，我已经了解了之前的对话内容，请继续吧！' }] });

  const normalizedKw = (keywords || []).map(kw => typeof kw === 'string' ? { text: kw, type: 'other' } : kw);

  chatSessions.set(sessionId, {
    history,
    keywords: normalizedKw,
    chatLog: [...chatLog], // 复制历史记录
    lastActive: Date.now()
  });

  res.json({ success: true, sessionId });
});

// ========== 获取会话数据（关键词+聊天记录） ==========
app.get('/api/chat/:sessionId', (req, res) => {
  const session = chatSessions.get(req.params.sessionId);
  if (!session) return res.status(404).json({ error: '会话不存在' });
  res.json({ keywords: session.keywords, chatLog: session.chatLog });
});

app.listen(PORT, () => {
  console.log(`服务器运行在 http://localhost:${PORT}`);
});
