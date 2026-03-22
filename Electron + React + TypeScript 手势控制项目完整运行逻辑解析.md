# Electron + React + TypeScript 手势控制项目完整解析

### 🗺️ 核心运行逻辑总图

整个项目的执行顺序如下：

1. **启动触发**：执行 `npm run dev` -> 读取 `package.json` -> 调用 Vite 构建。

2. **主进程启动**：Vite 插件编译 `electron/main.ts` -> 创建 Electron 窗口。

3. **渲染进程挂载**：窗口加载 React 前端代码（`src/index.tsx`）。

4. **功能闭环**：前端通过 `preload.ts` 调用主进程 API，完成「手势识别 -> 系统控制」。

---

### 📁 关键文件位置与功能对应

根据你的目录结构，各核心模块所在的物理位置如下：

#### 1. 主进程（Electron 核心逻辑）

决定应用是否能启动、窗口如何显示、以及如何操作硬件。

- **入口文件**：`electron/main.ts`（**最核心文件**）

    - **作用**：包含 `app.whenReady()`、`createWindow()` 等启动逻辑。

    - **包含内容**：初始化 `electron-store`（配置）、配置 `robotjs`（键鼠控制）、定义 IPC 通信接口。

- **预加载脚本（通信桥）**：`electron/preload.ts`

    - **作用**：通过 `contextBridge` 把 API 暴露给前端，是前端调用底层系统能力的唯一入口。

- **环境声明**：`electron/electron-env.d.ts`

    - **作用**：TypeScript 类型声明文件，定义全局变量类型。

#### 2. 渲染进程（React 前端界面逻辑）

负责 UI 显示、摄像头采集、手势识别算法。

- **前端入口**：`src/main.tsx`

    - **作用**：React 应用的根入口，挂载 Redux Store，渲染根组件。

- **根组件**：`src/App.tsx`

    - **作用**：包含摄像头组件（`react-webcam`）、调用 MediaPipe 手势识别、处理 UI 交互。

- **功能模块目录**：

    - `src/components/`：通用 UI 组件（按钮、设置面板等）。

    - `src/pages/`：多页面路由组件（如首页、设置页）。

    - `src/hooks/`：自定义 React Hooks（可能封装了手势识别逻辑）。

    - `src/stores/`：Redux 状态管理（存储手势配置、运行状态）。

    - `src/helpers/`：工具函数（可能包含坐标转换、手势判定逻辑）。

#### 3. 静态资源与模型

程序运行所需的非代码文件。

- `public/models/`：**核心资源**！存放 MediaPipe 手势识别模型文件（`.task` 文件）。

- `public/images/`：图标、背景图等静态资源。

#### 4. 公共定义（类型与常量）

- `common/constants/`：定义全局常量（如配置项 Key、默认参数）。

- `common/types/`：全局 TypeScript 类型定义（如手势数据结构、IPC 通信类型）。

---

### 🔄 详细执行流程拆解

#### 第一阶段：应用启动

1. **命令**：在终端运行 `npm run dev`。

2. **Vite 构建**：Vite 根据 `vite.config.ts` 配置，同时编译主线程（`electron/main.ts`）和渲染线程（`src/`）。

3. **主进程运行**：Electron 启动，执行 `electron/main.ts`。

    - 初始化 `Store`（读取本地配置）。

    - 创建浏览器窗口 (`BrowserWindow`)。

    - 加载前端 URL（开发环境通常是 `http://localhost:xxxx`）。

#### 第二阶段：前端初始化

1. **React 启动**：窗口加载 `src/main.tsx`。

2. **Store 初始化**：挂载 Redux，从 `stores` 目录中加载状态切片（Slices）。

3. **组件渲染**：渲染 `App.tsx`，挂载 UI 界面。

#### 第三阶段：手势控制核心逻辑（闭环）

1. **视频采集**：`App.tsx` 中通过 `react-webcam` 调用摄像头，获取实时视频流。

2. **手势识别**：

    - 加载 `public/models/` 下的模型。

    - 使用 `@mediapipe/tasks-vision` 对每一帧视频进行手部关键点检测。

3. **坐标转换与决策**：在 `helpers` 或 `hooks` 中计算关键点坐标，判定具体手势（如握拳、手掌张开）。

4. **调用系统 API**：

    - 前端通过 `window.api.moveMouse()`（由 `preload.ts` 注入）发送请求。

    - 请求通过 IPC 通道传递给主进程。

5. **底层执行**：主进程的 `main.ts` 接收到请求，调用 `robotjs` 控制鼠标移动或键盘输入。

---

### 💡 快速定位代码的实操建议

如果你想修改某个功能，直接按以下路径查找：

|需求场景|查找文件/路径|
|---|---|
|**修改窗口大小/标题**|`electron/main.ts` -> `createWindow` 函数|
|**修改手势识别模型**|替换 `public/models/` 下的 `.task` 文件|
|**修改 UI 外观**|`src/App.tsx` 或 `components/` 目录|
|**修改 Redux 状态**|`src/stores/` 目录|
|**修改键鼠控制逻辑**|`electron/main.ts` -> 搜索 `robotjs` 相关代码|
|**修改 IPC 通信定义**|`electron/preload.ts` (导出) + `src/` (调用)|
这个项目结构非常标准，核心逻辑全部集中在 `electron/main.ts`（后台）和 `src/App.tsx`（前台），你可以重点从这两个文件入手阅读。
