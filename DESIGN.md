---
version: alpha
name: "AI Workspace Agent"
description: "面向知识工作者的中文 AI 工作台，以克制的深色侧栏、明亮对话区与靛紫交互色组织长期工作。"
colors:
  primary: "#5B5BD6"
  primary-soft: "#F3F2FF"
  secondary: "#7770EE"
  background: "#F8FAFC"
  surface: "#FFFFFF"
  sidebar: "#111827"
  sidebar-surface: "#172033"
  text: "#172033"
  text-secondary: "#8A94A6"
  border: "#E8ECF2"
  success: "#36B989"
  danger: "#DC5656"
  focus: "#8582F2"
typography:
  sans:
    fontFamily: 'Inter, "PingFang SC", "Microsoft YaHei", system-ui, sans-serif'
  mono:
    fontFamily: 'ui-monospace, "Cascadia Code", "SFMono-Regular", monospace'
rounded:
  sm: "7px"
  DEFAULT: "10px"
  lg: "15px"
spacing:
  sidebar-width: "292px"
  header-height: "76px"
  conversation-measure: "820px"
  sidebar-padding: "16px"
components:
  button: { }
  card: { }
  dialog: { }
---

# AI Workspace Agent Design System

## Overview

### Creative North Star

把界面做成一张安静、耐久的数字工作台：左侧像工具柜一样收纳会话与知识，右侧留出宽阔、明亮的思考空间。结构优先于装饰，靛紫色只标记主要动作、身份与专注状态。

### Product context and register

- **Audience and primary job:** 使用中文的个人知识工作者；通过 AI 对话、会话记录和知识库完成检索与任务拆解。
- **Target market(s) and evidence:** 个人及小型团队的线上 AI 工具；依据项目 README 的知识问答、RAG、工具调用和 MCP 能力说明，不暗示企业合规或服务等级。
- **Locale(s) and language policy:** 简体中文为主要界面语言，工具名等必要技术词保留英文；时间和访客额度重置采用北京时间。
- **Usage scene:** 桌面浏览器是主要操作环境，也需支持窄屏移动端的提问、查看会话和管理员登录。
- **Register:** 产品型工作台。访客和管理员使用相同工作区外壳，管理员操作按权限出现。
- **Memorable signature:** 深色会话侧栏与明亮对话画布之间清晰、稳定的纵向分界。
- **Restraint:** 对话内容、知识库状态和权限说明优先；不用渐变大图或营销式 Hero（首屏主视觉）抢占工作空间。
- **Anti-references:** 不做密集的运维监控大屏、不做游戏化聊天气泡，也不模仿系统设置页的多层表格。
- **Token ownership/runtime mapping:** Model B（既有运行样式为准）。`frontend/src/App.css` 与 `frontend/src/index.css` 是已实施颜色、排版、布局和形状的运行时来源；本文件记录其稳定意图，不生成另一套 CSS token。当前尚无自动 token 漂移检查，涉及系统级调整时需同步核验本文与运行样式。

## Colors

主要动作与用户消息使用 `primary` 靛紫；`primary-soft` 用于状态徽标等低强调底色。工作区域使用 `background` 和 `surface` 分层，侧栏使用 `sidebar` / `sidebar-surface`。`text-secondary` 用于非关键说明，但正文仍须满足可读对比；边框仅承担分组。成功、危险和键盘焦点分别用语义色表达，不能只靠颜色传达状态。

## Typography

运行时无衬线字体采用 Inter 优先，并回退到苹方、微软雅黑及系统字体，以覆盖中文界面。正文短说明使用舒适行高；会话标题与问题输入保持明显层级。等宽字体仅用于未来确有代码或机器标识的内容。中文文案简短直接，避免把内部权限细节作为用户需要理解的前提。

## Layout

桌面使用固定 292px 侧栏和弹性对话区；对话正文与输入框以 820px 为阅读宽度上限。76px 顶栏维持在线状态、身份与管理员动作。移动端侧栏变为抽屉，关键额度与登录入口仍留在主工作区，不把长提示挤进侧栏。消息列表独立滚动，输入区固定在可视工作区底部并预留移动安全区。

## Elevation & Depth

层级主要依靠明暗色面与细边框，而非堆叠阴影。管理员确认对话框使用原生 `dialog` 模态语义，并加柔和遮罩和单层投影；普通知识库和会话区域不做浮层效果。加载、错误和访客额度状态保留固定空间，避免界面跳动。

## Shapes

按钮、输入与会话项用中小圆角；对话画布保持大面积方正，消息气泡只在发送方向上形成轻微不对称。徽标可用胶囊形，但不可只用徽标颜色说明身份。聚焦轮廓清晰且不被裁切。

## Components

### Foundational visual states

交互项需具备默认、悬停、键盘聚焦、禁用和忙碌状态。错误使用带文字的浅红底提示；访客额度耗尽时输入框禁用并说明恢复时间。减少动态效果偏好下关闭不必要动画。

### Buttons and actions

发送提问和登录是主要动作；取消为中性动作；永久删除使用独立危险样式，并在应用内对话框中明确影响对象。按钮忙碌时保留尺寸并禁止重复提交。仅图标操作必须有中文可访问名称。

### Navigation and data display

会话侧栏支持移动抽屉；访客只看到明确开放的知识库，管理员可在同一目录选择、上传与管理文档。管理员徽标与访客额度常驻于顶部/输入区，权限不通过前端自报，而是由服务器会话返回。

### Forms and overlays

管理员登录使用邮箱和密码字段，允许密码管理器填充与粘贴，错误在表单内显示。登录弹窗可通过关闭按钮、取消或 Escape 退出，并将焦点交给邮箱输入。永久删除弹窗先聚焦安全的取消动作，等待服务器确认后才关闭。

### Iconography

现有应用使用少量 Unicode 符号与简化文字标记，不混用第三方图标库。涉及重要操作时优先保留文字标签；符号只作辅助，且不能替代可访问名称。

### Motion

抽屉进出和加载点动画仅表示当前状态，使用短促、低幅度动效。系统 `prefers-reduced-motion` 生效时缩减过渡和动画。

### Content and data visualization

用语面向使用者，例如“今日还可提问 7 次”“北京时间 00:00 重置”，避免暴露 HMAC、代理头或服务端账本等内部术语。在线/离线状态同时配文字；不使用只有颜色的额度告警。

## Do's and Don'ts

- **Do:** 保持身份、额度和恢复时刻可见且能被读屏器理解。
- **Do:** 管理员和访客共享一致的工作区布局，但只呈现各自可用操作。
- **Don't:** 把共享访问令牌、服务端密钥或管理员权限判断放到访客界面。
- **Don't:** 为了移动适配隐藏额度、登录或知识库选择等关键操作。
