# 故障记录：DataFrame 表格在 Streamlit 1.59.2 上渲染空白

## 问题

`st.dataframe()` 在深色/浅色模式下均显示空白（只显示 toolbar 图标，无数据行/列），但点击单元格时数据（白底黑字）可见。

## 根因

`web_app/app.py` 全局 CSS 中（原第 169 行附近）用 `!important` 强行覆盖了 Streamlit 1.59.2 的 DataFrame 组件内部样式。

Streamlit 1.59.2 的 `st.dataframe` 底层使用 `@glideapps/glide-data-grid`，该库通过哈希化 CSS 类名（如 `.gdg-r17m35ur`���+ CSS 变量（`--gdg-bg-cell`、`--gdg-text-dark` 等）+ Canvas JS 绘制来渲染表格。

原有 CSS 中多处用旧类名（`.gdg-cell`、`.dvn-scroller`、`.dvn-stack`）加 `!important` 覆盖背景色和文字色，这些类名在 1.59.2 中已不存在；同时 `div[data-testid="stDataFrame"] canvas { background-color: _card !important; }` 覆盖了 Canvas 背景，但 Canvas 内文字由 Glide 的 JS 直接绘制（不受 CSS `color` 影响），导致背景深色 + 默认浅色主题黑字 = 全不可见。

## 修复

- 删除所有对 DataFrame 内部子元素的 CSS 覆盖（`.gdg-cell`、`.dvn-*`、canvas、`*` 统配等）
- 仅保留外边框样式：
  ```css
  div[data-testid="stDataFrame"] {
    border: 1px solid {_border};
    border-radius: 8px;
    overflow: hidden;
  }
  ```
- 让 Streamlit/Glide 原生控制表格内部渲染，主题切换时只需通过父级 `color-scheme` 让 Glide 自动适配深色/浅色主题

## 涉及文件

- `web_app/app.py` — 全局 CSS 配置，删除了 `display: none` 和内部元素 `!important` 覆盖

## 相关版本

- Streamlit 1.59.2
- @glideapps/glide-data-grid（随 Streamlit 前端构建内嵌）
