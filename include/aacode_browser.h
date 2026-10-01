/*
 * Copyright (c) 2026 xiefujin <490021684@qq.com>
 * Licensed under GPL-3.0, see LICENSE file for full license terms.
 *
 * aacode_browser.h — host WebView bridge for aacode's browser tools.
 *
 * The host app implements an **offscreen** native WebView (Android WebView /
 * iOS WKWebView) behind this function table and registers it once, before
 * starting any task. aacode then drives it via the fastbrowser kernel
 * (engine = "webview"); the user never sees a browser UI.
 *
 * Memory contract (mirrors fastbrowser):
 *   - `evaluate` returns a malloc'd JSON string; aacode releases it via
 *     `free_string`.
 *   - `screenshot` writes width/height and returns a malloc'd RGBA buffer;
 *     released via `screenshot_free`.
 *   - Return negative handles / non-zero codes on failure.
 *
 * Requires aacode-rs built with `--features browser`.
 */

#ifndef AACODE_BROWSER_H
#define AACODE_BROWSER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FbWebViewOps {
    /* Create a native webview. Returns handle (>=0) or -1. */
    int64_t (*create)(const char *url, uint32_t width, uint32_t height);
    /* Destroy a webview. */
    int (*destroy)(int64_t handle);
    /* Evaluate JS; returns malloc'd JSON string or NULL on failure. */
    char *(*evaluate)(int64_t handle, const char *script);
    /* Navigate the webview. */
    int (*navigate)(int64_t handle, const char *url);
    /* Capture screenshot; returns malloc'd RGBA buffer (free via screenshot_free). */
    uint8_t *(*screenshot)(int64_t handle, uint32_t *out_w, uint32_t *out_h);
    void (*screenshot_free)(uint8_t *ptr);
    /* Set viewport. */
    int (*set_viewport)(int64_t handle, uint32_t width, uint32_t height);
    /* Return a native view handle for embedding (>=0) or -1. Offscreen: return handle. */
    int64_t (*native_view)(int64_t handle);
    /* Dispatch an input event given as JSON. */
    int (*dispatch_event)(int64_t handle, const char *event_json);
    /* History navigation. */
    int (*go_back)(int64_t handle);
    int (*go_forward)(int64_t handle);
    /* Free a string returned by evaluate. */
    void (*free_string)(char *ptr);
} FbWebViewOps;

/* Register the host offscreen WebView backend. Call once, before any task.
 * Returns 0 on success, non-zero on error (-1 when `browser` is not compiled). */
int aacode_browser_register_webview_ops(const FbWebViewOps *ops);

/* 1 if a browser backend is available (feature compiled + ops registered). */
int aacode_browser_available(void);

#ifdef __cplusplus
}
#endif

#endif /* AACODE_BROWSER_H */
