// feed_probe.h - Eden Vulkan scene-texture discovery probe.
//
// A diagnostic-only module: watches the game's Vulkan texture resources and
// render-target bindings through the ReShade add-on events, scores candidates
// for "internal scene color before UI", and offers a manual one-shot capture
// (GPU copy -> readback buffer -> BMP + metadata under dlss5-probe/).
//
// Hard rules (per the probe spec):
//  - metadata tracking only; no per-frame readback, no VRAM pokes, no CPU
//    memory scanning;
//  - never modifies game rendering -- selecting a candidate only marks it;
//  - capture happens solely on an explicit overlay button press;
//  - the DLSS feeder path is inert while the probe is enabled.
//
// Everything is gated behind the "probe" config key: when it is 0 the event
// handlers return immediately and the only cost is a branch per event.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include <imgui.h>
#include <reshade.hpp>

namespace probe {

// ---------------------------------------------------------------------------
// Config (persisted in dlss5-feed.cfg by the host translation unit)
// ---------------------------------------------------------------------------

static int  g_enabled    = 0; // "probe" key: 1 = discovery active, DLSS feed inert
static int  g_candidate  = 1; // "probe_candidate" key: selected rank, 1-based
static int  g_log_summary_every = 300;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct TexInfo
{
    bool     alive = false;
    bool     is_swapchain = false;

    uint32_t w = 0, h = 0;
    uint32_t mips = 0, layers = 0, samples = 0;
    unsigned fmt = 0;

    reshade::api::resource_usage last_usage = reshade::api::resource_usage::undefined;

    // Cumulative
    bool     ever_rt = false, ever_depth = false, ever_srv = false;
    bool     ever_copy_src = false, ever_copy_dst = false;
    uint64_t create_frame = 0, destroy_frame = 0;
    uint32_t frames_seen = 0;
    uint64_t total_draws = 0;

    // Per-frame (reset on present)
    uint32_t fr_draws = 0, fr_rt_binds = 0;
    bool     fr_depth = false, fr_feeds_swapchain = false, fr_copy_to = false;
    uint32_t fr_vp_w = 0, fr_vp_h = 0;
    uint64_t fr_first_event = 0, fr_last_event = 0;
    bool     fr_active = false;

    // Snapshot
    uint32_t score = 0;
};

struct ActiveRt  // per command list
{
    uint64_t rt_res[4] = {};
    uint32_t rt_count = 0;
    bool     depth = false;
};

static SRWLOCK                       g_lock = SRWLOCK_INIT;
static bool                          g_inited = false;
static reshade::api::device         *g_device = nullptr;
static uint64_t                      g_frame = 0;
static uint64_t                      g_seq = 0;

static std::unordered_map<uint64_t, TexInfo> g_tex;          // key: resource handle
static std::unordered_map<uint64_t, ActiveRt> g_active;      // key: command list handle
static std::vector<std::pair<uint64_t, uint64_t>> g_edges;   // per-frame copy edges (src, dst)
static std::vector<uint64_t>         g_swapchain_res;        // known backbuffer resources

struct Snapshot
{
    uint64_t handle = 0;
    uint32_t w = 0, h = 0;
    unsigned fmt = 0;
    uint32_t draws = 0, binds = 0, score = 0;
    bool     depth = false, feeds = false;
    uint64_t first = 0, last = 0;
    uint32_t layers = 0, samples = 0;
};
static std::vector<Snapshot> g_top;                          // last frame's top-10

// Capture state
static reshade::api::fence      g_capture_fence = { 0 };
static uint64_t                 g_capture_fence_value = 0;
static reshade::api::resource   g_capture_buffer = { 0 };
static uint64_t                 g_capture_buffer_size = 0;
static volatile long            g_capture_request = 0;   // set by overlay button
static uint64_t                 g_capture_handle = 0;    // resolved target
static bool                     g_capture_busy = false;

// ---------------------------------------------------------------------------

static void Log(const char *fmt, ...)
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    reshade::log::message(reshade::log::level::info, buf);
}

static TexInfo *FindTex(uint64_t h)
{
    auto it = g_tex.find(h);
    return it != g_tex.end() ? &it->second : nullptr;
}

static const char *FmtName(unsigned f)
{
    switch (reshade::api::format(f))
    {
    case reshade::api::format::r8g8b8a8_typeless:   return "R8G8B8A8_TYPELESS";
    case reshade::api::format::r8g8b8a8_unorm:      return "R8G8B8A8_UNORM";
    case reshade::api::format::r8g8b8a8_unorm_srgb: return "R8G8B8A8_SRGB";
    case reshade::api::format::b8g8r8a8_typeless:   return "B8G8R8A8_TYPELESS";
    case reshade::api::format::b8g8r8a8_unorm:      return "B8G8R8A8_UNORM";
    case reshade::api::format::b8g8r8a8_unorm_srgb: return "B8G8R8A8_SRGB";
    case reshade::api::format::r10g10b10a2_typeless:return "R10G10B10A2_TYPELESS";
    case reshade::api::format::r16g16b16a16_float:  return "R16G16B16A16_FLOAT";
    case reshade::api::format::r11g11b10_float:     return "R11G11B10_FLOAT";
    case reshade::api::format::r10g10b10a2_unorm:   return "R10G10B10A2_UNORM";
    case reshade::api::format::r32_float:           return "R32_FLOAT";
    case reshade::api::format::d32_float:           return "D32_FLOAT";
    case reshade::api::format::d16_unorm:           return "D16_UNORM";
    case reshade::api::format::d24_unorm_s8_uint:   return "D24S8";
    case reshade::api::format::d32_float_s8_uint:   return "D32S8";
    case reshade::api::format::r8_unorm:            return "R8_UNORM";
    case reshade::api::format::r16g16_float:        return "R16G16_FLOAT";
    default: return "?";
    }
}

static bool IsDepthFmt(unsigned f)
{
    switch (reshade::api::format(f))
    {
    case reshade::api::format::d32_float:
    case reshade::api::format::d16_unorm:
    case reshade::api::format::d24_unorm_s8_uint:
    case reshade::api::format::d32_float_s8_uint:
        return true;
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// Event handlers (registered once; every one bails immediately when disabled)
// ---------------------------------------------------------------------------

static void OnResourceInit(reshade::api::device *device, const reshade::api::resource_desc &desc,
                           const reshade::api::subresource_data *, reshade::api::resource_usage, reshade::api::resource resource)
{
    if (!g_enabled || desc.type != reshade::api::resource_type::texture_2d)
        return;

    AcquireSRWLockExclusive(&g_lock);
    if (g_tex.size() >= 8192 && FindTex(resource.handle) == nullptr)
    {
        static bool said = false;
        if (!said) { said = true; Log("[probe] resource table full (8192); no longer tracking new textures"); }
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }
    TexInfo &t = g_tex[resource.handle];
    t.alive = true;
    t.w = desc.texture.width;
    t.h = desc.texture.height;
    t.layers = desc.texture.depth_or_layers;
    t.mips = desc.texture.levels;
    t.samples = desc.texture.samples;
    t.fmt = unsigned(desc.texture.format);
    t.last_usage = desc.usage;
    t.create_frame = g_frame;
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnResourceDestroy(reshade::api::device *, reshade::api::resource resource)
{
    if (!g_enabled)
        return;
    AcquireSRWLockExclusive(&g_lock);
    if (TexInfo *t = FindTex(resource.handle))
    {
        t->alive = false;
        t->destroy_frame = g_frame;
        if (t->is_swapchain)
        {
            for (size_t i = 0; i < g_swapchain_res.size(); ++i)
                if (g_swapchain_res[i] == resource.handle) { g_swapchain_res[i] = g_swapchain_res.back(); g_swapchain_res.pop_back(); break; }
        }
    }
    g_active.erase(resource.handle);
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnResourceViewInit(reshade::api::device *device, reshade::api::resource resource,
                               reshade::api::resource_usage usage_type, const reshade::api::resource_view_desc &, reshade::api::resource_view)
{
    if (!g_enabled || resource.handle == 0)
        return;
    AcquireSRWLockExclusive(&g_lock);
    if (TexInfo *t = FindTex(resource.handle))
    {
        if (usage_type == reshade::api::resource_usage::depth_stencil) t->ever_depth = true;
        else if (usage_type == reshade::api::resource_usage::render_target) t->ever_rt = true;
        else if (usage_type == reshade::api::resource_usage::shader_resource) t->ever_srv = true;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static void TouchActive(reshade::api::command_list *cmd_list, uint64_t res_handle, bool depth)
{
    // Caller holds no lock; we take it. Records the binding and the event order.
    AcquireSRWLockExclusive(&g_lock);
    ActiveRt &a = g_active[cmd_list->get_native()];
    if (!depth)
    {
        bool already = false;
        for (uint32_t i = 0; i < a.rt_count; ++i) already |= (a.rt_res[i] == res_handle);
        if (!already && a.rt_count < 4) a.rt_res[a.rt_count++] = res_handle;
    }
    a.depth |= depth;

    if (TexInfo *t = FindTex(res_handle))
    {
        if (g_seq == 0) g_seq = 1;
        if (t->fr_first_event == 0) t->fr_first_event = g_seq;
        t->fr_last_event = g_seq;
        t->fr_active = true;
        if (depth) { t->fr_depth = true; t->ever_depth = true; }
        else       { t->fr_rt_binds++;  t->ever_rt = true;    }
        t->last_usage = depth ? reshade::api::resource_usage::depth_stencil
                              : reshade::api::resource_usage::render_target;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnBindRenderTargets(reshade::api::command_list *cmd_list, uint32_t count,
                                const reshade::api::resource_view *rtvs, reshade::api::resource_view dsv)
{
    if (!g_enabled || g_device == nullptr)
        return;
    g_seq++;
    for (uint32_t i = 0; i < count && i < 8; ++i)
    {
        if (rtvs[i].handle == 0) continue;
        const reshade::api::resource res = g_device->get_resource_from_view(rtvs[i]);
        if (res.handle != 0) TouchActive(cmd_list, res.handle, false);
    }
    if (dsv.handle != 0)
    {
        const reshade::api::resource res = g_device->get_resource_from_view(dsv);
        if (res.handle != 0) TouchActive(cmd_list, res.handle, true);
    }
}

static bool OnBeginRenderPass(reshade::api::command_list *cmd_list, uint32_t count,
                              const reshade::api::render_pass_render_target_desc *rts,
                              const reshade::api::render_pass_depth_stencil_desc *ds, reshade::api::render_pass_flags)
{
    if (!g_enabled)
        return false;
    g_seq++;
    for (uint32_t i = 0; i < count && i < 8; ++i)
    {
        if (rts[i].view.handle == 0 || g_device == nullptr) continue;
        const reshade::api::resource res = g_device->get_resource_from_view(rts[i].view);
        if (res.handle != 0) TouchActive(cmd_list, res.handle, false);
    }
    if (ds != nullptr && ds->view.handle != 0 && g_device != nullptr)
    {
        const reshade::api::resource res = g_device->get_resource_from_view(ds->view);
        if (res.handle != 0) TouchActive(cmd_list, res.handle, true);
    }
    return false;
}

static bool OnEndRenderPass(reshade::api::command_list *)
{
    if (!g_enabled) return false;
    g_seq++;
    return false;
}

static void RecordDraw(reshade::api::command_list *cmd_list, uint32_t n)
{
    if (!g_enabled)
        return;
    g_seq++;
    AcquireSRWLockExclusive(&g_lock);
    auto it = g_active.find(cmd_list->get_native());
    if (it != g_active.end())
    {
        for (uint32_t i = 0; i < it->second.rt_count; ++i)
            if (TexInfo *t = FindTex(it->second.rt_res[i]))
            {
                t->fr_draws += n;
                t->total_draws += n;
                if (t->fr_first_event == 0) t->fr_first_event = g_seq;
                t->fr_last_event = g_seq;
            }
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static bool OnDraw(reshade::api::command_list *cl, uint32_t vertices, uint32_t, uint32_t, uint32_t)
{
    RecordDraw(cl, 1 + vertices / 1000); // weight tiny draws less, huge draws a bit more
    return false;
}

static bool OnDrawIndexed(reshade::api::command_list *cl, uint32_t indices, uint32_t, uint32_t, int32_t, uint32_t)
{
    RecordDraw(cl, 1 + indices / 1000);
    return false;
}

static bool OnDrawIndirect(reshade::api::command_list *cl, reshade::api::indirect_command, reshade::api::resource, uint64_t, uint32_t count, uint32_t)
{
    RecordDraw(cl, count * 10); // indirect draws are typically heavy scene geometry
    return false;
}

static void OnViewport(reshade::api::command_list *cmd_list, uint32_t, uint32_t count, const reshade::api::viewport *vps)
{
    if (!g_enabled || count == 0 || vps == nullptr)
        return;
    AcquireSRWLockExclusive(&g_lock);
    auto it = g_active.find(cmd_list->get_native());
    if (it != g_active.end())
    {
        for (uint32_t i = 0; i < it->second.rt_count; ++i)
            if (TexInfo *t = FindTex(it->second.rt_res[i]))
            {
                t->fr_vp_w = uint32_t(vps[0].width);
                t->fr_vp_h = uint32_t(vps[0].height);
            }
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static void RecordCopy(uint64_t src, uint64_t dst)
{
    AcquireSRWLockExclusive(&g_lock);
    if (g_edges.size() < 2048) g_edges.push_back({ src, dst });
    if (TexInfo *s = FindTex(src)) { s->ever_copy_src = true; s->fr_active = true; }
    if (TexInfo *d = FindTex(dst)) { d->ever_copy_dst = true; d->fr_active = true; d->fr_copy_to = true; }
    ReleaseSRWLockExclusive(&g_lock);
}

static bool OnCopyResource(reshade::api::command_list *, reshade::api::resource src, reshade::api::resource dst)
{
    if (!g_enabled) return false;
    g_seq++;
    RecordCopy(src.handle, dst.handle);
    return false;
}

static bool OnCopyTextureRegion(reshade::api::command_list *, reshade::api::resource src, uint32_t,
                                const reshade::api::subresource_box *, reshade::api::resource dst, uint32_t,
                                const reshade::api::subresource_box *, reshade::api::filter_mode)
{
    if (!g_enabled) return false;
    g_seq++;
    RecordCopy(src.handle, dst.handle);
    return false;
}

static bool OnResolveTextureRegion(reshade::api::command_list *, reshade::api::resource src, uint32_t,
                                   const reshade::api::subresource_box *, reshade::api::resource dst, uint32_t,
                                   uint32_t, uint32_t, uint32_t, reshade::api::format)
{
    if (!g_enabled) return false;
    g_seq++;
    RecordCopy(src.handle, dst.handle);
    return false;
}

// ---------------------------------------------------------------------------

static void MarkSwapchainResource(uint64_t res_handle)
{
    if (!g_enabled || res_handle == 0)
        return;
    AcquireSRWLockExclusive(&g_lock);
    if (TexInfo *t = FindTex(res_handle))
    {
        if (!t->is_swapchain)
        {
            t->is_swapchain = true;
            bool have = false;
            for (uint64_t h : g_swapchain_res) have |= (h == res_handle);
            if (!have) g_swapchain_res.push_back(res_handle);
        }
    }
    ReleaseSRWLockExclusive(&g_lock);
}

// ---------------------------------------------------------------------------
// Scoring (spec section 8)
// ---------------------------------------------------------------------------

static uint32_t ScoreTex(const TexInfo &t, uint32_t swapchain_area)
{
    if (t.is_swapchain)
        return 0;

    int s = 0;
    if (t.fr_rt_binds > 0) s += 15;
    if (t.fr_draws >= 500) s += 25;
    else if (t.fr_draws >= 100) s += 20;
    else if (t.fr_draws >= 20) s += 10;
    else if (t.fr_draws > 0) s += 5;
    if (t.fr_depth) s += 15;
    if (t.fr_vp_w != 0 && t.w != 0 &&
        t.fr_vp_w >= (t.w >> 1) && t.fr_vp_w <= t.w * 2 &&
        t.fr_vp_h >= (t.h >> 1) && t.fr_vp_h <= t.h * 2) s += 10;
    if (swapchain_area != 0 && uint64_t(t.w) * t.h >= uint64_t(swapchain_area) / 2) s += 15;
    if (t.fr_feeds_swapchain) s += 20;
    if (t.frames_seen >= 100) s += 5;

    if (IsDepthFmt(t.fmt)) s -= 40;
    if (t.w < 256 || t.h < 256) s -= 20;
    if (t.samples > 1) s -= 10; // MSAA capture is unsupported; deprioritize

    if (s < 0) s = 0;
    if (s > 100) s = 100;
    return uint32_t(s);
}

static bool FeedsSwapchain(uint64_t start)
{
    // Reverse BFS over this frame's copy edges: from `start` forward, does any
    // path reach a swapchain image? (edge src -> dst; walk dst side)
    if (g_swapchain_res.empty()) return false;
    std::vector<uint64_t> frontier = { start };
    std::unordered_map<uint64_t, bool> seen;
    seen[start] = true;
    for (size_t step = 0; step < frontier.size() && step < 64; ++step)
    {
        const uint64_t cur = frontier[step];
        for (uint64_t sc : g_swapchain_res) if (cur == sc) return true;
        for (const auto &e : g_edges)
            if (e.first == cur && !seen[e.second]) { seen[e.second] = true; frontier.push_back(e.second); }
    }
    return false;
}

static void FinalizeFrame()
{
    uint32_t swapchain_area = 0;
    {
        AcquireSRWLockExclusive(&g_lock);
        for (uint64_t h : g_swapchain_res)
            if (const TexInfo *t = FindTex(h)) { swapchain_area = t->w * t->h; break; }
        for (auto &kv : g_tex)
        {
            TexInfo &t = kv.second;
            if (!t.alive || !t.fr_active || t.is_swapchain) continue;
            t.fr_feeds_swapchain = FeedsSwapchain(kv.first);
            t.score = ScoreTex(t, swapchain_area);
            t.frames_seen++;
        }
        // Top-10 snapshot (exclude swapchain itself)
        g_top.clear();
        for (int pass = 0; pass < 10; ++pass)
        {
            uint32_t best = 0; uint64_t best_h = 0;
            for (auto &kv : g_tex)
            {
                const TexInfo &t = kv.second;
                if (!t.alive || !t.fr_active || t.is_swapchain) continue;
                bool taken = false;
                for (const Snapshot &s : g_top) taken |= (s.handle == kv.first);
                if (taken) continue;
                if (t.score > best || (t.score == best && best_h == 0)) { best = t.score; best_h = kv.first; }
            }
            if (best_h == 0) break;
            const TexInfo &t = g_tex[best_h];
            Snapshot s;
            s.handle = best_h; s.w = t.w; s.h = t.h; s.fmt = t.fmt;
            s.draws = t.fr_draws; s.binds = t.fr_rt_binds; s.score = t.score;
            s.depth = t.fr_depth; s.feeds = t.fr_feeds_swapchain;
            s.first = t.fr_first_event; s.last = t.fr_last_event;
            s.layers = t.layers; s.samples = t.samples;
            g_top.push_back(s);
        }
        ReleaseSRWLockExclusive(&g_lock);
    }

    if (g_log_summary_every != 0 && (g_frame % uint64_t(g_log_summary_every)) == 0)
    {
        uint32_t alive = 0, active = 0;
        AcquireSRWLockShared(&g_lock);
        for (auto &kv : g_tex) { if (kv.second.alive) alive++; if (kv.second.fr_active) active++; }
        if (!g_top.empty())
        {
            const Snapshot &t = g_top[0];
            Log("[probe] frame %llu: tracked=%u active=%u | top: %ux%u %s draws=%u depth=%d feeds_swapchain=%d score=%u",
                (unsigned long long)g_frame, alive, active, t.w, t.h, FmtName(t.fmt), t.draws, int(t.depth), int(t.feeds), t.score);
        }
        else
            Log("[probe] frame %llu: tracked=%u active=%u | no candidates this frame", (unsigned long long)g_frame, alive, active);
        ReleaseSRWLockShared(&g_lock);
    }

    // Reset per-frame stats
    AcquireSRWLockExclusive(&g_lock);
    for (auto &kv : g_tex)
    {
        TexInfo &t = kv.second;
        t.fr_draws = t.fr_rt_binds = 0;
        t.fr_depth = t.fr_feeds_swapchain = t.fr_copy_to = false;
        t.fr_vp_w = t.fr_vp_h = 0;
        t.fr_first_event = t.fr_last_event = 0;
        t.fr_active = false;
    }
    g_edges.clear();
    g_active.clear();
    g_seq = 0;
    ReleaseSRWLockExclusive(&g_lock);
    g_frame++;
}

static void OnPresent(reshade::api::command_queue *, reshade::api::swapchain *, const reshade::api::rect *,
                      const reshade::api::rect *, uint32_t, const reshade::api::rect *)
{
    if (!g_enabled) return;
    FinalizeFrame();
}

// ---------------------------------------------------------------------------
// One-shot capture (spec sections 11-13)
// ---------------------------------------------------------------------------

static void CaptureAndSave(reshade::api::effect_runtime *rt, reshade::api::command_list *cl)
{
    if (g_device == nullptr) return;
    const reshade::api::resource res = { g_capture_handle };
    const reshade::api::resource_desc desc = g_device->get_resource_desc(res);
    if (desc.type != reshade::api::resource_type::texture_2d || desc.texture.samples != 1)
    {
        Log("[probe] capture skipped: not a 2D non-MSAA texture");
        return;
    }

    struct Fmt { unsigned f; const char *name; uint32_t bpp; bool bgra; };
    Fmt fmt = {};
    {
        const unsigned raw = unsigned(desc.texture.format);
        const Fmt known[] = {
            { unsigned(reshade::api::format::r8g8b8a8_unorm),     "R8G8B8A8_UNORM", 32, false },
            { unsigned(reshade::api::format::b8g8r8a8_unorm),     "B8G8R8A8_UNORM", 32, true },
            { unsigned(reshade::api::format::r8g8b8a8_unorm_srgb),     "R8G8B8A8_SRGB",  32, false },
            { unsigned(reshade::api::format::b8g8r8a8_unorm_srgb),     "B8G8R8A8_SRGB",  32, true },
            // TYPELESS resources are captured interpreted as the matching UNORM
            // family; if the real backing store is the opposite channel order the
            // BMP will show swapped R/B -- still diagnostic gold.
            { unsigned(reshade::api::format::r8g8b8a8_typeless),  "R8G8B8A8_TYPELESS(as RGBA8)", 32, false },
            { unsigned(reshade::api::format::b8g8r8a8_typeless),  "B8G8R8A8_TYPELESS(as BGRA8)", 32, true },
            { unsigned(reshade::api::format::r10g10b10a2_typeless),"R10G10B10A2_TYPELESS", 32, false },
            { unsigned(reshade::api::format::r10g10b10a2_unorm),  "R10G10B10A2_UNORM", 32, false },
            { unsigned(reshade::api::format::r11g11b10_float),    "R11G11B10_FLOAT", 32, false },
            { unsigned(reshade::api::format::r16g16b16a16_float), "R16G16B16A16_FLOAT", 64, false },
        };
        bool ok = false;
        for (const Fmt &k : known) if (raw == k.f) { fmt = k; ok = true; break; }
        if (!ok) { Log("[probe] capture unsupported format: %s (%u)", FmtName(raw), raw); return; }
    }

    const uint32_t w = desc.texture.width, h = desc.texture.height;
    const uint64_t pitch = (uint64_t(w) * (fmt.bpp / 8) + 255) & ~uint64_t(255);
    const uint64_t buf_size = pitch * h;

    if (g_capture_buffer.handle == 0 || g_capture_buffer_size < buf_size)
    {
        if (g_capture_buffer.handle != 0) g_device->destroy_resource(g_capture_buffer), g_capture_buffer = { 0 };
        reshade::api::resource_desc bd(buf_size, reshade::api::memory_heap::readback, reshade::api::resource_usage::copy_dest);
        if (!g_device->create_resource(bd, nullptr, reshade::api::resource_usage::copy_dest, &g_capture_buffer))
        {
            Log("[probe] capture failed: readback buffer creation (%llu bytes)", (unsigned long long)buf_size);
            return;
        }
        g_capture_buffer_size = buf_size;
    }
    if (g_capture_fence.handle == 0 &&
        !g_device->create_fence(0, reshade::api::fence_flags::none, &g_capture_fence))
    {
        Log("[probe] capture failed: fence creation failed");
        return;
    }

    // Candidate -> copy_source (assume its last observed usage), copy, restore.
    reshade::api::resource_usage from = reshade::api::resource_usage::render_target;
    if (const TexInfo *t = FindTex(g_capture_handle))
        if (t->last_usage != reshade::api::resource_usage::undefined)
            from = t->last_usage;
    const reshade::api::resource_usage to = reshade::api::resource_usage::copy_source;
    cl->barrier(1, &res, &from, &to);
    cl->copy_texture_to_buffer(res, 0, nullptr, g_capture_buffer, 0, w, h);
    const reshade::api::resource_usage back = from;
    cl->barrier(1, &res, &to, &back);

    reshade::api::command_queue *queue = rt->get_command_queue();
    queue->flush_immediate_command_list();
    const uint64_t v = ++g_capture_fence_value;
    queue->signal(g_capture_fence, v);
    queue->wait(g_capture_fence, v); // CPU-blocking; acceptable for a manual debug capture

    void *mapped = nullptr;
    if (!g_device->map_buffer_region(g_capture_buffer, 0, buf_size, reshade::api::map_access::read_only, &mapped) || mapped == nullptr)
    {
        Log("[probe] capture failed: buffer map");
        return;
    }

    char dir[] = "dlss5-probe";
    CreateDirectoryA(dir, nullptr);
    char pathbmp[MAX_PATH], pathtxt[MAX_PATH];
    snprintf(pathbmp, sizeof(pathbmp), "%s/frame_%06llu_candidate_%02u_%ux%u.bmp", dir,
             (unsigned long long)g_frame, g_candidate, w, h);
    snprintf(pathtxt, sizeof(pathtxt), "%s/frame_%06llu_candidate_%02u.txt", dir,
             (unsigned long long)g_frame, g_candidate);

    // Re-pack rows: ReShade's copy_to_buffer uses the requested row_length; the
    // driver may still require a device-dependent pitch, so walk with `pitch`
    // and emit tightly packed BMP rows from the mapped data.
    const bool f16 = fmt.bpp == 64;
    FILE *fp = nullptr;
    if (fopen_s(&fp, pathbmp, "wb") == 0 && fp != nullptr)
    {
        const uint32_t row = (w * 3 + 3) & ~3u;
        const uint32_t datasz = row * h;
        uint8_t hdr[54] = {};
        hdr[0] = 'B'; hdr[1] = 'M';
        *(uint32_t *)(hdr + 2) = 54 + datasz;
        *(uint32_t *)(hdr + 10) = 54;
        *(uint32_t *)(hdr + 14) = 40;
        *(int32_t  *)(hdr + 18) = int32_t(w);
        *(int32_t  *)(hdr + 22) = int32_t(h);
        *(uint16_t *)(hdr + 26) = 1;
        *(uint16_t *)(hdr + 28) = 24;
        *(uint32_t *)(hdr + 34) = datasz;
        fwrite(hdr, 1, 54, fp);
        std::vector<uint8_t> rowbuf(row, 0);
        for (int32_t y = int32_t(h) - 1; y >= 0; --y)
        {
            const uint8_t *s = reinterpret_cast<const uint8_t *>(mapped) + uint64_t(y) * pitch;
            for (uint32_t x = 0; x < w; ++x)
            {
                uint8_t r, g, b;
                if (f16)
                {
                    const uint16_t *px = reinterpret_cast<const uint16_t *>(s + uint64_t(x) * 8);
                    auto to8 = [](uint16_t hb) {
                        const uint32_t exp = (hb >> 10) & 0x1F, man = hb & 0x3FF;
                        float v = (exp == 0) ? man * (1.0f / 1024.0f) * (1.0f / 16384.0f)
                                  : (exp == 31) ? 1.0f : ldexpf(float(man | 0x400), int(exp) - 25);
                        if (v < 0) v = 0; if (v > 1) v = 1;
                        return uint8_t(v * 255.0f + 0.5f);
                    };
                    r = to8(px[0]); g = to8(px[1]); b = to8(px[2]);
                }
                else
                {
                    const uint32_t px = *reinterpret_cast<const uint32_t *>(s + uint64_t(x) * 4);
                    if (fmt.name[0] == 'R' && fmt.name[1] == '1' && fmt.name[2] == '1')
                    {
                        // R11G11B10_FLOAT decode
                        auto f11 = [](uint32_t e, uint32_t m) {
                            float v;
                            if (e == 0) v = float(m) * ldexpf(1.0f, -14) / 64.0f;
                            else if (e == 31) v = 1.0f;
                            else v = (1.0f + float(m) / 64.0f) * ldexpf(1.0f, int(e) - 15);
                            if (v < 0) v = 0; if (v > 1) v = 1;
                            return uint8_t(v * 255.0f + 0.5f);
                        };
                        auto f10 = [](uint32_t e, uint32_t m) {
                            float v;
                            if (e == 0) v = float(m) * ldexpf(1.0f, -14) / 32.0f;
                            else if (e == 31) v = 1.0f;
                            else v = (1.0f + float(m) / 32.0f) * ldexpf(1.0f, int(e) - 15);
                            if (v < 0) v = 0; if (v > 1) v = 1;
                            return uint8_t(v * 255.0f + 0.5f);
                        };
                        r = f11((px >> 6) & 0x1F, (px >> 0) & 0x3F);
                        g = f11((px >> 17) & 0x1F, (px >> 11) & 0x3F);
                        b = f10((px >> 27) & 0x1F, (px >> 22) & 0x1F);
                    }
                    else if (fmt.name[0] == 'R' && fmt.name[1] == '1' && fmt.name[2] == '0')
                    {
                        r = uint8_t(((px >> 0) & 0x3FF) >> 2);
                        g = uint8_t(((px >> 10) & 0x3FF) >> 2);
                        b = uint8_t(((px >> 20) & 0x3FF) >> 2);
                    }
                    else if (fmt.bgra) { b = uint8_t(px); g = uint8_t(px >> 8); r = uint8_t(px >> 16); }
                    else               { r = uint8_t(px); g = uint8_t(px >> 8); b = uint8_t(px >> 16); }
                }
                rowbuf[x * 3 + 0] = b;
                rowbuf[x * 3 + 1] = g;
                rowbuf[x * 3 + 2] = r;
            }
            fwrite(rowbuf.data(), 1, row, fp);
        }
        fclose(fp);
        Log("[probe] captured %s (%ux%u %s)", pathbmp, w, h, fmt.name);
    }
    else
        Log("[probe] capture failed: cannot open %s", pathbmp);

    if (fopen_s(&fp, pathtxt, "w") == 0 && fp != nullptr)
    {
        AcquireSRWLockShared(&g_lock);
        const TexInfo *t = FindTex(g_capture_handle);
        fprintf(fp, "resource=0x%016llX\nformat=%s (%u)\nsize=%ux%u\nlayers=%u samples=%u\nframe=%llu\n",
                (unsigned long long)g_capture_handle, fmt.name, unsigned(desc.texture.format), w, h,
                desc.texture.depth_or_layers, desc.texture.samples, (unsigned long long)g_frame);
        if (t != nullptr)
            fprintf(fp, "score=%u\ndraws_this_frame=%u\nrt_binds=%u\ndepth=%d\nfeeds_swapchain=%d\nevent_first=%llu\nevent_last=%llu\nframes_seen=%u\ntotal_draws=%llu\n",
                    t->score, t->fr_draws, t->fr_rt_binds, int(t->fr_depth), int(t->fr_feeds_swapchain),
                    (unsigned long long)t->fr_first_event, (unsigned long long)t->fr_last_event,
                    t->frames_seen, (unsigned long long)t->total_draws);
        fprintf(fp, "copy_edges_this_frame=%zu\n", g_edges.size());
        for (const auto &e : g_edges)
            fprintf(fp, "copy 0x%016llX -> 0x%016llX\n", (unsigned long long)e.first, (unsigned long long)e.second);
        ReleaseSRWLockShared(&g_lock);
        fclose(fp);
    }

    g_device->unmap_buffer_region(g_capture_buffer);
}

static void OnRenderTechniqueTick(reshade::api::effect_runtime *rt, reshade::api::command_list *cl, reshade::api::resource_view rtv)
{
    if (!g_enabled)
        return;
    // Track the current backbuffer as the swapchain resource.
    if (g_device != nullptr)
    {
        const reshade::api::resource bb = g_device->get_resource_from_view(rtv);
        MarkSwapchainResource(bb.handle);
    }

    // Manual one-shot capture on the game's immediate command list.
    if (InterlockedCompareExchange(&g_capture_request, 0, 1) == 1 && g_capture_handle != 0 && !g_capture_busy)
    {
        g_capture_busy = true;
        CaptureAndSave(rt, cl);
        g_capture_busy = false;
    }
}

// ---------------------------------------------------------------------------
// Overlay panel (spec sections 9-10)
// ---------------------------------------------------------------------------

static void DrawProbeOverlay(reshade::api::effect_runtime * /*runtime*/)
{
    ImGui::Checkbox("Probe enabled (DLSS feed is inert while on)", (bool *)&g_enabled);
    ImGui::Text("Frame: %llu   Event seq: %llu", (unsigned long long)g_frame, (unsigned long long)g_seq);
    ImGui::Separator();

    if (g_candidate < 1) g_candidate = 1;
    if (g_candidate > 10) g_candidate = 10;
    if (ImGui::Button("<< Prev")) g_candidate = (g_candidate > 1) ? g_candidate - 1 : 1;
    ImGui::SameLine();
    if (ImGui::Button("Next >>")) g_candidate = (g_candidate < 10) ? g_candidate + 1 : 10;
    ImGui::SameLine();
    ImGui::Text("Selected: #%d", g_candidate);

    if (ImGui::Button("Capture selected candidate"))
    {
        AcquireSRWLockShared(&g_lock);
        bool ok = false;
        if (g_candidate >= 1 && g_candidate <= int(g_top.size()))
        {
            g_capture_handle = g_top[g_candidate - 1].handle;
            ok = true;
        }
        ReleaseSRWLockShared(&g_lock);
        if (ok)
            InterlockedExchange(&g_capture_request, 1);
        else
            Log("[probe] capture: no candidate #%d this frame", g_candidate);
    }
    ImGui::Separator();

    if (g_top.empty()) { ImGui::TextDisabled("No candidates yet (needs a few frames with the probe on)"); return; }

    for (size_t i = 0; i < g_top.size(); ++i)
    {
        const Snapshot &s = g_top[i];
        char label[32]; snprintf(label, sizeof(label), "#%u", unsigned(i + 1));
        ImGui::BeginGroup();
        ImGui::Text("%s %s 0x%016llX", label, (int(i + 1) == g_candidate) ? ">" : " ", (unsigned long long)s.handle);
        ImGui::Text("  %ux%u %s layers=%u samples=%u", s.w, s.h, FmtName(s.fmt), s.layers, s.samples);
        ImGui::Text("  draws=%u binds=%u depth=%d first=%llu last=%llu", s.draws, s.binds, int(s.depth),
                    (unsigned long long)s.first, (unsigned long long)s.last);
        ImGui::Text("  feeds_swapchain=%s score=%u", s.feeds ? "yes" : "no", s.score);
        ImGui::EndGroup();
        ImGui::Separator();
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static void ProbeRegisterEvents()
{
    reshade::register_event<reshade::addon_event::init_device>([](reshade::api::device *device) {
        g_device = device;
    });
    reshade::register_event<reshade::addon_event::destroy_device>([](reshade::api::device *device) {
        if (g_device == device)
        {
            if (g_capture_buffer.handle != 0) { /* device is going away */ g_capture_buffer = { 0 }; }
            if (g_capture_fence.handle != 0) { g_capture_fence = { 0 }; }
            g_device = nullptr;
            AcquireSRWLockExclusive(&g_lock);
            g_tex.clear(); g_active.clear(); g_edges.clear(); g_swapchain_res.clear(); g_top.clear();
            g_frame = 0; g_seq = 0;
            ReleaseSRWLockExclusive(&g_lock);
        }
    });
    reshade::register_event<reshade::addon_event::init_resource>(OnResourceInit);
    reshade::register_event<reshade::addon_event::destroy_resource>(OnResourceDestroy);
    reshade::register_event<reshade::addon_event::init_resource_view>(OnResourceViewInit);
    reshade::register_event<reshade::addon_event::begin_render_pass>(OnBeginRenderPass);
    reshade::register_event<reshade::addon_event::end_render_pass>(OnEndRenderPass);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBindRenderTargets);
    reshade::register_event<reshade::addon_event::bind_viewports>(OnViewport);
    reshade::register_event<reshade::addon_event::draw>(OnDraw);
    reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
    reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(OnDrawIndirect);
    reshade::register_event<reshade::addon_event::copy_resource>(OnCopyResource);
    reshade::register_event<reshade::addon_event::copy_texture_region>(OnCopyTextureRegion);
    reshade::register_event<reshade::addon_event::resolve_texture_region>(OnResolveTextureRegion);
    reshade::register_event<reshade::addon_event::present>(OnPresent);
}

} // namespace probe
