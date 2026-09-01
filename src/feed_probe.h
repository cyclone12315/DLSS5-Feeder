// feed_probe.h - Eden Vulkan scene-texture discovery probe.
//
// A diagnostic-only module: watches the game's Vulkan texture resources and
// render-target bindings through the ReShade add-on events, scores candidates
// for "internal scene color before UI", and offers a manual one-shot capture
// (GPU snapshot -> readback buffer -> BMP + metadata under dlss5-probe/).
//
// Hard rules (per the probe spec):
//  - metadata tracking only; no per-frame readback, no VRAM pokes, no CPU
//    memory scanning;
//  - never modifies game rendering -- selecting a candidate only marks it;
//  - capture happens solely on an explicit overlay button press;
//  - the DLSS feeder path is inert while the probe is enabled.
//
// Capture model (staged, per spec section 7):
//  Frame N:   user selects a candidate from last frame's Top-10 snapshot.
//  Frame N+1: when the selected resource transitions AWAY from render_target
//             (observed through the barrier event), a GPU snapshot of its
//             contents is taken at exactly that point (copy into a probe-owned
//             texture). Later, during a ReShade technique tick, the snapshot is
//             copied into the readback buffer, the fence is waited on the CPU,
//             and the BMP + metadata are written.
//  If the trigger never fires, a "late" fallback reads the resource directly
//  using its tracked current state -- and the metadata is explicit about it.
//
// Everything is gated behind the "probe" config key: when it is 0 the event
// handlers return immediately and the only cost is a branch per event.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <atomic>
#include <algorithm>
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

// Set by the host (dlss5-feed.cpp) so the overlay toggle can persist itself
// through CfgSave() without a hard dependency in the other direction.
static void (*g_save_cfg)() = nullptr;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct TexInfo
{
    bool     alive = false;
    bool     is_swapchain = false;   // ONLY set for real ReShade swapchain back buffers

    uint32_t w = 0, h = 0;
    uint32_t mips = 0, layers = 0, samples = 0;
    unsigned fmt = 0;

    // Actual resource state (layout/usage), tracked from the events whose
    // contract guarantees it. NOT the create-time desc.usage (that describes
    // what is *allowed*, not what the resource is *in*).
    reshade::api::resource_usage current_usage = reshade::api::resource_usage::undefined;

    // Cumulative
    bool     ever_rt = false, ever_depth = false, ever_srv = false;
    bool     ever_copy_src = false, ever_copy_dst = false;
    uint64_t create_frame = 0;
    uint32_t frames_seen = 0;
    uint64_t total_draw_calls = 0;
    uint64_t total_draw_weight = 0;

    // Per-frame (reset on present)
    uint32_t fr_draw_calls = 0;               // real draw-call count
    uint32_t fr_draw_weight = 0;              // weighted (1 + verts/1000 etc.) -- NOT call count
    uint32_t fr_rt_binds = 0;
    bool     fr_depth = false, fr_feeds_swapchain = false, fr_copy_to = false;
    bool     fr_had_depth = false;            // this COLOR target was bound while a DSV was active
    uint64_t fr_depth_resource = 0;           // the simultaneously-bound depth-stencil resource
    uint32_t fr_vp_w = 0, fr_vp_h = 0;        // primary viewport
    uint32_t fr_vp_count = 0;                 // viewports bound while this was active
    uint32_t fr_vp_max_w = 0, fr_vp_max_h = 0; // max extent over all viewports
    uint64_t fr_first_event = 0, fr_last_event = 0;
    bool     fr_active = false;

    // Snapshot
    uint32_t score = 0;
};

struct ActiveRt  // per command list; REPLACED on every full binding operation
{
    // Full MRT width: ReShade binding events may report up to 8 render targets.
    uint64_t rt_res[8] = {};
    uint32_t rt_count = 0;
    uint64_t dsv_res = 0;

    // Command-list-local viewport state, kept independently of the attachment
    // set (apps may bind viewport before RT or RT before viewport).
    bool     vp_set = false;
    uint32_t vp_count = 0;
    uint32_t vp_primary_w = 0, vp_primary_h = 0;
    uint32_t vp_max_w = 0, vp_max_h = 0;
};

static SRWLOCK                       g_lock = SRWLOCK_INIT;
static bool                          g_inited = false;
static reshade::api::device         *g_device = nullptr;
static std::atomic<uint64_t>         g_frame { 0 };
// CPU callback recording sequence. NOT a render-graph execution timeline: with
// multiple concurrently recorded command buffers the interleaving across lists
// is arbitrary. Renamed from g_seq to make that explicit.
static std::atomic<uint64_t>         g_record_event_seq { 0 };

static std::unordered_map<uint64_t, TexInfo> g_tex;          // key: resource handle
static std::unordered_map<uint64_t, ActiveRt> g_active;      // key: command list handle
static std::vector<std::pair<uint64_t, uint64_t>> g_edges;   // per-frame copy edges (src, dst)
static std::vector<uint64_t>         g_swapchain_res;        // EXACTLY ReShade's current back buffers

struct Snapshot
{
    uint64_t handle = 0;
    uint32_t w = 0, h = 0;
    unsigned fmt = 0;
    uint32_t draw_calls = 0, draw_weight = 0, binds = 0, score = 0;
    bool     had_depth = false, feeds = false;
    uint64_t depth_handle = 0;
    uint32_t depth_w = 0, depth_h = 0;
    unsigned depth_fmt = 0;
    uint32_t vp_w = 0, vp_h = 0, vp_count = 0;
    uint32_t vp_max_w = 0, vp_max_h = 0;
    uint64_t first = 0, last = 0;
    uint32_t layers = 0, samples = 0;
    uint64_t frame = 0;                        // finalized frame the stats belong to
};
static std::vector<Snapshot> g_top;                          // last frame's top-10

// Staged capture state (guarded by g_lock)
struct PendingCapture
{
    Snapshot snap;           // full selected snapshot (previous-frame stats, kept separate)
    uint64_t armed_frame = 0;
    int      stage = 0;      // 0 = none, 1 = armed (waiting for RT -> read transition),
                             // 2 = GPU snapshot taken, 3 = late fallback requested
};
static PendingCapture g_pending;
static reshade::api::resource   g_snap_res = { 0 };          // probe-owned snapshot texture
static bool                     g_snap_valid = false;

// Readback plumbing
static reshade::api::fence      g_capture_fence = { 0 };
static uint64_t                 g_capture_fence_value = 0;
static reshade::api::resource   g_capture_buffer = { 0 };
static uint64_t                 g_capture_buffer_size = 0;
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
// Swapchain identification (real API only -- never inferred from technique RTVs)
// ---------------------------------------------------------------------------

// Rebuild g_swapchain_res to be EXACTLY the current back buffers of this
// swapchain, and keep their TexInfo entries in sync. Called from init_swapchain
// and present; idempotent and cheap.
static void SyncSwapchainBackbuffers(reshade::api::swapchain *swapchain)
{
    if (swapchain == nullptr)
        return;

    uint64_t backs[8];
    reshade::api::resource_desc descs[8];
    uint32_t n = swapchain->get_back_buffer_count();
    if (n > 8) n = 8;
    for (uint32_t i = 0; i < n; ++i)
    {
        backs[i] = swapchain->get_back_buffer(i).handle;
        if (backs[i] != 0 && g_device != nullptr)
            descs[i] = g_device->get_resource_desc({ backs[i] });
    }

    AcquireSRWLockExclusive(&g_lock);
    // Unmark anything that is no longer a back buffer (stale resize handles).
    for (size_t i = 0; i < g_swapchain_res.size(); )
    {
        const uint64_t h = g_swapchain_res[i];
        bool still = false;
        for (uint32_t b = 0; b < n; ++b) still |= (backs[b] == h);
        if (still) { ++i; continue; }
        if (TexInfo *t = FindTex(h)) t->is_swapchain = false;
        g_swapchain_res[i] = g_swapchain_res.back();
        g_swapchain_res.pop_back();
    }
    // Mark / create entries for the real back buffers.
    for (uint32_t b = 0; b < n; ++b)
    {
        const uint64_t h = backs[b];
        if (h == 0) continue;
        TexInfo *t = FindTex(h);
        if (t == nullptr)
        {
            // A back buffer not observed through init_resource: register it
            // explicitly from its queried resource_desc.
            t = &g_tex[h];
            *t = {};
            t->alive = true;
            t->w = descs[b].texture.width;
            t->h = descs[b].texture.height;
            t->layers = descs[b].texture.depth_or_layers;
            t->mips = descs[b].texture.levels;
            t->samples = descs[b].texture.samples;
            t->fmt = unsigned(descs[b].texture.format);
            t->create_frame = g_frame.load(std::memory_order_relaxed);
            // current_usage stays undefined: ReShade gives no initial state for
            // images created by the presentation engine itself.
        }
        t->alive = true;
        t->is_swapchain = true;
        bool have = false;
        for (uint64_t e : g_swapchain_res) have |= (e == h);
        if (!have) g_swapchain_res.push_back(h);
    }
    ReleaseSRWLockExclusive(&g_lock);
}

// ---------------------------------------------------------------------------
// Event handlers (registered once; every one bails immediately when disabled)
// ---------------------------------------------------------------------------

static void OnInitDevice(reshade::api::device *device)
{
    g_device = device;
}

static void OnDestroyDevice(reshade::api::device *device)
{
    if (g_device == device)
    {
        g_capture_buffer = { 0 };   // device is going away
        g_capture_fence = { 0 };
        g_snap_res = { 0 };
        g_snap_valid = false;
        g_device = nullptr;
        AcquireSRWLockExclusive(&g_lock);
        g_tex.clear(); g_active.clear(); g_edges.clear(); g_swapchain_res.clear(); g_top.clear();
        g_pending = {};
        g_frame.store(0, std::memory_order_relaxed);
        g_record_event_seq.store(0, std::memory_order_relaxed);
        ReleaseSRWLockExclusive(&g_lock);
    }
}

static void OnInitSwapchain(reshade::api::swapchain *swapchain, bool /*resize*/)
{
    // Runs regardless of g_enabled: it is a once-per-creation call and keeps
    // the swapchain set correct even if the probe is toggled on mid-session.
    SyncSwapchainBackbuffers(swapchain);
}

static void OnResourceInit(reshade::api::device *device, const reshade::api::resource_desc &desc,
                           const reshade::api::subresource_data *, reshade::api::resource_usage initial_state,
                           reshade::api::resource resource)
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
    // The initial STATE of the resource (what the event contract guarantees),
    // not desc.usage (which merely lists allowed usages).
    t.current_usage = initial_state;
    t.create_frame = g_frame.load(std::memory_order_relaxed);
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnResourceDestroy(reshade::api::device *, reshade::api::resource resource)
{
    if (!g_enabled)
        return;
    const uint64_t h = resource.handle;
    if (h == 0)
        return;

    bool cancelled_pending = false;

    AcquireSRWLockExclusive(&g_lock);
    // Retire the entry entirely so dead resources never count against the
    // 8192 live limit and a reused handle cannot inherit old metadata.
    g_tex.erase(h);
    for (size_t i = 0; i < g_swapchain_res.size(); ++i)
        if (g_swapchain_res[i] == h) { g_swapchain_res[i] = g_swapchain_res.back(); g_swapchain_res.pop_back(); break; }
    // g_active is keyed by COMMAND LIST, not resource handle: walk every active
    // attachment set and scrub the destroyed resource out of it.
    for (auto it = g_active.begin(); it != g_active.end(); )
    {
        ActiveRt &a = it->second;
        if (a.dsv_res == h) a.dsv_res = 0;
        for (uint32_t i = 0; i < a.rt_count; )
        {
            if (a.rt_res[i] == h)
            {
                for (uint32_t j = i + 1; j < a.rt_count; ++j) a.rt_res[j - 1] = a.rt_res[j];
                --a.rt_count;
            }
            else ++i;
        }
        if (a.rt_count == 0 && a.dsv_res == 0) it = g_active.erase(it);
        else ++it;
    }
    if (g_pending.stage == 1 && g_pending.snap.handle == h)
    {
        g_pending.stage = 0;
        cancelled_pending = true;
    }
    ReleaseSRWLockExclusive(&g_lock);

    if (cancelled_pending)
        Log("[probe] pending capture cancelled: selected resource was destroyed");
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

// Push the command list's current viewport state into a freshly active target.
static void ApplyViewportToTarget(TexInfo *t, const ActiveRt &a)
{
    if (t == nullptr || !a.vp_set)
        return;
    if (a.vp_primary_w != 0) { t->fr_vp_w = a.vp_primary_w; t->fr_vp_h = a.vp_primary_h; }
    t->fr_vp_count = a.vp_count;
    t->fr_vp_max_w = a.vp_max_w;
    t->fr_vp_max_h = a.vp_max_h;
}

// Replace the active attachment set of this command list. Called on every complete
// render-target binding operation (bind_render_targets_and_depth_stencil and
// begin_render_pass) -- the new set REPLACES whatever was active before, otherwise
// draws get attributed to every RT ever bound on this list (the contamination that
// made several 2496x1404 candidates report identical draw counts).
static void SetActiveAttachments(reshade::api::command_list *cmd_list, const uint64_t *rt_res, uint32_t n, uint64_t dsv_res)
{
    AcquireSRWLockExclusive(&g_lock);
    ActiveRt &a = g_active[cmd_list->get_native()];
    a.rt_count = (n < 8) ? n : 8;
    for (uint32_t i = 0; i < a.rt_count; ++i) a.rt_res[i] = rt_res[i];
    a.dsv_res = dsv_res;

    const uint64_t seq = g_record_event_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    for (uint32_t i = 0; i < a.rt_count; ++i)
        if (TexInfo *t = FindTex(a.rt_res[i]))
        {
            t->ever_rt = true;
            t->current_usage = reshade::api::resource_usage::render_target; // guaranteed by the binding contract
            t->fr_rt_binds++;
            t->fr_active = true;
            if (t->fr_first_event == 0) t->fr_first_event = seq;
            t->fr_last_event = seq;
            ApplyViewportToTarget(t, a);
        }
    if (TexInfo *d = FindTex(dsv_res))
    {
        d->ever_depth = true;
        d->current_usage = reshade::api::resource_usage::depth_stencil;
        d->fr_active = true;
        d->fr_depth = true;
        if (d->fr_first_event == 0) d->fr_first_event = seq;
        d->fr_last_event = seq;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static void ClearActiveAttachments(reshade::api::command_list *cmd_list)
{
    AcquireSRWLockExclusive(&g_lock);
    g_active.erase(cmd_list->get_native());
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnBindRenderTargets(reshade::api::command_list *cmd_list, uint32_t count,
                                const reshade::api::resource_view *rtvs, reshade::api::resource_view dsv)
{
    if (!g_enabled || g_device == nullptr)
        return;
    uint64_t rt_res[8] = {};
    uint32_t n = 0;
    for (uint32_t i = 0; i < count && i < 8 && n < 8; ++i)
    {
        if (rtvs[i].handle == 0) continue;
        const reshade::api::resource res = g_device->get_resource_from_view(rtvs[i]);
        if (res.handle != 0) rt_res[n++] = res.handle;
    }
    uint64_t dsv_res = 0;
    if (dsv.handle != 0)
    {
        const reshade::api::resource res = g_device->get_resource_from_view(dsv);
        if (res.handle != 0) dsv_res = res.handle;
    }
    SetActiveAttachments(cmd_list, rt_res, n, dsv_res);
}

static bool OnBeginRenderPass(reshade::api::command_list *cmd_list, uint32_t count,
                              const reshade::api::render_pass_render_target_desc *rts,
                              const reshade::api::render_pass_depth_stencil_desc *ds, reshade::api::render_pass_flags)
{
    if (!g_enabled)
        return false;
    uint64_t rt_res[8] = {};
    uint32_t n = 0;
    for (uint32_t i = 0; i < count && i < 8 && n < 8; ++i)
    {
        if (rts[i].view.handle == 0 || g_device == nullptr) continue;
        const reshade::api::resource res = g_device->get_resource_from_view(rts[i].view);
        if (res.handle != 0) rt_res[n++] = res.handle;
    }
    uint64_t dsv_res = 0;
    if (ds != nullptr && ds->view.handle != 0 && g_device != nullptr)
    {
        const reshade::api::resource res = g_device->get_resource_from_view(ds->view);
        if (res.handle != 0) dsv_res = res.handle;
    }
    SetActiveAttachments(cmd_list, rt_res, n, dsv_res); // a render pass establishes a NEW attachment set
    return false;
}

static bool OnEndRenderPass(reshade::api::command_list *cmd_list)
{
    if (!g_enabled) return false;
    ClearActiveAttachments(cmd_list);
    return false;
}

static void RecordDraw(reshade::api::command_list *cmd_list, uint32_t calls, uint32_t weight)
{
    if (!g_enabled)
        return;
    AcquireSRWLockExclusive(&g_lock);
    const uint64_t seq = g_record_event_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    auto it = g_active.find(cmd_list->get_native());
    if (it != g_active.end())
    {
        for (uint32_t i = 0; i < it->second.rt_count; ++i)
            if (TexInfo *t = FindTex(it->second.rt_res[i]))
            {
                t->fr_draw_calls += calls;
                t->fr_draw_weight += weight;
                t->total_draw_calls += calls;
                t->total_draw_weight += weight;
                if (t->fr_first_event == 0) t->fr_first_event = seq;
                t->fr_last_event = seq;
                // Color <-> depth association: was a depth-stencil resource bound
                // alongside this COLOR target while it was being drawn into?
                if (it->second.dsv_res != 0)
                {
                    t->fr_had_depth = true;
                    t->fr_depth_resource = it->second.dsv_res;
                }
            }
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static bool OnDraw(reshade::api::command_list *cl, uint32_t vertices, uint32_t, uint32_t, uint32_t)
{
    RecordDraw(cl, 1, 1 + vertices / 1000); // weight tiny draws less, huge draws a bit more
    return false;
}

static bool OnDrawIndexed(reshade::api::command_list *cl, uint32_t indices, uint32_t, uint32_t, int32_t, uint32_t)
{
    RecordDraw(cl, 1, 1 + indices / 1000);
    return false;
}

// Only INDIRECT DRAWS count as geometry here. Compute dispatches produce their
// results through different paths and must not inflate the "geometry" signal.
static bool OnDrawOrDispatchIndirect(reshade::api::command_list *cl, reshade::api::indirect_command type,
                                     reshade::api::resource, uint64_t, uint32_t count, uint32_t)
{
    if (!g_enabled)
        return false;
    if (type != reshade::api::indirect_command::draw &&
        type != reshade::api::indirect_command::draw_indexed)
        return false; // dispatch / dispatch_mesh / dispatch_rays: ignored
    RecordDraw(cl, count, count * 10); // indirect draws are typically heavy scene geometry
    return false;
}

static void OnViewport(reshade::api::command_list *cmd_list, uint32_t first, uint32_t count, const reshade::api::viewport *vps)
{
    if (!g_enabled || count == 0 || vps == nullptr)
        return;
    AcquireSRWLockExclusive(&g_lock);
    ActiveRt &a = g_active[cmd_list->get_native()];
    a.vp_set = true;
    if (first == 0 && vps[0].width > 0.0f) { a.vp_primary_w = uint32_t(vps[0].width); a.vp_primary_h = uint32_t(vps[0].height); }
    if (first + count > a.vp_count) a.vp_count = first + count;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (uint32_t(vps[i].width) > a.vp_max_w) a.vp_max_w = uint32_t(vps[i].width);
        if (uint32_t(vps[i].height) > a.vp_max_h) a.vp_max_h = uint32_t(vps[i].height);
    }
    // Viewports may be bound after the RT: push to anything already active.
    for (uint32_t i = 0; i < a.rt_count; ++i)
        ApplyViewportToTarget(FindTex(a.rt_res[i]), a);
    ReleaseSRWLockExclusive(&g_lock);
}

static void RecordCopy(uint64_t src, uint64_t dst)
{
    AcquireSRWLockExclusive(&g_lock);
    const uint64_t seq = g_record_event_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    if (g_edges.size() < 2048) g_edges.push_back({ src, dst });
    if (TexInfo *s = FindTex(src))
    {
        s->ever_copy_src = true; s->fr_active = true;
        s->current_usage = reshade::api::resource_usage::copy_source; // guaranteed by the copy contract
        if (s->fr_first_event == 0) s->fr_first_event = seq;
        s->fr_last_event = seq;
    }
    if (TexInfo *d = FindTex(dst))
    {
        d->ever_copy_dst = true; d->fr_active = true; d->fr_copy_to = true;
        d->current_usage = reshade::api::resource_usage::copy_dest;
        if (d->fr_first_event == 0) d->fr_first_event = seq;
        d->fr_last_event = seq;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static bool OnCopyResource(reshade::api::command_list *, reshade::api::resource src, reshade::api::resource dst)
{
    if (!g_enabled) return false;
    RecordCopy(src.handle, dst.handle);
    return false;
}

static bool OnCopyTextureRegion(reshade::api::command_list *, reshade::api::resource src, uint32_t,
                                const reshade::api::subresource_box *, reshade::api::resource dst, uint32_t,
                                const reshade::api::subresource_box *, reshade::api::filter_mode)
{
    if (!g_enabled) return false;
    RecordCopy(src.handle, dst.handle);
    return false;
}

static bool OnResolveTextureRegion(reshade::api::command_list *, reshade::api::resource src, uint32_t,
                                   const reshade::api::subresource_box *, reshade::api::resource dst, uint32_t,
                                   uint32_t, uint32_t, uint32_t, reshade::api::format)
{
    if (!g_enabled) return false;
    RecordCopy(src.handle, dst.handle);
    return false;
}

// ---------------------------------------------------------------------------
// Barrier tracking: real resource state + the staged-capture trigger
// ---------------------------------------------------------------------------

static void OnBarrier(reshade::api::command_list *cmd_list, uint32_t count,
                      const reshade::api::resource *resources, const reshade::api::resource_usage *old_states,
                      const reshade::api::resource_usage *new_states)
{
    if (!g_enabled || count == 0 || resources == nullptr || new_states == nullptr)
        return;

    // Pass 1: track state, detect the staged-capture trigger.
    bool trigger = false;
    uint64_t trigger_res = 0;
    reshade::api::resource_usage trigger_old = reshade::api::resource_usage::undefined;

    AcquireSRWLockExclusive(&g_lock);
    for (uint32_t i = 0; i < count; ++i)
    {
        const uint64_t h = resources[i].handle;
        if (h == 0) continue;
        if (TexInfo *t = FindTex(h))
        {
            t->current_usage = new_states[i]; // the state the transition will result in

            if (!trigger && g_pending.stage == 1 && h == g_pending.snap.handle && t->alive &&
                (old_states[i] & reshade::api::resource_usage::render_target) != reshade::api::resource_usage::undefined &&
                (new_states[i] & reshade::api::resource_usage::render_target) == reshade::api::resource_usage::undefined)
            {
                trigger = true;
                trigger_res = h;
                trigger_old = old_states[i];
            }
        }
    }
    ReleaseSRWLockExclusive(&g_lock);

    if (!trigger)
        return;

    // Pass 2 (outside the lock): take the GPU snapshot at exactly this point in
    // the recording, BEFORE the app's own transition executes. We temporarily
    // move the resource render_target -> copy_source, copy it into a
    // probe-owned texture, move it back to render_target; the app's own
    // transition (old -> new) then proceeds as if we were never here.
    const reshade::api::resource res = { trigger_res };
    const reshade::api::resource_desc desc = g_device->get_resource_desc(res);
    if (desc.type != reshade::api::resource_type::texture_2d || desc.texture.samples != 1)
    {
        Log("[probe] staged capture aborted: target is not a non-MSAA 2D texture");
        AcquireSRWLockExclusive(&g_lock);
        g_pending.stage = 0;
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }

    reshade::api::resource_desc snap_desc = desc;
    snap_desc.usage = reshade::api::resource_usage::copy_dest;
    reshade::api::resource snap = { 0 };
    if (!g_device->create_resource(snap_desc, nullptr, reshade::api::resource_usage::copy_dest, &snap) || snap.handle == 0)
    {
        Log("[probe] staged capture aborted: snapshot texture creation failed");
        AcquireSRWLockExclusive(&g_lock);
        g_pending.stage = 0;
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }

    const reshade::api::resource_usage from = trigger_old;             // render_target (bit)
    const reshade::api::resource_usage to = reshade::api::resource_usage::copy_source;
    cmd_list->barrier(1, &res, &from, &to);
    cmd_list->copy_resource(res, snap);
    cmd_list->barrier(1, &res, &to, &from);

    AcquireSRWLockExclusive(&g_lock);
    g_snap_res = snap;
    g_snap_valid = true;
    g_pending.stage = 2;
    ReleaseSRWLockExclusive(&g_lock);
    Log("[probe] staged snapshot taken at render_target -> %s transition (frame %llu)",
        FmtName(unsigned(desc.texture.format)), (unsigned long long)g_frame.load(std::memory_order_relaxed));
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
    if (t.fr_draw_weight >= 500) s += 25;
    else if (t.fr_draw_weight >= 100) s += 20;
    else if (t.fr_draw_weight >= 20) s += 10;
    else if (t.fr_draw_weight > 0) s += 5;
    if (t.fr_had_depth) s += 15; // COLOR target drawn with a live depth attachment: strong 3D-scene signal
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
    // NOTE: this only sees copy/resolve edges, NOT SRV sampling -- the result is
    // informational only and must never be a hard rejection criterion.
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

// Deterministic candidate ordering (spec section 9): equal-score candidates
// must not shuffle between runs or frames just because unordered_map iteration
// order differs. Tie-break order:
//   score desc, had_depth desc, draw_weight desc, rt_bind_count desc,
//   area desc, first_event asc, handle asc.
struct CandRow
{
    uint64_t handle = 0;
    uint32_t score = 0, draw_weight = 0, rt_binds = 0, w = 0, h = 0;
    bool had_depth = false;
    uint64_t first_event = 0;
};

static bool CandBetter(const CandRow &a, const CandRow &b)
{
    if (a.score != b.score) return a.score > b.score;
    if (a.had_depth != b.had_depth) return a.had_depth > b.had_depth;
    if (a.draw_weight != b.draw_weight) return a.draw_weight > b.draw_weight;
    if (a.rt_binds != b.rt_binds) return a.rt_binds > b.rt_binds;
    const uint64_t area_a = uint64_t(a.w) * a.h, area_b = uint64_t(b.w) * b.h;
    if (area_a != area_b) return area_a > area_b;
    if (a.first_event != b.first_event) return a.first_event < b.first_event;
    return a.handle < b.handle;
}

static void FinalizeFrame()
{
    uint32_t swapchain_area = 0;
    std::vector<CandRow> cands;
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

            CandRow r;
            r.handle = kv.first;
            r.score = t.score;
            r.draw_weight = t.fr_draw_weight;
            r.rt_binds = t.fr_rt_binds;
            r.w = t.w; r.h = t.h;
            r.had_depth = t.fr_had_depth;
            r.first_event = t.fr_first_event;
            cands.push_back(r);
        }
        std::sort(cands.begin(), cands.end(), CandBetter);
        if (cands.size() > 10) cands.resize(10);

        g_top.clear();
        g_top.reserve(cands.size());
        for (const CandRow &r : cands)
        {
            const TexInfo &t = g_tex[r.handle];
            Snapshot s;
            s.handle = r.handle; s.w = t.w; s.h = t.h; s.fmt = t.fmt;
            s.draw_calls = t.fr_draw_calls; s.draw_weight = t.fr_draw_weight;
            s.binds = t.fr_rt_binds; s.score = t.score;
            s.had_depth = t.fr_had_depth; s.feeds = t.fr_feeds_swapchain;
            s.depth_handle = t.fr_depth_resource;
            if (const TexInfo *d = FindTex(t.fr_depth_resource))
            { s.depth_w = d->w; s.depth_h = d->h; s.depth_fmt = d->fmt; }
            s.vp_w = t.fr_vp_w; s.vp_h = t.fr_vp_h; s.vp_count = t.fr_vp_count;
            s.vp_max_w = t.fr_vp_max_w; s.vp_max_h = t.fr_vp_max_h;
            s.first = t.fr_first_event; s.last = t.fr_last_event;
            s.layers = t.layers; s.samples = t.samples;
            s.frame = g_frame.load(std::memory_order_relaxed);
            g_top.push_back(s);
        }
        ReleaseSRWLockExclusive(&g_lock);
    }

    if (g_log_summary_every != 0 && (g_frame.load(std::memory_order_relaxed) % uint64_t(g_log_summary_every)) == 0)
    {
        uint32_t alive = 0, active = 0;
        AcquireSRWLockShared(&g_lock);
        for (auto &kv : g_tex) { if (kv.second.alive) alive++; if (kv.second.fr_active) active++; }
        if (!g_top.empty())
        {
            const Snapshot &t = g_top[0];
            Log("[probe] frame %llu: tracked=%u active=%u | top: %ux%u %s draw_calls=%u draw_weight=%u had_depth=%d feeds_swapchain=%d score=%u",
                (unsigned long long)g_frame.load(std::memory_order_relaxed), alive, active, t.w, t.h, FmtName(t.fmt),
                t.draw_calls, t.draw_weight, int(t.had_depth), int(t.feeds), t.score);
        }
        else
            Log("[probe] frame %llu: tracked=%u active=%u | no candidates this frame",
                (unsigned long long)g_frame.load(std::memory_order_relaxed), alive, active);
        ReleaseSRWLockShared(&g_lock);
    }

    // Reset per-frame stats
    AcquireSRWLockExclusive(&g_lock);
    for (auto &kv : g_tex)
    {
        TexInfo &t = kv.second;
        t.fr_draw_calls = t.fr_draw_weight = t.fr_rt_binds = 0;
        t.fr_depth = t.fr_feeds_swapchain = t.fr_copy_to = false;
        t.fr_had_depth = false;
        t.fr_depth_resource = 0;
        t.fr_vp_w = t.fr_vp_h = t.fr_vp_count = t.fr_vp_max_w = t.fr_vp_max_h = 0;
        t.fr_first_event = t.fr_last_event = 0;
        t.fr_active = false;
    }
    g_edges.clear();
    g_active.clear();
    g_record_event_seq.store(0, std::memory_order_relaxed);
    ReleaseSRWLockExclusive(&g_lock);
    g_frame.fetch_add(1, std::memory_order_relaxed);
}

static void OnPresent(reshade::api::command_queue *, reshade::api::swapchain *swapchain, const reshade::api::rect *,
                      const reshade::api::rect *, uint32_t, const reshade::api::rect *)
{
    if (!g_enabled) return;
    // Keep the swapchain set exact even if init_swapchain was missed (probe
    // enabled mid-session): idempotent, two virtual calls per present.
    SyncSwapchainBackbuffers(swapchain);
    FinalizeFrame();
}

// ---------------------------------------------------------------------------
// One-shot capture (staged model, spec sections 7, 8 and 13)
// ---------------------------------------------------------------------------

struct FmtInfo { unsigned f; const char *name; uint32_t bpp; bool bgra; };

static bool LookupCaptureFormat(unsigned raw, FmtInfo &out)
{
    static const FmtInfo known[] = {
        { unsigned(reshade::api::format::r8g8b8a8_unorm),          "R8G8B8A8_UNORM", 32, false },
        { unsigned(reshade::api::format::b8g8r8a8_unorm),          "B8G8R8A8_UNORM", 32, true },
        { unsigned(reshade::api::format::r8g8b8a8_unorm_srgb),     "R8G8B8A8_SRGB",  32, false },
        { unsigned(reshade::api::format::b8g8r8a8_unorm_srgb),     "B8G8R8A8_SRGB",  32, true },
        // TYPELESS resources are captured interpreted as the matching UNORM
        // family; if the real backing store is the opposite channel order the
        // BMP will show swapped R/B -- still diagnostic gold.
        { unsigned(reshade::api::format::r8g8b8a8_typeless),       "R8G8B8A8_TYPELESS(as RGBA8)", 32, false },
        { unsigned(reshade::api::format::b8g8r8a8_typeless),       "B8G8R8A8_TYPELESS(as BGRA8)", 32, true },
        { unsigned(reshade::api::format::r10g10b10a2_typeless),    "R10G10B10A2_TYPELESS", 32, false },
        { unsigned(reshade::api::format::r10g10b10a2_unorm),       "R10G10B10A2_UNORM", 32, false },
        { unsigned(reshade::api::format::r11g11b10_float),         "R11G11B10_FLOAT", 32, false },
        { unsigned(reshade::api::format::r16g16b16a16_float),      "R16G16B16A16_FLOAT", 64, false },
    };
    for (const FmtInfo &k : known) if (raw == k.f) { out = k; return true; }
    return false;
}

// Write the BMP + metadata from a mapped readback buffer (tightly packed rows:
// the copy used row_length = 0 / slice_height = 0, so pitch is w * bytes/px).
static void WriteCaptureOutput(const char *pathbmp, const char *pathtxt, const FmtInfo &fi,
                               uint32_t w, uint32_t h, uint64_t pitch, const void *mapped,
                               uint64_t target_handle, const char *stage, const Snapshot &sel,
                               const TexInfo *live)
{
    const bool f16 = fi.bpp == 64;
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
                    if (fi.name[0] == 'R' && fi.name[1] == '1' && fi.name[2] == '1')
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
                    else if (fi.name[0] == 'R' && fi.name[1] == '1' && fi.name[2] == '0')
                    {
                        r = uint8_t(((px >> 0) & 0x3FF) >> 2);
                        g = uint8_t(((px >> 10) & 0x3FF) >> 2);
                        b = uint8_t(((px >> 20) & 0x3FF) >> 2);
                    }
                    else if (fi.bgra) { b = uint8_t(px); g = uint8_t(px >> 8); r = uint8_t(px >> 16); }
                    else              { r = uint8_t(px); g = uint8_t(px >> 8); b = uint8_t(px >> 16); }
                }
                rowbuf[x * 3 + 0] = b;
                rowbuf[x * 3 + 1] = g;
                rowbuf[x * 3 + 2] = r;
            }
            fwrite(rowbuf.data(), 1, row, fp);
        }
        fclose(fp);
        Log("[probe] captured %s (%ux%u %s, stage=%s)", pathbmp, w, h, fi.name, stage);
    }
    else
    {
        Log("[probe] capture failed: cannot open %s", pathbmp);
        return;
    }

    // Metadata: previous-frame ranking stats ("selected_snapshot") are kept
    // strictly separate from the resource's live state at capture time.
    if (fopen_s(&fp, pathtxt, "w") == 0 && fp != nullptr)
    {
        fprintf(fp, "[resource]\nhandle=0x%016llX\nformat=%s (%u)\nsize=%ux%u\nlayers=%u samples=%u\ncapture_stage=%s\n",
                (unsigned long long)target_handle, fi.name, fi.f, w, h, sel.layers, sel.samples, stage);
        if (std::strcmp(stage, "late") == 0)
            fprintf(fp, "warning=late snapshot: the stage trigger never fired; contents may have been overwritten by later passes\n");
        fprintf(fp, "\n[selected_snapshot]\nframe=%llu\nscore=%u\ndraw_calls=%u\ndraw_weight=%u\nrt_binds=%u\n"
                    "had_depth=%d\ndepth_resource=0x%016llX\ndepth_size=%ux%u\ndepth_format=%s\n"
                    "viewport=%ux%u\nviewport_count=%u\nviewport_max=%ux%u\n"
                    "event_first=%llu\nevent_last=%llu\nfeeds_swapchain_copygraph=%d\n",
                (unsigned long long)sel.frame, sel.score, sel.draw_calls, sel.draw_weight, sel.binds,
                int(sel.had_depth), (unsigned long long)sel.depth_handle, sel.depth_w, sel.depth_h, FmtName(sel.depth_fmt),
                sel.vp_w, sel.vp_h, sel.vp_count, sel.vp_max_w, sel.vp_max_h,
                (unsigned long long)sel.first, (unsigned long long)sel.last, int(sel.feeds));
        if (live != nullptr)
        {
            fprintf(fp, "\n[capture_time]\nframe=%llu\ndraw_calls=%u\ndraw_weight=%u\nrt_binds=%u\nhad_depth=%d\n"
                        "depth_resource=0x%016llX\ncurrent_usage=0x%X\nframes_seen=%u\n"
                        "total_draw_calls=%llu\ntotal_draw_weight=%llu\n",
                    (unsigned long long)g_frame.load(std::memory_order_relaxed),
                    live->fr_draw_calls, live->fr_draw_weight, live->fr_rt_binds, int(live->fr_had_depth),
                    (unsigned long long)live->fr_depth_resource, unsigned(live->current_usage),
                    live->frames_seen, (unsigned long long)live->total_draw_calls,
                    (unsigned long long)live->total_draw_weight);
        }
        fprintf(fp, "\n[copy_graph]\ncopy_edges_this_frame=%zu\n", g_edges.size());
        for (const auto &e : g_edges)
            fprintf(fp, "copy 0x%016llX -> 0x%016llX\n", (unsigned long long)e.first, (unsigned long long)e.second);
        fprintf(fp, "note=copy graph sees copy/resolve edges only, not SRV sampling; feeds_swapchain is informational\n");
        fclose(fp);
    }
}

// Read back `copy_res` (currently in `from_state`) and write BMP + metadata.
// All validity checks happen BEFORE any barrier is recorded, so an unsupported
// target never leaves the resource in a transitional state. The caller
// restores nothing -- this function records the restore barrier itself.
static void ReadbackAndSave(reshade::api::effect_runtime *rt, reshade::api::command_list *cl,
                            reshade::api::resource copy_res, uint64_t meta_handle,
                            reshade::api::resource_usage from_state, reshade::api::resource_usage restore_state,
                            const char *stage)
{
    const reshade::api::resource_desc desc = g_device->get_resource_desc(copy_res);
    if (desc.type != reshade::api::resource_type::texture_2d || desc.texture.samples != 1)
    {
        Log("[probe] capture skipped: not a 2D non-MSAA texture");
        return;
    }

    FmtInfo fi = {};
    if (!LookupCaptureFormat(unsigned(desc.texture.format), fi))
    {
        Log("[probe] capture unsupported format: %s (%u)", FmtName(unsigned(desc.texture.format)), unsigned(desc.texture.format));
        return;
    }

    const uint32_t w = desc.texture.width, h = desc.texture.height;
    // row_length = 0 means tightly packed rows: pitch is exactly w * bytes/px.
    // No invented 256-byte alignment (that corrupts widths like 1248 or 1719).
    const uint64_t pitch = uint64_t(w) * (fi.bpp / 8);
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

    const reshade::api::resource_usage copy_src = reshade::api::resource_usage::copy_source;
    cl->barrier(1, &copy_res, &from_state, &copy_src);
    cl->copy_texture_to_buffer(copy_res, 0, nullptr, g_capture_buffer, 0, 0, 0);

    reshade::api::command_queue *queue = rt->get_command_queue();
    queue->flush_immediate_command_list();
    const uint64_t v = ++g_capture_fence_value;
    queue->signal(g_capture_fence, v);       // GPU-side: signals after prior work (the copy) finishes
    // CPU-blocking wait (device::wait, not command_queue::wait -- the latter only
    // enqueues a GPU-side wait and returns immediately).
    if (!g_device->wait(g_capture_fence, v))
    {
        // GPU state is broken beyond our repair; do not issue further barriers.
        Log("[probe] capture failed: fence wait timed out");
        return;
    }

    void *mapped = nullptr;
    if (!g_device->map_buffer_region(g_capture_buffer, 0, buf_size, reshade::api::map_access::read_only, &mapped) || mapped == nullptr)
    {
        cl->barrier(1, &copy_res, &copy_src, &restore_state);
        Log("[probe] capture failed: buffer map");
        return;
    }

    char dir[] = "dlss5-probe";
    CreateDirectoryA(dir, nullptr);
    char pathbmp[MAX_PATH], pathtxt[MAX_PATH];
    snprintf(pathbmp, sizeof(pathbmp), "%s/frame_%06llu_candidate_%02u_%ux%u.bmp", dir,
             (unsigned long long)g_frame.load(std::memory_order_relaxed), g_candidate, w, h);
    snprintf(pathtxt, sizeof(pathtxt), "%s/frame_%06llu_candidate_%02u.txt", dir,
             (unsigned long long)g_frame.load(std::memory_order_relaxed), g_candidate);

    Snapshot sel;
    TexInfo live_copy = {};
    bool have_live = false;
    {
        AcquireSRWLockShared(&g_lock);
        sel = g_pending.snap;
        if (const TexInfo *t = FindTex(meta_handle)) { live_copy = *t; have_live = true; }
        ReleaseSRWLockShared(&g_lock);
    }
    WriteCaptureOutput(pathbmp, pathtxt, fi, w, h, pitch, mapped, meta_handle, stage, sel,
                       have_live ? &live_copy : nullptr);

    g_device->unmap_buffer_region(g_capture_buffer);
    cl->barrier(1, &copy_res, &copy_src, &restore_state);
}

// Called from the ReShade technique tick: performs the readback for a staged
// snapshot, or the explicit "late" fallback (with its warning in the metadata).
static void OnRenderTechniqueTick(reshade::api::effect_runtime *rt, reshade::api::command_list *cl, reshade::api::resource_view /*rtv*/)
{
    int stage = 0;
    {
        AcquireSRWLockShared(&g_lock);
        stage = g_pending.stage;
        ReleaseSRWLockShared(&g_lock);
    }
    if (stage == 0 || g_capture_busy || g_device == nullptr)
        return;

    // Timeout: the resource never left render_target state after 600 frames.
    if (stage == 1 &&
        g_frame.load(std::memory_order_relaxed) - g_pending.armed_frame > 600)
    {
        AcquireSRWLockExclusive(&g_lock);
        stage = g_pending.stage; // re-check under lock
        if (stage == 1)
        {
            const TexInfo *t = FindTex(g_pending.snap.handle);
            const bool state_known = t != nullptr && t->alive &&
                t->current_usage != reshade::api::resource_usage::undefined;
            if (state_known)
                g_pending.stage = 3; // late fallback with tracked current state
            else
            {
                g_pending.stage = 0;
                Log("[probe] capture abandoned: stage trigger never fired and resource state is unknown");
            }
        }
        stage = g_pending.stage;
        ReleaseSRWLockExclusive(&g_lock);
    }

    g_capture_busy = true;
    if (stage == 2)
    {
        // Staged path: read back the probe-owned snapshot. Its state is exactly
        // known (copy_dest -- we created and tracked it ourselves); no guessing.
        uint64_t meta_handle = 0;
        reshade::api::resource snap = { 0 };
        {
            AcquireSRWLockExclusive(&g_lock);
            snap = g_snap_res;
            meta_handle = g_pending.snap.handle;
            g_pending.stage = 0;
            g_snap_res = { 0 };
            g_snap_valid = false;
            ReleaseSRWLockExclusive(&g_lock);
        }
        if (snap.handle != 0)
        {
            ReadbackAndSave(rt, cl, snap, meta_handle,
                            reshade::api::resource_usage::copy_dest,
                            reshade::api::resource_usage::copy_dest, "staged");
            g_device->destroy_resource(snap);
        }
    }
    else if (stage == 3)
    {
        // Late fallback: read the ORIGINAL resource using its TRACKED current
        // state (never a guessed one). Metadata carries the late warning.
        uint64_t h = 0;
        reshade::api::resource_usage from = reshade::api::resource_usage::undefined;
        {
            AcquireSRWLockExclusive(&g_lock);
            h = g_pending.snap.handle;
            if (const TexInfo *t = FindTex(h)) from = t->current_usage;
            g_pending.stage = 0;
            ReleaseSRWLockExclusive(&g_lock);
        }
        if (h != 0 && from != reshade::api::resource_usage::undefined)
            ReadbackAndSave(rt, cl, { h }, h, from, from, "late");
        else
            Log("[probe] capture deferred/skipped: resource state unknown");
    }
    g_capture_busy = false;
}

// ---------------------------------------------------------------------------
// Overlay panel (spec sections 9-10)
// ---------------------------------------------------------------------------

static void DrawProbeOverlay(reshade::api::effect_runtime * /*runtime*/)
{
    bool en = g_enabled != 0;
    if (ImGui::Checkbox("Probe enabled (DLSS feed is inert while on)", &en))
    {
        g_enabled = en ? 1 : 0;
        if (g_save_cfg != nullptr) g_save_cfg();
    }
    ImGui::Text("Frame: %llu   Record event seq: %llu",
                (unsigned long long)g_frame.load(std::memory_order_relaxed),
                (unsigned long long)g_record_event_seq.load(std::memory_order_relaxed));
    ImGui::TextDisabled("(record seq is CPU callback order, not GPU execution order)");
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
        int status = 0; // 1 = armed, 2 = busy, 3 = dead handle, 4 = no such candidate
        {
            AcquireSRWLockExclusive(&g_lock);
            if (g_pending.stage != 0)
                status = 2;
            else if (g_candidate >= 1 && g_candidate <= int(g_top.size()))
            {
                const Snapshot &s = g_top[g_candidate - 1];
                const TexInfo *t = FindTex(s.handle);
                if (t != nullptr && t->alive)
                {
                    g_pending.snap = s;
                    g_pending.armed_frame = g_frame.load(std::memory_order_relaxed);
                    g_pending.stage = 1;
                    status = 1;
                }
                else status = 3;
            }
            else status = 4;
            ReleaseSRWLockExclusive(&g_lock);
        }
        if (status == 1) Log("[probe] capture armed: waiting for candidate to leave render_target state");
        else if (status == 2) Log("[probe] capture already pending");
        else if (status == 3) Log("[probe] capture rejected: candidate resource no longer alive (stale ranking entry)");
        else Log("[probe] capture: no candidate #%d this frame", g_candidate);
    }

    {
        AcquireSRWLockShared(&g_lock);
        if (g_pending.stage == 1) { ImGui::Text("Staged capture armed: waiting for render_target -> read transition..."); }
        else if (g_pending.stage == 2) { ImGui::Text("GPU snapshot taken; saving at next technique tick..."); }
        else if (g_pending.stage == 3) { ImGui::Text("Late fallback pending (trigger never fired)..."); }
        ReleaseSRWLockShared(&g_lock);
    }
    ImGui::Separator();

    if (g_top.empty()) { ImGui::TextDisabled("No candidates yet (needs a few frames with the probe on)"); return; }

    for (size_t i = 0; i < g_top.size(); ++i)
    {
        const Snapshot &s = g_top[i];
        char label[32]; snprintf(label, sizeof(label), "#%u", unsigned(i + 1));
        ImGui::BeginGroup();
        ImGui::Text("%s %s 0x%016llX", label, (int(i + 1) == g_candidate) ? ">" : " ", (unsigned long long)s.handle);
        ImGui::Text("  %ux%u %s layers=%u samples=%u viewport=%ux%u x%u", s.w, s.h, FmtName(s.fmt), s.layers, s.samples, s.vp_w, s.vp_h, s.vp_count);
        ImGui::Text("  draw_calls=%u draw_weight=%u binds=%u first=%llu last=%llu",
                    s.draw_calls, s.draw_weight, s.binds, (unsigned long long)s.first, (unsigned long long)s.last);
        if (s.had_depth && s.depth_handle != 0)
            ImGui::Text("  depth: 0x%016llX (%ux%u %s)", (unsigned long long)s.depth_handle, s.depth_w, s.depth_h, FmtName(s.depth_fmt));
        else
            ImGui::TextDisabled("  depth: none");
        ImGui::Text("  copy-graph-feeds-swapchain=%s (informational only) score=%u", s.feeds ? "yes" : "no", s.score);
        ImGui::EndGroup();
        ImGui::Separator();
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static void ProbeRegisterEvents()
{
    reshade::register_event<reshade::addon_event::init_device>(OnInitDevice);
    reshade::register_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
    reshade::register_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
    reshade::register_event<reshade::addon_event::init_resource>(OnResourceInit);
    reshade::register_event<reshade::addon_event::destroy_resource>(OnResourceDestroy);
    reshade::register_event<reshade::addon_event::init_resource_view>(OnResourceViewInit);
    reshade::register_event<reshade::addon_event::barrier>(OnBarrier);
    reshade::register_event<reshade::addon_event::begin_render_pass>(OnBeginRenderPass);
    reshade::register_event<reshade::addon_event::end_render_pass>(OnEndRenderPass);
    reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(OnBindRenderTargets);
    reshade::register_event<reshade::addon_event::bind_viewports>(OnViewport);
    reshade::register_event<reshade::addon_event::draw>(OnDraw);
    reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
    reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(OnDrawOrDispatchIndirect);
    reshade::register_event<reshade::addon_event::copy_resource>(OnCopyResource);
    reshade::register_event<reshade::addon_event::copy_texture_region>(OnCopyTextureRegion);
    reshade::register_event<reshade::addon_event::resolve_texture_region>(OnResolveTextureRegion);
    reshade::register_event<reshade::addon_event::present>(OnPresent);
}

} // namespace probe
