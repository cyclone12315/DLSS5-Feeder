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
//  Frame N+1: when the selected safe capture-point occurrence is observed
//             through the after-barrier event, the snapshot and texture-to-
//             buffer copy are recorded in that same application command list. Submission is
//             tracked explicitly; a queue fence is waited before the buffer is
//             mapped and the BMP + metadata are written. There is no guessed
//             late-state fallback.
//
// Resource and command-list lifetime events remain active while disabled so a
// later enable cannot inherit stale handles. High-frequency draw/copy/barrier
// accounting remains gated behind the "probe" config key.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <imgui.h>
#include <reshade.hpp>

namespace probe {

// ---------------------------------------------------------------------------
// Config (persisted in dlss5-feed.cfg by the host translation unit)
// ---------------------------------------------------------------------------

static std::atomic<int> g_enabled { 0 }; // "probe" key: 1 = discovery active, DLSS feed inert
static std::atomic<int> g_candidate { 1 }; // "probe_candidate" key: selected rank, 1-based
static std::atomic<int> g_capture_point { 1 }; // 0 = explicit RT exit, 1 = pre-RT reuse
static std::atomic<int> g_capture_occurrence { 1 }; // selected safe point occurrence, 1-based
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
    bool     copy_source_capable = false;

    uint32_t w = 0, h = 0;
    uint32_t mips = 0, layers = 0, samples = 0;
    unsigned fmt = 0;

    // Last state from the initial-state contract or a submitted command-list
    // journal. This is a hint across multiple queues, not a license to issue a
    // barrier; unknown implicit render-pass final layouts are left undefined.
    reshade::api::resource_usage submitted_usage_hint = reshade::api::resource_usage::undefined;

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
    static constexpr uint32_t kMaxViewports = 16;
    bool     vp_set = false;
    bool     vp_valid[kMaxViewports] = {};
    uint32_t vp_w[kMaxViewports] = {}, vp_h[kMaxViewports] = {};
    uint32_t vp_count = 0;
    uint32_t vp_primary_w = 0, vp_primary_h = 0;
    uint32_t vp_max_w = 0, vp_max_h = 0;
};

struct ResourceDelta
{
    uint32_t draw_calls = 0, draw_weight = 0, rt_binds = 0;
    bool active = false, depth = false, had_depth = false;
    uint64_t depth_resource = 0;
    uint32_t vp_w = 0, vp_h = 0, vp_count = 0, vp_max_w = 0, vp_max_h = 0;
    uint64_t first_event = 0, last_event = 0;
};

struct CommandRecord
{
    ActiveRt active;
    bool in_render_pass = false;
    std::unordered_map<uint64_t, ResourceDelta> deltas;
    std::unordered_map<uint64_t, reshade::api::resource_usage> final_states;
    std::vector<std::pair<uint64_t, uint64_t>> edges;
};

static SRWLOCK                       g_lock = SRWLOCK_INIT;
static bool                          g_inited = false;
static reshade::api::device         *g_device = nullptr;
static std::atomic<uint64_t>         g_frame { 0 };
// CPU callback recording sequence. NOT a render-graph execution timeline: with
// multiple concurrently recorded command buffers the interleaving across lists
// is arbitrary. Renamed from g_seq to make that explicit.
static std::atomic<uint64_t>         g_record_event_seq { 0 };

static std::unordered_map<uint64_t, TexInfo> g_tex;          // key: application resource handle
static std::unordered_set<uint64_t> g_private_resources;     // probe-owned snapshot/readback resources
static std::unordered_map<uint64_t, CommandRecord> g_commands; // recorded journals, keyed by command list
static std::unordered_map<uint64_t, std::vector<reshade::api::resource>> g_retired_resources;
static std::vector<std::pair<uint64_t, uint64_t>> g_edges;   // submitted copy edges in this present interval
static std::vector<uint64_t>         g_swapchain_res;        // EXACTLY ReShade's current back buffers

struct Snapshot
{
    uint64_t handle = 0;
    uint32_t w = 0, h = 0;
    unsigned fmt = 0;
    uint32_t draw_calls = 0, draw_weight = 0, binds = 0, score = 0;
    bool     had_depth = false, feeds = false, copy_source_capable = false;
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
    uint32_t candidate_rank = 0;
    uint32_t capture_point = 1, desired_occurrence = 1, observed_points = 0;
    uint32_t all_barriers = 0, observed_exits = 0, observed_reuses = 0;
    uint32_t render_pass_begins = 0, render_pass_ends = 0;
    uint32_t skipped_inside_render_pass = 0, skipped_without_submitted_draw = 0;
    reshade::api::resource_usage last_old_state = reshade::api::resource_usage::undefined;
    reshade::api::resource_usage last_new_state = reshade::api::resource_usage::undefined;
    reshade::api::resource_usage trigger_old_state = reshade::api::resource_usage::undefined;
    reshade::api::resource_usage trigger_new_state = reshade::api::resource_usage::undefined;
    uint64_t trigger_frame = 0;
    uint64_t source_cmd = 0;      // command list that owns the recorded capture commands
    uint64_t submission_cmd = 0;  // primary command list whose submission executes them
    reshade::api::command_queue *source_queue = nullptr;
    int      stage = 0;      // 0 = none, 1 = armed, 2 = recording claimed,
                             // 3 = recorded, 4 = submitted, 5 = completing
};
static PendingCapture g_pending;

enum class CaptureResultStatus : uint32_t { none, success, timeout, failed, cancelled };
struct LastCaptureResult
{
    CaptureResultStatus status = CaptureResultStatus::none;
    PendingCapture capture;
    uint64_t result_frame = 0;
    char reason[96] = {};
};
static LastCaptureResult g_last_capture;
static reshade::api::resource   g_snap_res = { 0 };          // probe-owned snapshot texture
static bool                     g_snap_valid = false;

// Readback plumbing
static reshade::api::fence      g_capture_fence = { 0 };
static uint64_t                 g_capture_fence_value = 0;
static reshade::api::resource   g_capture_buffer = { 0 };
static uint64_t                 g_capture_buffer_size = 0;

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

static const char *CapturePointName(uint32_t point)
{
    return point == 0 ? "explicit_rt_exit" : "pre_rt_reuse";
}

static const char *CaptureResultName(CaptureResultStatus status)
{
    switch (status)
    {
    case CaptureResultStatus::success: return "SUCCESS";
    case CaptureResultStatus::timeout: return "TIMEOUT";
    case CaptureResultStatus::failed: return "FAILED";
    case CaptureResultStatus::cancelled: return "CANCELLED";
    default: return "NONE";
    }
}

// Caller holds g_lock exclusively. Keep the terminal diagnostics after the
// active request is cleared so the overlay can explain what happened.
static void StoreLastCaptureResultLocked(CaptureResultStatus status, const char *reason)
{
    g_last_capture.status = status;
    g_last_capture.capture = g_pending;
    g_last_capture.result_frame = g_frame.load(std::memory_order_relaxed);
    strncpy_s(g_last_capture.reason, reason != nullptr ? reason : "", _TRUNCATE);
}

static void ClearPendingWithResultLocked(CaptureResultStatus status, const char *reason)
{
    StoreLastCaptureResultLocked(status, reason);
    g_pending = {};
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
            // submitted_usage_hint stays undefined: ReShade gives no initial state for
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
        g_tex.clear(); g_private_resources.clear(); g_commands.clear(); g_retired_resources.clear();
        g_edges.clear(); g_swapchain_res.clear(); g_top.clear();
        g_pending = {}; g_last_capture = {};
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

static bool OnCreateResource(reshade::api::device *, reshade::api::resource_desc &desc,
                             reshade::api::subresource_data *, reshade::api::resource_usage)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0 ||
        desc.type != reshade::api::resource_type::texture_2d || desc.texture.samples != 1 ||
        (desc.usage & reshade::api::resource_usage::render_target) == reshade::api::resource_usage::undefined ||
        (desc.usage & reshade::api::resource_usage::copy_source) != reshade::api::resource_usage::undefined)
        return false;
    // Vulkan images need VK_IMAGE_USAGE_TRANSFER_SRC_BIT at creation time.
    // This adds capture capability without changing contents or render state.
    desc.usage |= reshade::api::resource_usage::copy_source;
    return true;
}

static void OnResourceInit(reshade::api::device *device, const reshade::api::resource_desc &desc,
                           const reshade::api::subresource_data *, reshade::api::resource_usage initial_state,
                           reshade::api::resource resource)
{
    if (desc.type != reshade::api::resource_type::texture_2d)
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
    t.copy_source_capable =
        (desc.usage & reshade::api::resource_usage::copy_source) != reshade::api::resource_usage::undefined;
    // The initial STATE of the resource (what the event contract guarantees),
    // not desc.usage (which merely lists allowed usages).
    t.submitted_usage_hint = initial_state;
    t.create_frame = g_frame.load(std::memory_order_relaxed);
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnResourceDestroy(reshade::api::device *, reshade::api::resource resource)
{
    const uint64_t h = resource.handle;
    if (h == 0)
        return;

    bool cancelled_pending = false;

    AcquireSRWLockExclusive(&g_lock);
    g_private_resources.erase(h);
    // Retire the entry entirely so dead resources never count against the
    // 8192 live limit and a reused handle cannot inherit old metadata.
    g_tex.erase(h);
    for (size_t i = 0; i < g_swapchain_res.size(); ++i)
        if (g_swapchain_res[i] == h) { g_swapchain_res[i] = g_swapchain_res.back(); g_swapchain_res.pop_back(); break; }
    // Scrub the resource from every recorded command-list journal. This keeps
    // handle reuse from inheriting old attachment or delta data.
    for (auto &entry : g_commands)
    {
        CommandRecord &record = entry.second;
        ActiveRt &a = record.active;
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
        record.deltas.erase(h);
        record.final_states.erase(h);
        record.edges.erase(std::remove_if(record.edges.begin(), record.edges.end(),
            [h](const auto &edge) { return edge.first == h || edge.second == h; }), record.edges.end());
    }
    if (g_pending.stage == 1 && g_pending.snap.handle == h)
    {
        ClearPendingWithResultLocked(CaptureResultStatus::cancelled, "selected resource was destroyed");
        cancelled_pending = true;
    }
    ReleaseSRWLockExclusive(&g_lock);

    if (cancelled_pending)
        Log("[probe] pending capture cancelled: selected resource was destroyed");
}

static void OnResourceViewInit(reshade::api::device *device, reshade::api::resource resource,
                               reshade::api::resource_usage usage_type, const reshade::api::resource_view_desc &, reshade::api::resource_view)
{
    if (resource.handle == 0)
        return;
    AcquireSRWLockExclusive(&g_lock);
    if (TexInfo *t = FindTex(resource.handle))
    {
        if ((usage_type & reshade::api::resource_usage::depth_stencil) != reshade::api::resource_usage::undefined) t->ever_depth = true;
        if ((usage_type & reshade::api::resource_usage::render_target) != reshade::api::resource_usage::undefined) t->ever_rt = true;
        if ((usage_type & reshade::api::resource_usage::shader_resource) != reshade::api::resource_usage::undefined) t->ever_srv = true;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

// Push the command list's current viewport state into a recorded target delta.
static void ApplyViewportToTarget(ResourceDelta &delta, const ActiveRt &a)
{
    if (!a.vp_set)
        return;
    if (a.vp_primary_w != 0) { delta.vp_w = a.vp_primary_w; delta.vp_h = a.vp_primary_h; }
    delta.vp_count = std::max(delta.vp_count, a.vp_count);
    delta.vp_max_w = std::max(delta.vp_max_w, a.vp_max_w);
    delta.vp_max_h = std::max(delta.vp_max_h, a.vp_max_h);
}

static void RecomputeViewportState(ActiveRt &a)
{
    a.vp_set = false;
    a.vp_count = a.vp_primary_w = a.vp_primary_h = a.vp_max_w = a.vp_max_h = 0;
    for (uint32_t i = 0; i < ActiveRt::kMaxViewports; ++i)
    {
        if (!a.vp_valid[i]) continue;
        a.vp_set = true;
        a.vp_count = i + 1;
        if (i == 0) { a.vp_primary_w = a.vp_w[i]; a.vp_primary_h = a.vp_h[i]; }
        a.vp_max_w = std::max(a.vp_max_w, a.vp_w[i]);
        a.vp_max_h = std::max(a.vp_max_h, a.vp_h[i]);
    }
}

// Replace the active attachment set of this command list. Called on every complete
// render-target binding operation (bind_render_targets_and_depth_stencil and
// begin_render_pass) -- the new set REPLACES whatever was active before, otherwise
// draws get attributed to every RT ever bound on this list (the contamination that
// made several 2496x1404 candidates report identical draw counts).
static void SetActiveAttachments(reshade::api::command_list *cmd_list, const uint64_t *rt_res, uint32_t n, uint64_t dsv_res)
{
    AcquireSRWLockExclusive(&g_lock);
    CommandRecord &record = g_commands[cmd_list->get_native()];
    ActiveRt &a = record.active;
    a.rt_count = (n < 8) ? n : 8;
    for (uint32_t i = 0; i < a.rt_count; ++i) a.rt_res[i] = rt_res[i];
    a.dsv_res = dsv_res;

    const uint64_t seq = g_record_event_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    for (uint32_t i = 0; i < a.rt_count; ++i)
    {
        const uint64_t h = a.rt_res[i];
        if (TexInfo *t = FindTex(h)) t->ever_rt = true;
        ResourceDelta &delta = record.deltas[h];
        delta.rt_binds++;
        delta.active = true;
        if (delta.first_event == 0) delta.first_event = seq;
        delta.last_event = seq;
        ApplyViewportToTarget(delta, a);
        record.final_states[h] = reshade::api::resource_usage::render_target;
    }
    if (dsv_res != 0)
    {
        if (TexInfo *d = FindTex(dsv_res)) d->ever_depth = true;
        ResourceDelta &delta = record.deltas[dsv_res];
        delta.active = true;
        delta.depth = true;
        if (delta.first_event == 0) delta.first_event = seq;
        delta.last_event = seq;
        record.final_states[dsv_res] = reshade::api::resource_usage::depth_stencil;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static void ClearActiveAttachments(reshade::api::command_list *cmd_list)
{
    AcquireSRWLockExclusive(&g_lock);
    auto it = g_commands.find(cmd_list->get_native());
    if (it != g_commands.end())
    {
        CommandRecord &record = it->second;
        if (g_pending.stage == 1)
            for (uint32_t i = 0; i < record.active.rt_count; ++i)
                if (record.active.rt_res[i] == g_pending.snap.handle)
                {
                    ++g_pending.render_pass_ends;
                    break;
                }
        // The add-on render-pass descriptor does not expose Vulkan finalLayout.
        // Mark attachment state unknown until a later explicit barrier observes it.
        for (uint32_t i = 0; i < record.active.rt_count; ++i)
            record.final_states[record.active.rt_res[i]] = reshade::api::resource_usage::undefined;
        if (record.active.dsv_res != 0)
            record.final_states[record.active.dsv_res] = reshade::api::resource_usage::undefined;
        // end_render_pass is a before-event. No command callback can interleave
        // before the native end returns, so clearing this flag here accurately
        // describes every later barrier callback on this command list.
        record.in_render_pass = false;
        // Render-pass end unbinds attachments, but Vulkan dynamic viewport state
        // remains command-list state until changed or the list is reset.
        record.active.rt_count = 0;
        record.active.dsv_res = 0;
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnBindRenderTargets(reshade::api::command_list *cmd_list, uint32_t count,
                                const reshade::api::resource_view *rtvs, reshade::api::resource_view dsv)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0 || g_device == nullptr)
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
    if (g_enabled.load(std::memory_order_relaxed) == 0)
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
    AcquireSRWLockExclusive(&g_lock);
    CommandRecord &record = g_commands[cmd_list->get_native()];
    record.in_render_pass = true;
    if (g_pending.stage == 1)
        for (uint32_t i = 0; i < n; ++i)
            if (rt_res[i] == g_pending.snap.handle)
            {
                ++g_pending.render_pass_begins;
                break;
            }
    ReleaseSRWLockExclusive(&g_lock);
    return false;
}

static bool OnEndRenderPass(reshade::api::command_list *cmd_list)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0) return false;
    ClearActiveAttachments(cmd_list);
    return false;
}

static void RecordDraw(reshade::api::command_list *cmd_list, uint32_t calls, uint32_t weight)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0)
        return;
    AcquireSRWLockExclusive(&g_lock);
    const uint64_t seq = g_record_event_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    auto it = g_commands.find(cmd_list->get_native());
    if (it != g_commands.end())
    {
        CommandRecord &record = it->second;
        for (uint32_t i = 0; i < record.active.rt_count; ++i)
        {
            ResourceDelta &delta = record.deltas[record.active.rt_res[i]];
            delta.draw_calls += calls;
            delta.draw_weight += weight;
            delta.active = true;
            if (delta.first_event == 0) delta.first_event = seq;
            delta.last_event = seq;
            if (record.active.dsv_res != 0)
            {
                delta.had_depth = true;
                delta.depth_resource = record.active.dsv_res;
            }
        }
    }
    ReleaseSRWLockExclusive(&g_lock);
}

static uint32_t SaturatingDrawWeight(uint32_t elements, uint32_t instances)
{
    const uint64_t work = uint64_t(elements) * std::max(1u, instances);
    return uint32_t(std::min<uint64_t>(UINT32_MAX, 1 + work / 1000));
}

static bool OnDraw(reshade::api::command_list *cl, uint32_t vertices, uint32_t instances, uint32_t, uint32_t)
{
    RecordDraw(cl, 1, SaturatingDrawWeight(vertices, instances));
    return false;
}

static bool OnDrawIndexed(reshade::api::command_list *cl, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t)
{
    RecordDraw(cl, 1, SaturatingDrawWeight(indices, instances));
    return false;
}

// Only INDIRECT DRAWS count as geometry here. Compute dispatches produce their
// results through different paths and must not inflate the "geometry" signal.
static bool OnDrawOrDispatchIndirect(reshade::api::command_list *cl, reshade::api::indirect_command type,
                                     reshade::api::resource, uint64_t, uint32_t count, uint32_t)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0)
        return false;
    if (type != reshade::api::indirect_command::draw &&
        type != reshade::api::indirect_command::draw_indexed)
        return false; // dispatch / dispatch_mesh / dispatch_rays: ignored
    RecordDraw(cl, count, uint32_t(std::min<uint64_t>(UINT32_MAX, uint64_t(count) * 10)));
    return false;
}

static void OnViewport(reshade::api::command_list *cmd_list, uint32_t first, uint32_t count, const reshade::api::viewport *vps)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0 || count == 0 || vps == nullptr)
        return;
    AcquireSRWLockExclusive(&g_lock);
    CommandRecord &record = g_commands[cmd_list->get_native()];
    ActiveRt &a = record.active;
    if (first == 0)
        for (uint32_t i = 0; i < ActiveRt::kMaxViewports; ++i) a.vp_valid[i] = false;
    for (uint32_t i = 0; i < count && first + i < ActiveRt::kMaxViewports; ++i)
    {
        const uint32_t index = first + i;
        a.vp_valid[index] = vps[i].width > 0.0f && vps[i].height > 0.0f;
        a.vp_w[index] = a.vp_valid[index] ? uint32_t(vps[i].width) : 0;
        a.vp_h[index] = a.vp_valid[index] ? uint32_t(vps[i].height) : 0;
    }
    RecomputeViewportState(a);
    for (uint32_t i = 0; i < a.rt_count; ++i)
        ApplyViewportToTarget(record.deltas[a.rt_res[i]], a);
    ReleaseSRWLockExclusive(&g_lock);
}

static void RecordCopy(reshade::api::command_list *cmd_list, uint64_t src, uint64_t dst)
{
    AcquireSRWLockExclusive(&g_lock);
    if (g_private_resources.count(src) != 0 || g_private_resources.count(dst) != 0)
    {
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }
    CommandRecord &record = g_commands[cmd_list->get_native()];
    const uint64_t seq = g_record_event_seq.fetch_add(1, std::memory_order_relaxed) + 1;
    if (record.edges.size() < 2048) record.edges.push_back({ src, dst });
    if (TexInfo *s = FindTex(src)) s->ever_copy_src = true;
    ResourceDelta &source = record.deltas[src];
    source.active = true;
    if (source.first_event == 0) source.first_event = seq;
    source.last_event = seq;
    record.final_states[src] = reshade::api::resource_usage::copy_source;
    if (TexInfo *d = FindTex(dst)) d->ever_copy_dst = true;
    ResourceDelta &dest = record.deltas[dst];
    dest.active = true;
    if (dest.first_event == 0) dest.first_event = seq;
    dest.last_event = seq;
    record.final_states[dst] = reshade::api::resource_usage::copy_dest;
    ReleaseSRWLockExclusive(&g_lock);
}

static bool OnCopyResource(reshade::api::command_list *cl, reshade::api::resource src, reshade::api::resource dst)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0) return false;
    RecordCopy(cl, src.handle, dst.handle);
    return false;
}

static bool OnCopyTextureRegion(reshade::api::command_list *cl, reshade::api::resource src, uint32_t,
                                const reshade::api::subresource_box *, reshade::api::resource dst, uint32_t,
                                const reshade::api::subresource_box *, reshade::api::filter_mode)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0) return false;
    RecordCopy(cl, src.handle, dst.handle);
    return false;
}

static bool OnResolveTextureRegion(reshade::api::command_list *cl, reshade::api::resource src, uint32_t,
                                   const reshade::api::subresource_box *, reshade::api::resource dst, uint32_t,
                                   uint32_t, uint32_t, uint32_t, reshade::api::format)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0) return false;
    RecordCopy(cl, src.handle, dst.handle);
    return false;
}

struct FmtInfo { unsigned f; const char *name; uint32_t bpp; bool bgra; };

static bool LookupCaptureFormat(unsigned raw, FmtInfo &out)
{
    static const FmtInfo known[] = {
        { unsigned(reshade::api::format::r8g8b8a8_unorm),          "R8G8B8A8_UNORM", 32, false },
        { unsigned(reshade::api::format::b8g8r8a8_unorm),          "B8G8R8A8_UNORM", 32, true },
        { unsigned(reshade::api::format::r8g8b8a8_unorm_srgb),     "R8G8B8A8_SRGB",  32, false },
        { unsigned(reshade::api::format::b8g8r8a8_unorm_srgb),     "B8G8R8A8_SRGB",  32, true },
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

// ---------------------------------------------------------------------------
// Barrier tracking: real resource state + the staged-capture trigger
// ---------------------------------------------------------------------------

static void OnBarrier(reshade::api::command_list *cmd_list, uint32_t count,
                      const reshade::api::resource *resources, const reshade::api::resource_usage *old_states,
                      const reshade::api::resource_usage *new_states)
{
    if (g_enabled.load(std::memory_order_relaxed) == 0 || count == 0 || resources == nullptr ||
        old_states == nullptr || new_states == nullptr || g_device == nullptr)
        return;

    bool trigger = false;
    uint64_t trigger_res = 0;
    uint32_t trigger_point = 0, trigger_occurrence = 0;
    reshade::api::resource_usage trigger_state = reshade::api::resource_usage::undefined;

    AcquireSRWLockExclusive(&g_lock);
    CommandRecord &record = g_commands[cmd_list->get_native()];
    for (uint32_t i = 0; i < count; ++i)
    {
        const uint64_t h = resources[i].handle;
        if (h == 0 || g_private_resources.count(h) != 0) continue;
        record.final_states[h] = new_states[i];

        if (g_pending.stage != 1 || h != g_pending.snap.handle)
            continue;

        ++g_pending.all_barriers;
        g_pending.last_old_state = old_states[i];
        g_pending.last_new_state = new_states[i];

        const bool old_rt =
            (old_states[i] & reshade::api::resource_usage::render_target) != reshade::api::resource_usage::undefined;
        const bool new_rt =
            (new_states[i] & reshade::api::resource_usage::render_target) != reshade::api::resource_usage::undefined;
        const bool rt_exit = old_rt && !new_rt;
        // An undefined old state may legally discard prior contents, so it is
        // diagnostic-only and never a pre-reuse capture point.
        const bool pre_rt_reuse = !old_rt && new_rt && old_states[i] != reshade::api::resource_usage::undefined;
        if (rt_exit) ++g_pending.observed_exits;
        if (pre_rt_reuse) ++g_pending.observed_reuses;

        const bool selected_point = g_pending.capture_point == 0 ? rt_exit : pre_rt_reuse;
        if (trigger || !selected_point)
            continue;

        const TexInfo *tracked = FindTex(h);
        if (record.in_render_pass)
        {
            ++g_pending.skipped_inside_render_pass;
            continue;
        }
        if (tracked == nullptr || tracked->total_draw_calls == 0)
        {
            ++g_pending.skipped_without_submitted_draw;
            continue;
        }

        ++g_pending.observed_points;
        if (g_pending.observed_points == g_pending.desired_occurrence)
        {
            // Claim under the lock so concurrent command-buffer recording
            // cannot create two snapshots for one request.
            g_pending.stage = 2;
            g_pending.source_cmd = cmd_list->get_native();
            g_pending.submission_cmd = cmd_list->get_native();
            g_pending.trigger_old_state = old_states[i];
            g_pending.trigger_new_state = new_states[i];
            g_pending.trigger_frame = g_frame.load(std::memory_order_relaxed);
            trigger = true;
            trigger_res = h;
            trigger_state = new_states[i]; // barrier callback is AFTER the application barrier
            trigger_point = g_pending.capture_point;
            trigger_occurrence = g_pending.desired_occurrence;
        }
    }
    ReleaseSRWLockExclusive(&g_lock);

    if (!trigger)
        return;

    const reshade::api::resource res = { trigger_res };
    const reshade::api::resource_desc desc = g_device->get_resource_desc(res);
    FmtInfo fi = {};
    if (desc.type != reshade::api::resource_type::texture_2d || desc.texture.samples != 1 ||
        (desc.usage & reshade::api::resource_usage::copy_source) == reshade::api::resource_usage::undefined ||
        !LookupCaptureFormat(unsigned(desc.texture.format), fi))
    {
        Log("[probe] staged capture aborted: texture %s is unsupported or lacks copy_source usage",
            FmtName(unsigned(desc.texture.format)));
        AcquireSRWLockExclusive(&g_lock);
        ClearPendingWithResultLocked(CaptureResultStatus::failed, "unsupported format or missing copy_source usage");
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }

    const reshade::api::resource_usage snap_usage =
        reshade::api::resource_usage::copy_dest | reshade::api::resource_usage::copy_source;
    reshade::api::resource_desc snap_desc(desc.texture.width, desc.texture.height, 1, 1,
        desc.texture.format, 1, reshade::api::memory_heap::default_, snap_usage,
        reshade::api::resource_flags::none);
    reshade::api::resource snap = { 0 };
    if (!g_device->create_resource(snap_desc, nullptr, reshade::api::resource_usage::copy_dest, &snap) || snap.handle == 0)
    {
        Log("[probe] staged capture aborted: snapshot texture creation failed");
        AcquireSRWLockExclusive(&g_lock);
        ClearPendingWithResultLocked(CaptureResultStatus::failed, "snapshot texture creation failed");
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }

    const uint64_t pitch = uint64_t(desc.texture.width) * (fi.bpp / 8);
    const uint64_t buffer_size = pitch * desc.texture.height;
    reshade::api::resource buffer = { 0 };
    reshade::api::resource_desc buffer_desc(buffer_size, reshade::api::memory_heap::readback,
        reshade::api::resource_usage::copy_dest);
    if (!g_device->create_resource(buffer_desc, nullptr, reshade::api::resource_usage::copy_dest, &buffer))
    {
        g_device->destroy_resource(snap);
        Log("[probe] staged capture aborted: readback buffer creation failed");
        AcquireSRWLockExclusive(&g_lock);
        ClearPendingWithResultLocked(CaptureResultStatus::failed, "readback buffer creation failed");
        ReleaseSRWLockExclusive(&g_lock);
        return;
    }

    AcquireSRWLockExclusive(&g_lock);
    g_private_resources.insert(snap.handle);
    g_private_resources.insert(buffer.handle);
    g_tex.erase(snap.handle);
    g_tex.erase(buffer.handle);
    ReleaseSRWLockExclusive(&g_lock);

    // ReShade's barrier event is after the application's barrier, so the source
    // is in trigger_state here. Record the complete GPU capture, including the
    // texture-to-buffer copy, in this same command list. No later command list
    // is allowed to race the snapshot producer.
    const reshade::api::resource_usage copy_source = reshade::api::resource_usage::copy_source;
    const reshade::api::resource_usage copy_dest = reshade::api::resource_usage::copy_dest;
    cmd_list->barrier(1, &res, &trigger_state, &copy_source);
    cmd_list->copy_texture_region(res, 0, nullptr, snap, 0, nullptr);
    cmd_list->barrier(1, &res, &copy_source, &trigger_state);
    cmd_list->barrier(1, &snap, &copy_dest, &copy_source);
    cmd_list->copy_texture_to_buffer(snap, 0, nullptr, buffer, 0, 0, 0);

    AcquireSRWLockExclusive(&g_lock);
    g_snap_res = snap;
    g_snap_valid = true;
    g_capture_buffer = buffer;
    g_capture_buffer_size = buffer_size;
    g_pending.stage = 3;
    ReleaseSRWLockExclusive(&g_lock);
    Log("[probe] staged capture recorded at %s occurrence %u (frame %llu)",
        CapturePointName(trigger_point), trigger_occurrence,
        (unsigned long long)g_frame.load(std::memory_order_relaxed));
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

static void MergeDelta(ResourceDelta &dest, const ResourceDelta &source)
{
    dest.draw_calls += source.draw_calls;
    dest.draw_weight += source.draw_weight;
    dest.rt_binds += source.rt_binds;
    dest.active |= source.active;
    dest.depth |= source.depth;
    dest.had_depth |= source.had_depth;
    if (source.depth_resource != 0) dest.depth_resource = source.depth_resource;
    if (source.vp_count != 0)
    {
        dest.vp_w = source.vp_w; dest.vp_h = source.vp_h;
        dest.vp_count = std::max(dest.vp_count, source.vp_count);
        dest.vp_max_w = std::max(dest.vp_max_w, source.vp_max_w);
        dest.vp_max_h = std::max(dest.vp_max_h, source.vp_max_h);
    }
    if (dest.first_event == 0 || (source.first_event != 0 && source.first_event < dest.first_event))
        dest.first_event = source.first_event;
    dest.last_event = std::max(dest.last_event, source.last_event);
}

static void MergeCommandRecord(CommandRecord &dest, const CommandRecord &source)
{
    for (const auto &entry : source.deltas) MergeDelta(dest.deltas[entry.first], entry.second);
    for (const auto &entry : source.final_states) dest.final_states[entry.first] = entry.second;
    for (const auto &edge : source.edges)
        if (dest.edges.size() < 2048) dest.edges.push_back(edge);
}

static void CompleteCaptureAtPresent(reshade::api::command_queue *completion_queue);

static void OnExecuteSecondaryCommandList(reshade::api::command_list *primary, reshade::api::command_list *secondary)
{
    AcquireSRWLockExclusive(&g_lock);
    auto secondary_it = g_commands.find(secondary->get_native());
    if (secondary_it != g_commands.end())
        MergeCommandRecord(g_commands[primary->get_native()], secondary_it->second);
    if (g_pending.stage == 3 && g_pending.submission_cmd == secondary->get_native())
        g_pending.submission_cmd = primary->get_native();
    ReleaseSRWLockExclusive(&g_lock);
}

static void OnExecuteCommandList(reshade::api::command_queue *queue, reshade::api::command_list *cmd_list)
{
    bool complete_previous_submission = false;
    AcquireSRWLockExclusive(&g_lock);
    complete_previous_submission = g_pending.stage == 4 && g_pending.source_queue == queue;
    const auto it = g_commands.find(cmd_list->get_native());
    if (g_enabled.load(std::memory_order_relaxed) != 0 && it != g_commands.end())
    {
        for (const auto &entry : it->second.deltas)
        {
            TexInfo *t = FindTex(entry.first);
            if (t == nullptr || !t->alive) continue;
            const ResourceDelta &delta = entry.second;
            t->fr_draw_calls += delta.draw_calls;
            t->fr_draw_weight += delta.draw_weight;
            t->fr_rt_binds += delta.rt_binds;
            t->fr_active |= delta.active;
            t->fr_depth |= delta.depth;
            t->fr_had_depth |= delta.had_depth;
            if (delta.depth_resource != 0) t->fr_depth_resource = delta.depth_resource;
            if (delta.vp_count != 0)
            {
                t->fr_vp_w = delta.vp_w; t->fr_vp_h = delta.vp_h;
                t->fr_vp_count = delta.vp_count;
                t->fr_vp_max_w = delta.vp_max_w; t->fr_vp_max_h = delta.vp_max_h;
            }
            if (t->fr_first_event == 0 || (delta.first_event != 0 && delta.first_event < t->fr_first_event))
                t->fr_first_event = delta.first_event;
            t->fr_last_event = std::max(t->fr_last_event, delta.last_event);
            t->total_draw_calls += delta.draw_calls;
            t->total_draw_weight += delta.draw_weight;
        }
        for (const auto &entry : it->second.final_states)
            if (TexInfo *t = FindTex(entry.first)) t->submitted_usage_hint = entry.second;
        for (const auto &edge : it->second.edges)
            if (g_edges.size() < 2048) g_edges.push_back(edge);
    }
    if (g_pending.stage == 3 && g_pending.submission_cmd == cmd_list->get_native())
    {
        g_pending.source_queue = queue;
        g_pending.stage = 4;
    }
    ReleaseSRWLockExclusive(&g_lock);
    // On a later submission callback for this queue, the capture submission is
    // already enqueued. Complete before the new native submission occurs.
    if (complete_previous_submission) CompleteCaptureAtPresent(queue);
}

static void OnResetCommandList(reshade::api::command_list *cmd_list)
{
    std::vector<reshade::api::resource> release;
    const uint64_t command = cmd_list->get_native();
    AcquireSRWLockExclusive(&g_lock);
    g_commands.erase(command);
    auto retired = g_retired_resources.find(command);
    if (retired != g_retired_resources.end())
    {
        release = std::move(retired->second);
        g_retired_resources.erase(retired);
    }
    if (g_pending.stage == 3 && g_pending.source_cmd == command)
    {
        if (g_snap_res.handle != 0) release.push_back(g_snap_res);
        if (g_capture_buffer.handle != 0) release.push_back(g_capture_buffer);
        g_snap_res = { 0 }; g_capture_buffer = { 0 }; g_capture_buffer_size = 0;
        g_snap_valid = false;
        ClearPendingWithResultLocked(CaptureResultStatus::failed, "source command list reset before submission");
    }
    else if (g_pending.stage == 3 && g_pending.submission_cmd == command)
    {
        // A primary recording containing the secondary was reset. The capture
        // still belongs to the secondary and may be linked into another primary.
        g_pending.submission_cmd = g_pending.source_cmd;
    }
    ReleaseSRWLockExclusive(&g_lock);
    // The old recording is no longer executable after reset/destroy, so only
    // now is it legal to release resources referenced by that recording.
    if (g_device != nullptr)
        for (reshade::api::resource resource : release)
            if (resource.handle != 0) g_device->destroy_resource(resource);
}

static void OnDestroyCommandList(reshade::api::command_list *cmd_list)
{
    OnResetCommandList(cmd_list);
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
            s.copy_source_capable = t.copy_source_capable;
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
    g_record_event_seq.store(0, std::memory_order_relaxed);
    ReleaseSRWLockExclusive(&g_lock);
    g_frame.fetch_add(1, std::memory_order_relaxed);
}

static void OnPresent(reshade::api::command_queue *queue, reshade::api::swapchain *swapchain, const reshade::api::rect *,
                      const reshade::api::rect *, uint32_t, const reshade::api::rect *)
{
    CompleteCaptureAtPresent(queue);
    if (g_enabled.load(std::memory_order_relaxed) == 0) return;
    SyncSwapchainBackbuffers(swapchain);

    bool capture_timed_out = false;
    PendingCapture timeout_capture;
    AcquireSRWLockExclusive(&g_lock);
    if (g_pending.stage == 1 &&
        g_frame.load(std::memory_order_relaxed) - g_pending.armed_frame > 180)
    {
        timeout_capture = g_pending;
        ClearPendingWithResultLocked(CaptureResultStatus::timeout, "requested safe capture point did not occur");
        capture_timed_out = true;
    }
    ReleaseSRWLockExclusive(&g_lock);
    if (capture_timed_out)
        Log("[probe] capture timeout: resource=0x%016llX mode=%s occurrence=%u barriers=%u explicit_rt_exits=%u pre_rt_reuses=%u safe_points=%u render_pass_begin/end=%u/%u skipped_inside=%u skipped_no_draw=%u last_transition=0x%X->0x%X",
            (unsigned long long)timeout_capture.snap.handle, CapturePointName(timeout_capture.capture_point),
            timeout_capture.desired_occurrence, timeout_capture.all_barriers, timeout_capture.observed_exits,
            timeout_capture.observed_reuses, timeout_capture.observed_points, timeout_capture.render_pass_begins,
            timeout_capture.render_pass_ends, timeout_capture.skipped_inside_render_pass,
            timeout_capture.skipped_without_submitted_draw, unsigned(timeout_capture.last_old_state),
            unsigned(timeout_capture.last_new_state));
    FinalizeFrame();
}

// ---------------------------------------------------------------------------
// One-shot capture (staged model, spec sections 7, 8 and 13)
// ---------------------------------------------------------------------------

// Write the BMP + metadata from a mapped readback buffer (tightly packed rows:
// the copy used row_length = 0 / slice_height = 0, so pitch is w * bytes/px).
static bool WriteCaptureOutput(const char *pathbmp, const char *pathtxt, const FmtInfo &fi,
                               uint32_t w, uint32_t h, uint64_t pitch, const void *mapped,
                               const PendingCapture &capture, const TexInfo *live,
                               const std::vector<std::pair<uint64_t, uint64_t>> &edges)
{
    const Snapshot &sel = capture.snap;
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
                        const bool negative = (hb & 0x8000u) != 0;
                        const uint32_t exp = (hb >> 10) & 0x1F, man = hb & 0x3FF;
                        float v = (exp == 0) ? man * (1.0f / 1024.0f) * (1.0f / 16384.0f)
                                  : (exp == 31) ? 1.0f : ldexpf(float(man | 0x400), int(exp) - 25);
                        if (negative) v = -v;
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
        Log("[probe] captured %s (%ux%u %s, %s occurrence %u)", pathbmp, w, h, fi.name,
            CapturePointName(capture.capture_point), capture.desired_occurrence);
    }
    else
    {
        Log("[probe] capture failed: cannot open %s", pathbmp);
        return false;
    }

    // Metadata: previous-frame ranking stats ("selected_snapshot") are kept
    // strictly separate from the resource's live state at capture time.
    if (fopen_s(&fp, pathtxt, "w") == 0 && fp != nullptr)
    {
        fprintf(fp, "[resource]\nhandle=0x%016llX\nformat=%s (%u)\nsize=%ux%u\nlayers=%u samples=%u\n"
                    "capture_stage=barrier_after\ncapture_point=%s\ncapture_point_occurrence=%u\n"
                    "captured_subresource=0\ncopy_source_capable=%d\ncandidate_rank=%u\n"
                    "source_command_list=0x%016llX\nsubmission_command_list=0x%016llX\n"
                    "trigger_frame=%llu\ntrigger_old_usage=0x%X\ntrigger_new_usage=0x%X\n"
                    "barriers_seen_before_trigger=%u\nexplicit_rt_exits_seen=%u\npre_rt_reuses_seen=%u\n"
                    "safe_points_seen=%u\nrender_pass_begins_seen=%u\nrender_pass_ends_seen=%u\n"
                    "skipped_inside_render_pass=%u\nskipped_without_submitted_draw=%u\n",
                (unsigned long long)sel.handle, fi.name, fi.f, w, h, sel.layers, sel.samples,
                CapturePointName(capture.capture_point), capture.desired_occurrence,
                int(sel.copy_source_capable), capture.candidate_rank,
                (unsigned long long)capture.source_cmd, (unsigned long long)capture.submission_cmd,
                (unsigned long long)capture.trigger_frame, unsigned(capture.trigger_old_state),
                unsigned(capture.trigger_new_state), capture.all_barriers, capture.observed_exits,
                capture.observed_reuses, capture.observed_points, capture.render_pass_begins,
                capture.render_pass_ends, capture.skipped_inside_render_pass,
                capture.skipped_without_submitted_draw);
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
                        "depth_resource=0x%016llX\nsubmitted_usage_hint=0x%X\nframes_seen=%u\n"
                        "total_draw_calls=%llu\ntotal_draw_weight=%llu\n",
                    (unsigned long long)g_frame.load(std::memory_order_relaxed),
                    live->fr_draw_calls, live->fr_draw_weight, live->fr_rt_binds, int(live->fr_had_depth),
                    (unsigned long long)live->fr_depth_resource, unsigned(live->submitted_usage_hint),
                    live->frames_seen, (unsigned long long)live->total_draw_calls,
                    (unsigned long long)live->total_draw_weight);
        }
        fprintf(fp, "\n[copy_graph]\ncopy_edges_this_frame=%zu\n", edges.size());
        for (const auto &e : edges)
            fprintf(fp, "copy 0x%016llX -> 0x%016llX\n", (unsigned long long)e.first, (unsigned long long)e.second);
        fprintf(fp, "note=copy graph sees copy/resolve edges only, not SRV sampling; feeds_swapchain is informational\n");
        if (fi.bpp == 64)
            fprintf(fp, "preview_note=half-float RGB is clamped to [0,1] for the 24-bit BMP; alpha is omitted\n");
        fclose(fp);
        return true;
    }
    Log("[probe] capture failed: cannot open %s", pathtxt);
    return false;
}

static void CompleteCaptureAtPresent(reshade::api::command_queue *completion_queue)
{
    if (g_device == nullptr || completion_queue == nullptr) return;

    reshade::api::command_queue *source_queue = nullptr;
    reshade::api::resource snap = { 0 }, buffer = { 0 };
    uint64_t buffer_size = 0;
    PendingCapture capture;
    {
        AcquireSRWLockExclusive(&g_lock);
        if (g_pending.stage != 4 || g_pending.source_queue == nullptr ||
            g_pending.source_queue != completion_queue)
        {
            ReleaseSRWLockExclusive(&g_lock);
            return;
        }
        source_queue = g_pending.source_queue;
        snap = g_snap_res;
        buffer = g_capture_buffer;
        buffer_size = g_capture_buffer_size;
        g_pending.stage = 5; // completion is owned by this present callback
        capture = g_pending;
        ReleaseSRWLockExclusive(&g_lock);
    }

    bool completed = false;
    if (g_capture_fence.handle == 0 &&
        !g_device->create_fence(0, reshade::api::fence_flags::none, &g_capture_fence))
    {
        Log("[probe] capture failed: fence creation failed; waiting for source queue idle");
        source_queue->wait_idle();
        completed = true;
    }
    else
    {
        const uint64_t value = ++g_capture_fence_value;
        if (source_queue->signal(g_capture_fence, value) && g_device->wait(g_capture_fence, value))
            completed = true;
        else
        {
            Log("[probe] capture fence failed; waiting for source queue idle before cleanup");
            source_queue->wait_idle();
            completed = true;
        }
    }

    bool output_written = false;
    if (completed && snap.handle != 0 && buffer.handle != 0)
    {
        const reshade::api::resource_desc desc = g_device->get_resource_desc(snap);
        FmtInfo fi = {};
        const uint64_t pitch = LookupCaptureFormat(unsigned(desc.texture.format), fi)
            ? uint64_t(desc.texture.width) * (fi.bpp / 8) : 0;
        void *mapped = nullptr;
        if (pitch != 0 && buffer_size >= pitch * desc.texture.height &&
            g_device->map_buffer_region(buffer, 0, buffer_size, reshade::api::map_access::read_only, &mapped) && mapped != nullptr)
        {
            TexInfo live_copy = {};
            bool have_live = false;
            std::vector<std::pair<uint64_t, uint64_t>> edges;
            {
                AcquireSRWLockShared(&g_lock);
                if (const TexInfo *t = FindTex(capture.snap.handle)) { live_copy = *t; have_live = true; }
                edges = g_edges;
                ReleaseSRWLockShared(&g_lock);
            }

            char dir[] = "dlss5-probe";
            CreateDirectoryA(dir, nullptr);
            char pathbmp[MAX_PATH], pathtxt[MAX_PATH];
            snprintf(pathbmp, sizeof(pathbmp), "%s/frame_%06llu_candidate_%02u_%s_%02u_%ux%u.bmp", dir,
                     (unsigned long long)g_frame.load(std::memory_order_relaxed), capture.candidate_rank,
                     CapturePointName(capture.capture_point), capture.desired_occurrence,
                     desc.texture.width, desc.texture.height);
            snprintf(pathtxt, sizeof(pathtxt), "%s/frame_%06llu_candidate_%02u_%s_%02u.txt", dir,
                     (unsigned long long)g_frame.load(std::memory_order_relaxed), capture.candidate_rank,
                     CapturePointName(capture.capture_point), capture.desired_occurrence);
            output_written = WriteCaptureOutput(pathbmp, pathtxt, fi, desc.texture.width, desc.texture.height,
                               pitch, mapped, capture, have_live ? &live_copy : nullptr, edges);
            g_device->unmap_buffer_region(buffer);
        }
        else
            Log("[probe] capture failed: readback map or format validation");
    }

    AcquireSRWLockExclusive(&g_lock);
    // A Vulkan command buffer remains executable and may be submitted again.
    // Keep its private resources alive until reset/destroy invalidates the old
    // recording, even though the first execution has completed.
    if (snap.handle != 0) g_retired_resources[capture.source_cmd].push_back(snap);
    if (buffer.handle != 0) g_retired_resources[capture.source_cmd].push_back(buffer);
    g_snap_res = { 0 }; g_capture_buffer = { 0 }; g_capture_buffer_size = 0;
    g_snap_valid = false;
    StoreLastCaptureResultLocked(output_written ? CaptureResultStatus::success : CaptureResultStatus::failed,
                                 output_written ? "BMP and metadata written" : "readback or output failed");
    g_pending = {};
    ReleaseSRWLockExclusive(&g_lock);
}

// ---------------------------------------------------------------------------
// Overlay panel (spec sections 9-10)
// ---------------------------------------------------------------------------

static void DrawProbeOverlay(reshade::api::effect_runtime * /*runtime*/)
{
    bool en = g_enabled.load(std::memory_order_relaxed) != 0;
    if (ImGui::Checkbox("Probe enabled (DLSS feed is inert while on)", &en))
    {
        AcquireSRWLockExclusive(&g_lock);
        if (!en && g_pending.stage == 1)
            ClearPendingWithResultLocked(CaptureResultStatus::cancelled, "probe disabled while capture was armed");
        for (auto &entry : g_tex)
        {
            TexInfo &t = entry.second;
            t.fr_draw_calls = t.fr_draw_weight = t.fr_rt_binds = 0;
            t.fr_depth = t.fr_feeds_swapchain = t.fr_copy_to = t.fr_had_depth = false;
            t.fr_depth_resource = 0;
            t.fr_vp_w = t.fr_vp_h = t.fr_vp_count = t.fr_vp_max_w = t.fr_vp_max_h = 0;
            t.fr_first_event = t.fr_last_event = 0;
            t.fr_active = false;
        }
        g_edges.clear(); g_top.clear();
        ReleaseSRWLockExclusive(&g_lock);
        g_enabled.store(en ? 1 : 0, std::memory_order_relaxed);
        if (g_save_cfg != nullptr) g_save_cfg();
    }
    ImGui::Text("Frame: %llu   Record event seq: %llu",
                (unsigned long long)g_frame.load(std::memory_order_relaxed),
                (unsigned long long)g_record_event_seq.load(std::memory_order_relaxed));
    ImGui::TextDisabled("(statistics are committed when command lists are submitted)");
    ImGui::Separator();

    int candidate = std::clamp(g_candidate.load(std::memory_order_relaxed), 1, 10);
    if (ImGui::Button("<< Prev")) candidate = std::max(1, candidate - 1);
    ImGui::SameLine();
    if (ImGui::Button("Next >>")) candidate = std::min(10, candidate + 1);
    ImGui::SameLine();
    ImGui::Text("Selected: #%d", candidate);
    const int old_candidate = g_candidate.exchange(candidate, std::memory_order_relaxed);
    if (candidate != old_candidate && g_save_cfg != nullptr) g_save_cfg();

    int capture_point = g_capture_point.load(std::memory_order_relaxed) == 0 ? 0 : 1;
    const char *capture_points[] = { "Explicit RT exit", "Pre-RT reuse" };
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::Combo("Capture point", &capture_point, capture_points, 2))
    {
        g_capture_point.store(capture_point, std::memory_order_relaxed);
        if (g_save_cfg != nullptr) g_save_cfg();
    }

    int occurrence = std::max(1, g_capture_occurrence.load(std::memory_order_relaxed));
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputInt("Capture point occurrence", &occurrence))
    {
        g_capture_occurrence.store(std::max(1, occurrence), std::memory_order_relaxed);
        if (g_save_cfg != nullptr) g_save_cfg();
    }

    if (ImGui::Button("Capture selected candidate"))
    {
        int status = 0;
        uint64_t armed_handle = 0;
        AcquireSRWLockExclusive(&g_lock);
        if (g_pending.stage != 0)
            status = 2;
        else if (candidate >= 1 && candidate <= int(g_top.size()))
        {
            const Snapshot &s = g_top[candidate - 1];
            const TexInfo *t = FindTex(s.handle);
            if (t != nullptr && t->alive)
            {
                g_pending.snap = s;
                g_pending.candidate_rank = uint32_t(candidate);
                g_pending.armed_frame = g_frame.load(std::memory_order_relaxed);
                g_pending.capture_point = uint32_t(capture_point);
                g_pending.desired_occurrence = uint32_t(std::max(1, occurrence));
                g_pending.stage = 1;
                armed_handle = s.handle;
                status = 1;
            }
            else status = 3;
        }
        else status = 4;
        ReleaseSRWLockExclusive(&g_lock);
        if (status == 1) Log("[probe] capture armed: resource=0x%016llX mode=%s occurrence=%d",
                             (unsigned long long)armed_handle, CapturePointName(uint32_t(capture_point)),
                             std::max(1, occurrence));
        else if (status == 2) Log("[probe] capture already pending");
        else if (status == 3) Log("[probe] capture rejected: candidate resource is no longer alive");
        else Log("[probe] capture: no candidate #%d in the submitted-frame ranking", candidate);
    }

    PendingCapture pending_copy;
    LastCaptureResult last_copy;
    std::vector<Snapshot> top;
    AcquireSRWLockShared(&g_lock);
    pending_copy = g_pending;
    last_copy = g_last_capture;
    top = g_top;
    ReleaseSRWLockShared(&g_lock);
    if (pending_copy.stage == 1)
        ImGui::Text("Capture armed: %s #%u (barriers=%u, safe=%u)", CapturePointName(pending_copy.capture_point),
                    pending_copy.desired_occurrence, pending_copy.all_barriers, pending_copy.observed_points);
    else if (pending_copy.stage == 2) ImGui::Text("Capture resources are being recorded");
    else if (pending_copy.stage == 3) ImGui::Text("Capture recorded; waiting for command-list submission");
    else if (pending_copy.stage == 4 || pending_copy.stage == 5) ImGui::Text("Capture submitted; waiting for GPU completion");

    if (last_copy.status != CaptureResultStatus::none)
    {
        const PendingCapture &last = last_copy.capture;
        ImGui::Text("Last capture: %s - %s", CaptureResultName(last_copy.status), last_copy.reason);
        ImGui::TextDisabled("%s #%u | barriers=%u exits=%u reuses=%u safe=%u | render pass=%u/%u",
                            CapturePointName(last.capture_point), last.desired_occurrence, last.all_barriers,
                            last.observed_exits, last.observed_reuses, last.observed_points,
                            last.render_pass_begins, last.render_pass_ends);
        ImGui::TextDisabled("skipped: in-pass=%u no-submitted-draw=%u | last usage 0x%X -> 0x%X",
                            last.skipped_inside_render_pass, last.skipped_without_submitted_draw,
                            unsigned(last.last_old_state), unsigned(last.last_new_state));
    }
    ImGui::Separator();

    if (top.empty()) { ImGui::TextDisabled("No submitted candidates in the latest frame"); return; }
    for (size_t i = 0; i < top.size(); ++i)
    {
        const Snapshot &s = top[i];
        char label[32]; snprintf(label, sizeof(label), "#%u", unsigned(i + 1));
        ImGui::BeginGroup();
        ImGui::Text("%s %s 0x%016llX", label, (int(i + 1) == candidate) ? ">" : " ", (unsigned long long)s.handle);
        ImGui::Text("  %ux%u %s layers=%u samples=%u viewport=%ux%u x%u", s.w, s.h, FmtName(s.fmt), s.layers, s.samples, s.vp_w, s.vp_h, s.vp_count);
        ImGui::Text("  draw_calls=%u draw_weight=%u binds=%u first=%llu last=%llu",
                    s.draw_calls, s.draw_weight, s.binds, (unsigned long long)s.first, (unsigned long long)s.last);
        if (s.had_depth && s.depth_handle != 0)
            ImGui::Text("  depth: 0x%016llX (%ux%u %s)", (unsigned long long)s.depth_handle, s.depth_w, s.depth_h, FmtName(s.depth_fmt));
        else
            ImGui::TextDisabled("  depth: none");
        ImGui::Text("  copy-source=%s copy-graph-feeds-swapchain=%s score=%u",
                    s.copy_source_capable ? "yes" : "no", s.feeds ? "yes" : "no", s.score);
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
    reshade::register_event<reshade::addon_event::destroy_command_list>(OnDestroyCommandList);
    reshade::register_event<reshade::addon_event::reset_command_list>(OnResetCommandList);
    reshade::register_event<reshade::addon_event::execute_command_list>(OnExecuteCommandList);
    reshade::register_event<reshade::addon_event::execute_secondary_command_list>(OnExecuteSecondaryCommandList);
    reshade::register_event<reshade::addon_event::create_resource>(OnCreateResource);
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
