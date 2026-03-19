const std = @import("std");

/// Zig build script (optional).
///
/// This repo uses runtime JIT compilation for CUDA when possible, so you can work
/// without Zig/nvcc installed. When you *do* have Zig + a CUDA toolkit, this script
/// can help orchestrate:
/// - building the Rust planner (PyO3) via `cargo`
/// - compiling CUDA sources via `nvcc` (placeholder; kernel integration still uses PyTorch JIT)
///
/// The goal is reproducible builds and a single entrypoint for future "prebuild kernels".
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const run_cargo = b.addSystemCommand(&[_][]const u8{
        "cargo",
        "build",
        "--release",
        "-p",
        "synapsefast-planner",
    });
    _ = run_cargo.addOutputDirectoryArg(std.fs.path.dirname(b.pathFromRoot("target")) orelse ".");

    // Note: CUDA prebuild is intentionally not fully wired yet.
    // The Python layer currently builds CUDA at runtime via `torch.utils.cpp_extension.load`.
    // This hook exists so we can extend to "offline kernel builds" later.
    const build_step = b.step("default", "Build Rust planner (and optionally kernels later).");
    build_step.dependOn(&run_cargo.step);

    _ = target;
    _ = optimize;
}

