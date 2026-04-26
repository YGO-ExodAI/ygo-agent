package("edopro-core")

    -- Switched from stock edo9300 upstream to our YGO-ExodAI/ygopro-core
    -- fork (B.3.a Chunk 2b, 2026-04-20). The EDOPro client uses the same
    -- fork for its ocgcore submodule (per CLAUDE.md); training and eval
    -- on one source of truth is the only way "same engine" is actually
    -- true.
    --
    -- Pinned to c69921f9 (2026-04-25): fix constant.lua + utility.lua load
    -- order before Pass 1 new_card loop (Lua nil bug on LoadState). Prior
    -- pin 3626e69b (2026-04-25): chunk 9b schema v2 + field.core scratch
    -- serialization. Prior pin b118d05 (B.8 cutover, 2026-04-21) predated
    -- chunk 9b and emitted schema_version=1 blobs → MSG_RETRY.
    set_homepage("https://github.com/YGO-ExodAI/ygopro-core")

    set_urls("https://github.com/YGO-ExodAI/ygopro-core.git")
    add_versions("exodai-b118d05", "b118d0585d208c0f65586f0ea7404daaee135da1")
    add_versions("exodai-3626e69b", "3626e69b52b64804062194a5df25c381e90491f6")
    add_versions("exodai-c69921f", "c69921f9269024a6ffc0555d866b2bc839e9c15e")

    -- ExodAI Phase P1 Primitive 1 (chunk 1): protobuf runtime for the
    -- serialize/ module. Listed as a package dep so xmake pulls it and the
    -- inline build template picks up the link automatically. Matches the
    -- ygoenv side's `protobuf-cpp 28.*` pin; both must stay in lockstep with
    -- EDOPro's vcpkg `protobuf`. See phase_p1_primitive_1_plan.md §1.4.
    add_deps("protobuf-cpp 28.x")

    -- Chunk-1 local-dev overlay. The serialize/ scaffolding lives in the
    -- ocgcore submodule working copy at the path below. xmake fetches the
    -- pinned SHA from GitHub as usual, then on_install overlays serialize/
    -- from this path onto the fetched source before building. Per
    -- phase_p1_primitive_1_plan.md §10's cross-repo publication gate, this
    -- overlay stays in place until chunk 7; at that point the fork SHA gets
    -- bumped (fork has serialize/ merged) and this overlay block is removed.
    set_policy("package.install_always", true)

    -- SPIKE 2026-04-20: no longer depend on the xmake `lua` package. edo9300's
    -- ocgcore is designed to link against Lua-compiled-as-C++ (for longjmp-
    -- through-C++-frames correctness — see interpreter.h L10), and xmake's
    -- `lua` ships C-linkage, so linking a C++ ocgcore against it fails with
    -- mangled-vs-unmangled symbol mismatches (e.g. `_Z12luaopen_mathP9lua_State`).
    -- We build Lua from the bundled `lua/src/` via a single .cpp amalgam unit,
    -- matching edo9300's own build (premake5: `compileas "C++"`).

    on_install("linux", function (package)
        -- Chunk-5c (2026-04-22): switched from per-file overlay to
        -- whole-tree mirror. Reason: xmake's git fetch was pulling
        -- master-HEAD instead of the pinned b118d05 (the package's
        -- shallow-clone codepath ignores the SHA). That made the
        -- per-file overlay unstable — overlaid duel.h vs fetched
        -- duel.cpp had drifted incompatibly between upstream master
        -- HEAD and our exodai branch.
        --
        -- Whole-tree mirror sidesteps the version-resolution issue
        -- entirely: the static lib is built from EXACTLY the local
        -- ocgcore working tree. This is functionally equivalent to
        -- what chunk-7's fork SHA bump achieves (eliminates the
        -- fetched/overlaid divergence), just achieved via mirror
        -- rather than fork-push.
        --
        -- TODO(chunk-7): delete this overlay block once the fork SHA
        -- bump lands. The package_url + add_versions become the
        -- single source of truth.
        local ocgcore_working = "/mnt/c/Users/Joe/Documents/edopro/edopro/ocgcore"
        -- Wipe the fetched source dir, then mirror the whole local tree.
        os.tryrm("*.cpp")
        os.tryrm("*.h")
        for _, dir in ipairs({"serialize", "RNG", "lua"}) do
            os.tryrm(dir)
        end
        for _, ext in ipairs({"cpp", "h", "hpp"}) do
            for _, f in ipairs(os.files(path.join(ocgcore_working, "*." .. ext))) do
                os.cp(f, path.basename(f) .. "." .. ext)
            end
        end
        for _, dir in ipairs({"serialize", "RNG", "lua"}) do
            local src = path.join(ocgcore_working, dir)
            if os.isdir(src) then
                os.cp(src, dir, {rootdir = ocgcore_working})
            end
        end

        -- Drop in a tiny amalgam translation unit so Lua's sources compile
        -- as C++ (extension = .cpp) in a single TU. `onelua.c` is Lua's
        -- official amalgamation; MAKE_LIB strips out the standalone
        -- interpreter/compiler `main()` paths we don't want.
        io.writefile("lua_amalgam.cpp", [[
            #define MAKE_LIB
            #include "lua/src/onelua.c"
        ]])

        io.writefile("xmake.lua", [[
            add_rules("mode.debug", "mode.release")
            add_requires("protobuf-cpp 28.x")
            target("edopro-core")
                set_kind("static")
                set_languages("c++17")
                add_files("*.cpp")
                -- ExodAI Phase P1 Primitive 1 (chunk 1): generate .pb.cc/.pb.h
                -- from .proto at build time using xmake's protobuf.cpp rule.
                -- Generated files are NOT committed (see serialize/.gitignore);
                -- each build system uses its own protoc matching its own runtime
                -- to avoid protobuf's strict gencode-vs-runtime version check.
                -- See phase_p1_primitive_1_plan.md §1.3.
                add_rules("protobuf.cpp")
                add_files("serialize/*.proto", {rules = "protobuf.cpp", proto_public = true, proto_rootdir = "serialize"})
                add_files("serialize/*.cpp")
                add_headerfiles("*.h")
                add_headerfiles("RNG/*.hpp")
                add_headerfiles("lua/src/*.h")
                add_headerfiles("serialize/*.h")
                add_includedirs("lua/src", "serialize", {public = true})
                add_packages("protobuf-cpp", {public = true})
        ]])

        -- [SPIKE 2026-04-20] stderr probes in check_lua_stack_unwinding +
        -- interpreter ctor lived here during diagnosis. They proved the
        -- Lua-as-C++ unwinding path works correctly; the real crash was
        -- elsewhere (query-buffer parser drift in the wrapper's
        -- get_cards_in_location, not in this package). Probes removed.
        -- Git history on this file carries the full diagnostic patch.

        -- Previously this block had four `check_and_insert` calls that
        -- wrapped lines 12-14 and 16-19 of interpreter.h in `extern "C"`.
        -- Upstream interpreter.h has shifted; those lines are now STL
        -- includes (<list>, <unordered_map>, <utility>, <vector>), and
        -- wrapping STL in C linkage pulls in `<memory>`'s class templates
        -- → "template with C linkage" compile error. Current interpreter.h
        -- does not include <lua.h> at all; no wrapping is needed.
        -- See src/docs/edopro_envpool_spike.md §2 for full diagnosis.
        local configs = {}
        if package:config("shared") then
            configs.kind = "shared"
        end
        import("package.tools.xmake").install(package)
        os.cp("*.h", package:installdir("include", "edopro-core"))
        os.cp("RNG", package:installdir("include", "edopro-core"))
        os.cp("lua/src/*.h", package:installdir("include", "edopro-core"))
        -- Chunk-1 overlay: install the .proto source alongside other headers
        -- for reference. Generated .pb.h files land in the xmake build tree
        -- (via add_rules("protobuf.cpp")), not in source, so they don't need
        -- to be mirrored here.
        if os.isdir("serialize") then
            os.cp("serialize/*.proto", package:installdir("include", "edopro-core", "serialize"))
            if os.isfile("serialize/Makefile") then
                os.cp("serialize/Makefile", package:installdir("include", "edopro-core", "serialize"))
            end
        end
    end)
package_end()