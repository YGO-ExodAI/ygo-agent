add_rules("mode.debug", "mode.release")

add_repositories("my-repo repo")

-- SPIKE 2026-04-20: re-enabling edopro-core for the envpool integration
-- spike. The prior disable commit (bab5959) noted a "template with C
-- linkage" error from /usr/include/c++/11/bits/allocated_ptr.h.
add_requires(
    "ygopro-core 0.0.2", "edopro-core exodai-b118d05", "pybind11 2.13.*", "fmt 10.2.*", "glog 0.6.0",
    "sqlite3 3.43.0+200", "concurrentqueue 1.0.4", "unordered_dense 4.4.*",
    "sqlitecpp 3.2.1", "lua",
    -- ExodAI Phase P1 Primitive 1: state serialization runtime. Must stay in
    -- lockstep with the EDOPro client's vcpkg-supplied protobuf (both on the
    -- modern 4.x+/28.x lineage). See phase_p1_primitive_1_plan.md §1.4.
    "protobuf-cpp 28.*")


target("ygopro0_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro0/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro0"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)


target("ygopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core", "lua", "protobuf-cpp")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
        -- Keep debug symbols in release builds. Adds ~MB to .so size but
        -- no runtime cost, and makes gdb backtraces from core dumps
        -- useful when a segfault reaches the C++ engine. Without this,
        -- stack frames in gdb collapse to ?? at the ygopro-core boundary.
        set_symbols("debug")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)

target("edopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/edopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "edopro-core", "protobuf-cpp")
    set_languages("c++17")
    if is_mode("release") then
        -- SPIKE 2026-04-20: LTO disabled on this target while closing the
        -- reset() crash. Hypothesis under test: whole-program LTO elides or
        -- misinlines the `try { } catch (...)` frame around LUAI_TRY in the
        -- Lua amalgam TU, so throws raised inside edo9300's
        -- check_lua_stack_unwinding self-test escape the supposed-protected
        -- region instead of being caught. See docs/edopro_envpool_spike.md §4.2.
        -- set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
        -- SPIKE: keep debug symbols for named gdb backtraces (same policy as ygopro_ygoenv).
        set_symbols("debug")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/edopro"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)


target("alphazero_mcts")
    add_rules("python.library")
    add_files("mcts/mcts/alphazero/*.cpp")
    add_packages("pybind11")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("mcts")

    after_build(function (target)
        local install_target = "$(projectdir)/mcts/mcts/alphazero"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
        os.run("pybind11-stubgen mcts.alphazero.alphazero_mcts -o %s", "$(projectdir)/mcts")
    end)
