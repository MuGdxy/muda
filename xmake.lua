set_project("muda")


-- **********************************
-- 
-- all options
-- 
-- **********************************
option("eigen_dir")
set_default("default")
set_showmenu(true)
set_description("user defined eigen directory. if you want to use your own eigen, you should set this option to your eigen directory.")
option_end()

option("example")
set_default(true)
set_showmenu(true)
set_description("build muda examples. if you want to see how to use muda, you could enable this option.")
option_end()

option("test")
set_default(false)
set_showmenu(true)
set_description("build muda test. if you're the developer, you should enable this option.")
option_end()

option("playground")
set_default(false)
set_showmenu(true)
set_description("build muda playground. if you're the developer, you could enable this option.")
option_end()

option("dev")
set_default(false)
set_showmenu(true)
set_description("build muda example, playground and test. if you're the developer, you could enable this option.")
option_end()

option("core-only")
set_default(false)
set_showmenu(true)
set_description("only include the core functionality of muda.")
option_end()

option("gui")
set_default(true)
set_showmenu(true)
set_description("include gui support in cuda.")
option_end()

-- definitions for EASTL
add_defines("_CHAR16T")
add_defines("_CRT_SECURE_NO_WARNINGS")
add_defines("_SCL_SECURE_NO_WARNINGS")
add_defines("EASTL_OPENSOURCE=1")


-- **********************************
-- 
-- packages requirement
-- 
-- **********************************

add_requires("glfw", {optional = true})
add_requires("cuda", {optional = false})

-- **********************************
-- 
-- targets
-- 
-- **********************************
set_languages("cxx17")
add_rules("mode.debug", "mode.release")


target("muda")
    add_undefines("min","max")
    set_kind("headeronly")
    add_headerfiles("src/core/muda/**.h","src/core/muda/**.inl",{public = true})
    add_includedirs("src/core/", {public = true})

    if(is_config("eigen_dir", "default")) then
        add_headerfiles("external/default/**",{public = true})
        add_includedirs("external/default/", {public = true})
    else
        add_includedirs(get_config("eigen_dir"), {public = true})
    end

    add_cuflags("--extended-lambda", {public = true}) -- must be set for muda paradigm
    add_cuflags("--expt-relaxed-constexpr", {public = true})
target_end()

function add_h_and_inl(folder)
    add_headerfiles(folder.."**.h", folder.."**.inl")
end


if(not has_config("core-only")) then
    -- base ext-target, do nothing but add the include dirs
    target("muda-ext")
        add_deps("muda")
        set_kind("headeronly")
        add_includedirs("src/ext/",{public=true})
    target_end()

    -- this target includes thrust support, typically device_vector, device_var, device_buffer and so on.
    target("muda-buffer") 
        add_deps("muda-ext")
        set_kind("headeronly")
        add_h_and_inl("src/ext/muda/buffer") -- eg. device_buffer
        add_h_and_inl("src/ext/muda/container/") -- eg. device_vector/device_var/...
        add_h_and_inl("src/ext/muda/composite/") -- eg. host cse data structure
        add_headerfiles("src/ext/muda/buffer.h", "src/ext/muda/container.h")
    target_end()

    -- this target includes thread only container extension, typically thread_only::property_queue and so on.
    target("muda-thread-only")
        add_deps("muda-ext")
        set_kind("headeronly")
        add_h_and_inl("src/ext/muda/thread_only/")
        add_includedirs("src/ext/muda/thread_only", {public = true}) -- EASTL requirement
        add_includedirs("src/ext/muda/thread_only/EABase/include/common", {public = true}) -- EASTL requirement
    target_end()

    -- this target includes cublas cusparse support, typically sparse matrix and so on.
    target("muda-blas")
        add_deps("muda-buffer") -- need muda-buffer to create temp buffer
        set_kind("headeronly")
        add_h_and_inl("src/ext/muda/blas/")
    target_end()

    -- this target includes cuda CUB wrapper support, typically parallel scan, reduce and so on
    target("muda-algo")
        add_deps("muda-buffer") -- need muda-buffer to create temp buffer
        set_kind("headeronly")
        add_h_and_inl("src/ext/muda/blas/")
    target_end()

    -- this target includes Physically-Based-Animation algorithms
    target("muda-pba")
        add_deps("muda-buffer", "muda-thread-only", "muda-blas", "muda-algo", {public=true}) -- almost all deps
        set_kind("headeronly")
        add_h_and_inl("src/ext/muda/pba/")
    target_end()  

    -- this target includes GUI
    target("muda-gui")
        add_deps("muda","muda-buffer")
        add_packages("glfw", {public = true})
        set_kind("static")
        -- add imgui src
        add_headerfiles("external/imgui/**.h")
        add_files("external/imgui/**.cpp")
        add_includedirs("external/",{public=true})
        -- add muda gui src
        add_files(
            "src/ext/muda/gui/**.cpp",
            "src/ext/muda/gui/**.cu")
        add_headerfiles("src/ext/muda/gui/**.h")
    target_end() 

    -- this is a phony target to collect all muda functionalities which is convenient for quick-starts, examples and tests.
    target("muda-full")
        add_deps(
            "muda-ext",
            "muda-buffer", 
            "muda-thread-only", 
            "muda-blas", 
            "muda-algo", 
            "muda-pba",
            "muda-gui"
        )
        set_kind("headeronly")
    target_end()
end

-- a convenient function to create executable targets
function muda_app_base()
    add_deps("muda-full")
    add_undefines("min","max")
    
    set_kind("binary")
    add_includedirs("test/", {public = false})
    add_headerfiles("src/core/muda/**.h","src/core/muda/**.inl")
    add_headerfiles("src/ext/muda/**.h","src/ext/muda/**.inl")
    add_cugencodes("native")
    add_cugencodes("compute_75")
    add_links("cublas","cusparse")
end

-- test or dev has been defined
if has_config("test") or has_config("dev") then
    target("muda_test")
        muda_app_base()
        test_data_dir = path.absolute("test/data")
        add_defines("MUDA_TEST_DATA_DIR=R\"(".. test_data_dir..")\"")
        add_files("test/muda_test/**.cu","test/muda_test/**.cpp")
end

if has_config("example") or has_config("dev") then
    target("muda_example")
        muda_app_base()
        add_files("example/**.cu","example/**.cpp")
end

if has_config("playground") or has_config("dev") then
    target("muda_pg")
        muda_app_base()
        add_files("test/playground/**.cu","test/playground/**.cpp")
end

