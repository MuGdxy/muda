set_project("muda")

-- options
option("eigen_dir")
    set_default("default")
    set_showmenu(true)
    set_description("user defined eigen directory. if you want to use your own eigen, you should set this option to your eigen directory")
option_end()

option("example")
    set_default(false)
    set_showmenu(true)
    set_description("build muda examples. if you want to see how to use muda, you could enable this option")
option_end()

option("test")
    set_default(false)
    set_showmenu(true)
    set_description("build muda test. if you're the developer, you should enable this option")
option_end()

option("playground")
    set_default(false)
    set_showmenu(true)
    set_description("build muda playground. if you're the developer, you could enable this option")
option_end()

option("dev")
    set_default(false)
    set_showmenu(true)
    set_description("build muda example, playground and test. if you're the developer, you could enable this option")
option_end()

-- definitions for EASTL
add_defines("_CHAR16T")
add_defines("_CRT_SECURE_NO_WARNINGS")
add_defines("_SCL_SECURE_NO_WARNINGS")
add_defines("EASTL_OPENSOURCE=1")

-- targets
set_languages("cxx17")
add_rules("mode.debug", "mode.release")
add_requires("cuda", {optional = false})

target("muda")
    add_undefines("min","max")
    set_kind("static")
    add_headerfiles("src/muda/**.h","src/muda/**.inl")
    add_includedirs("src/", {public = true})
    add_includedirs("src/muda/thread_only", {public = true})
    add_includedirs("src/muda/thread_only/EABase/include/common", {public = true})
    add_files("src/muda/PBA/**.cu","src/muda/PBA/**.cpp")

    if(is_config("eigen_dir", "default")) then
        add_headerfiles("external/default/**")
        add_includedirs("external/default/", {public = true})
    else
        add_includedirs(get_config("eigen_dir"), {public = true})
    end

    add_cuflags("--extended-lambda", {public = true}) -- must be set for muda paradigm
    add_cuflags("--expt-relaxed-constexpr", {public = true})

function muda_app_base()
    add_deps("muda")
    add_undefines("min","max")
    
    set_kind("binary")
    add_includedirs("test/", {public = false})
    add_headerfiles("src/muda/**.h","src/muda/**.inl")
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
        add_headerfiles()
end
